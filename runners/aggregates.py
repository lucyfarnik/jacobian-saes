import json
import os
import sys
from abc import ABC
from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from itertools import product
from pprint import pprint
from typing import Callable, Iterable, NamedTuple, Sequence

import baukit  # type: ignore
import einops
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from nnsight import Envoy, LanguageModel  # type: ignore
from simple_parsing import Serializable
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer  # type: ignore
from transformer_lens import utils
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from jacobian_saes.sae_pair import SAEPair
from jacobian_saes.training.mlp_with_act_grads import MLPWithActGrads
from jacobian_saes.utils import default_device

# dimensions:
#
#   B: batch size
#   T: sequence/context length (tokens)
#   K: top-k activations
#   I: MLP hidden size (intermediate)
#   L: SAE hidden size (latent)
#


# copied from https://github.com/EleutherAI/sae-auto-interp/blob/9733e9f220c7c6b1546d547edb10f947c0fc5ca7/sae_auto_interp/config.py#L47-L71
@dataclass
class CacheConfig(Serializable):
    dataset_repo: str = "stas/c4-en-10k"
    dataset_split: str = "train"
    dataset_name: str = ""
    dataset_row: str = "text"
    batch_size: int = 1
    ctx_len: int = 64
    n_tokens: int = 10_000
    n_splits: int = 5


@dataclass
class ExamplesConfig:
    model_name: str
    cache: CacheConfig = field(default_factory=CacheConfig)
    dtype: str = "float32"
    seed: int = 42

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.dtype == "float16":
            return torch.float16
        if self.dtype == "float32":
            return torch.float32
        raise ValueError(f"Unsupported dtype: {self.dtype}")

    def __str__(self) -> str:
        out = self.cache.dataset_repo.replace("/", "_")
        out += "," + self.cache.dataset_split.replace("[:", "_").replace("]", "")
        out += ",batch_size=" + str(self.cache.batch_size)
        out += ",ctx_len=" + str(self.cache.ctx_len)
        out += ",tokens=" + str(self.cache.n_tokens)
        return out


# copied from https://github.com/EleutherAI/sae-auto-interp/blob/9733e9f220c7c6b1546d547edb10f947c0fc5ca7/sae_auto_interp/utils.py#L6-L27
def load_tokenized_data(
    ctx_len: int,
    tokenizer: AutoTokenizer | PreTrainedTokenizer | PreTrainedTokenizerFast,
    dataset_repo: str,
    dataset_split: str,
    dataset_name: str = "",
    dataset_row: str = "raw_content",
    seed: int = 22,
):
    data = load_dataset(dataset_repo, name=dataset_name, split=dataset_split)
    tokens_ds = utils.tokenize_and_concatenate(
        data, tokenizer, max_length=ctx_len, column_name=dataset_row  # type: ignore
    )
    tokens_ds = tokens_ds.shuffle(seed)
    return tokens_ds["tokens"]


class TokenDataset(torch.utils.data.Dataset[Tensor]):
    def __init__(self, data: Tensor) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tensor:
        return self.data[idx]


# copied from https://github.com/TransluceAI/observatory/blob/219fb5fadbc9501e2c695678ab7acde5bb72db96/lib/activations/activations/exemplars_computation.py#L19-L31
def collate_fn(batch: list[Tensor], pad_id: int, max_length: int) -> dict[str, Tensor]:
    # TODO: should already be the same length
    lengths = torch.tensor([seq.size(0) for seq in batch])

    padded_batch = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        if lengths[i] > 0:
            padded_batch[i, -lengths[i] :] = seq

    attn_mask = torch.arange(max_length).unsqueeze(0) >= (
        max_length - lengths
    ).unsqueeze(1)
    return {"input_ids": padded_batch, "attention_mask": attn_mask.int()}


def get_dataloader(
    config: ExamplesConfig,
) -> tuple[LanguageModel | Envoy, DataLoader[Tensor]]:
    model = LanguageModel(
        config.model_name,
        device_map=default_device,
        dispatch=True,
        torch_dtype=config.dtype,
    )

    assert isinstance(model.tokenizer.pad_token_id, int)  # type: ignore

    data: Tensor = load_tokenized_data(
        ctx_len=config.cache.ctx_len,
        tokenizer=model.tokenizer,  # type: ignore
        dataset_repo=config.cache.dataset_repo,
        dataset_split=config.cache.dataset_split,
        dataset_row=config.cache.dataset_row,
        seed=config.seed,
    )

    dataloader: DataLoader[Tensor] = DataLoader(
        TokenDataset(data),
        batch_size=config.cache.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(
            collate_fn,
            pad_id=model.tokenizer.pad_token_id,
            max_length=config.cache.ctx_len,
        ),
    )

    return model, dataloader


def find_checkpoints(top: str = "checkpoints") -> list[str]:
    dirpaths: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(top):
        if all(
            filename in filenames
            for filename in [
                "sae_weights.safetensors",
                "cfg.json",
                "sparsity.safetensors",
            ]
        ):
            dirpaths.append(dirpath)
    return dirpaths


# TODO: duplicate of utils.load_pretrained with local instead of wandb artifact
def load_checkpoint(
    dirpath: str, config: ExamplesConfig
) -> tuple[SAEPair, MLPWithActGrads]:
    sae_pair = SAEPair.load_from_pretrained(
        dirpath, device=default_device, dtype=config.dtype
    )

    model = HookedTransformer.from_pretrained(  # type: ignore
        sae_pair.cfg.model_name, device=default_device, dtype=config.dtype
    )

    mlp: torch.nn.Module = model.blocks[sae_pair.cfg.hook_layer].mlp
    mlp_with_act_grads = MLPWithActGrads(mlp.cfg)
    mlp_with_act_grads.load_state_dict(mlp.state_dict())
    mlp_with_act_grads.to(device=default_device, dtype=config.torch_dtype)

    # TODO: don't load the model twice (nnsight and transformerlens)
    del model

    if sae_pair.cfg.model_name != config.model_name.split("/")[-1]:
        raise ValueError(f"{dirpath} is not {sae_pair.cfg.model_name}")

    return sae_pair, mlp_with_act_grads


def get_mlp_acts_func(
    model: LanguageModel | Envoy, layer_idx: int
) -> Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]:
    def get_mlp_acts(input_ids: Tensor, attn_mask: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            with model.trace(
                {"input_ids": input_ids, "attention_mask": attn_mask}  # type: ignore
            ):
                mlp_acts_in = model.gpt_neox.layers[
                    layer_idx
                ].post_attention_layernorm.output.save()

                # TODO: not needed, we use MLPWithActGrads instead
                mlp_acts_out = model.gpt_neox.layers[layer_idx].mlp.output.save()

        return mlp_acts_in, mlp_acts_out  # type: ignore

    return get_mlp_acts


# TODO: duplicate of utils.get_jacobian without repeating matmuls
def get_jacobian_func(
    sae_pair: SAEPair, mlp: MLPWithActGrads
) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
    w_dec_in_LI = sae_pair.get_W_dec(is_output_sae=False) @ mlp.W_in
    w_out_enc_LI = (mlp.W_out @ sae_pair.get_W_enc(is_output_sae=True)).permute(1, 0)

    def get_jacobian(
        mlp_act_grads_BTI: torch.Tensor,
        sae_indices_in_BTK: torch.Tensor,
        sae_indices_out_BTK: torch.Tensor,
    ) -> torch.Tensor:
        return einops.einsum(
            w_dec_in_LI[sae_indices_in_BTK],
            mlp_act_grads_BTI,
            w_out_enc_LI[sae_indices_out_BTK],
            "B T K1 I, B T I, B T K2 I -> B T K1 K2",
        )

    return get_jacobian


class Activations(NamedTuple):
    input_ids_BT: Tensor
    mlp_acts_in_BTI: Tensor
    mlp_acts_out_BTI: Tensor
    mlp_act_grads_BTI: Tensor
    sae_acts_in_BTL: Tensor
    sae_indices_in_BTK: Tensor
    sae_acts_out_BTL: Tensor
    sae_indices_out_BTK: Tensor
    jacobian_BTKK: Tensor


class Stat(ABC):
    def add(self, step: int, acts: Activations) -> None:
        raise NotImplementedError

    def save(self, examples_config: ExamplesConfig, checkpoint_dirpath: str) -> pd.DataFrame:  # type: ignore
        raise NotImplementedError


class Examples(Stat):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        sae_index_in: int,
        sae_index_out: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.sae_index_in = sae_index_in
        self.sae_index_out = sae_index_out

        self.sae_acts_in_max = 0.0
        self.sae_acts_out_max = 0.0
        self.jacobian_max = 0.0

        self.data = []

    def __getitem__(self, key: str) -> float:
        return getattr(self, key)

    def add(self, step: int, acts: Activations) -> None:
        tokens_BT = [
            self.tokenizer.convert_ids_to_tokens(batch.tolist())
            for batch in acts.input_ids_BT
        ]

        sae_acts_in_BT = acts.sae_acts_in_BTL[:, :, self.sae_index_in]
        sae_acts_out_BT = acts.sae_acts_out_BTL[:, :, self.sae_index_out]

        self.sae_acts_in_max = max(self.sae_acts_in_max, sae_acts_in_BT.max().item())
        self.sae_acts_out_max = max(self.sae_acts_out_max, sae_acts_out_BT.max().item())
        self.jacobian_max = max(self.jacobian_max, acts.jacobian_BTKK.max().item())

        mask_in_BTK = acts.sae_indices_in_BTK == self.sae_index_in
        mask_out_BTK = acts.sae_indices_out_BTK == self.sae_index_out
        mask_BTKK = mask_in_BTK[:, :, :, None] & mask_out_BTK[:, :, None, :]

        rows = {}
        for (
            batch_index,
            token_index,
            topk_index_in,
            topk_index_out,
        ) in mask_BTKK.nonzero():
            batch_index_ = batch_index.item()
            token_index_ = token_index.item()

            if batch_index_ not in rows:
                rows[batch_index_] = {
                    "step": step,
                    "batch_index": batch_index_,
                    "tokens": tokens_BT[batch_index],
                    "jacobian": [0.0 for _ in range(len(tokens_BT[batch_index]))],
                    "sae_acts_in": sae_acts_in_BT[batch_index].tolist(),
                    "sae_acts_out": sae_acts_out_BT[batch_index].tolist(),
                }
            rows[batch_index_]["jacobian"][token_index_] = acts.jacobian_BTKK[
                batch_index, token_index, topk_index_in, topk_index_out
            ].item()
        for row in rows.values():
            self.data.append(row)

    def save(self, examples_config: ExamplesConfig, checkpoint_dirpath: str):
        for k in ["jacobian", "sae_acts_in", "sae_acts_out"]:
            for row in self.data:
                row[f"{k}_max"] = max(row[k])
                row[f"{k}_norm"] = json.dumps([x / self[f"{k}_max"] for x in row[k]])
                row[k] = json.dumps(row[k])

        filename = (
            f"ex-{self.sae_index_in}v{self.sae_index_out}-_{str(examples_config)}.csv"
        )
        dataframe = pd.DataFrame(self.data)
        dataframe = dataframe.sort_values("jacobian_max", ascending=False)
        dataframe.to_csv(os.path.join(checkpoint_dirpath, filename), index=False)
        return dataframe


class Pair(Stat):
    def __init__(self) -> None:
        self.counter = Counter[tuple[int, int]]()
        self.values = dict[tuple[int, int], list[float]]()

    def add(self, step: int, acts: Activations) -> None:
        _B, _T, K, _ = acts.jacobian_BTKK.shape
        for i, j in product(range(K), range(K)):
            sae_indices_in: list[int] = acts.sae_indices_in_BTK[:, :, i].flatten().tolist()  # type: ignore
            sae_indices_out: list[int] = acts.sae_indices_out_BTK[:, :, j].flatten().tolist()  # type: ignore
            self.counter.update(zip(sae_indices_in, sae_indices_out))

            for sae_index_in, sae_index_out, value in zip(
                sae_indices_in,
                sae_indices_out,
                acts.jacobian_BTKK[:, :, i, j].flatten().tolist(),
            ):
                if (sae_index_in, sae_index_out) not in self.values:
                    self.values[(sae_index_in, sae_index_out)] = []
                self.values[(sae_index_in, sae_index_out)].append(value)

    def save(self, examples_config: ExamplesConfig, checkpoint_dirpath: str):
        i, j = zip(*self.counter.keys())
        dataframe = pd.DataFrame(
            {
                "i": i,
                "j": j,
                "count": self.counter.values(),
                **get_stats(self.values.values()),
            }
        ).sort_values("count", ascending=False)
        dataframe.to_csv(
            os.path.join(checkpoint_dirpath, f"pair_{str(examples_config)}.csv"),
            index=False,
        )
        return dataframe


def get_stats(xs: Iterable[list[float]]) -> dict[str, list[float]]:
    stats = {"mean": [], "std": [], "abs_mean": [], "abs_std": []}
    for x in xs:
        stats["mean"].append(np.mean(x))
        stats["std"].append(np.std(x))
        stats["abs_mean"].append(np.mean(np.abs(x)))
        stats["abs_std"].append(np.std(np.abs(x)))
    return stats  # type: ignore


class Covariance(Stat):
    def __init__(self) -> None:
        self.covariance = baukit.Covariance()

    def add(self, step: int, acts: Activations) -> None:
        B, T, K, _ = acts.jacobian_BTKK.shape

        # select top-k SAE activations
        sae_values_in_BTK = acts.sae_acts_in_BTL.gather(-1, acts.sae_indices_in_BTK)
        sae_values_out_BTK = acts.sae_acts_out_BTL.gather(-1, acts.sae_indices_out_BTK)

        # expand activations to match Jacobian shape
        sae_values_in_BTKK = sae_values_in_BTK[:, :, :, None].expand(B, T, K, K)
        sae_values_out_BTKK = sae_values_out_BTK[:, :, None, :].expand(B, T, K, K)

        # construct samples of Jacobian and input/output SAE activations
        samples = torch.stack(
            [
                sae_values_in_BTKK.flatten(),  # i-th input SAE activation
                sae_values_out_BTKK.flatten(),  # j-th output SAE activation
                acts.jacobian_BTKK.flatten(),  # (i,j)-th Jacobian element
            ],
            dim=-1,
        )

        self.covariance.add(samples)  # type: ignore

    def save(self, examples_config: ExamplesConfig, checkpoint_dirpath: str):
        filename = f"stats_{str(examples_config)}.csv"
        labels = ["sae_in", "sae_out", "jacobian"]
        n_labels = len(labels)

        data1: dict[str, Sequence[str | float]] = {
            "x": labels,
            "mean": self.covariance.mean().tolist(),  # type: ignore
            "variance": self.covariance.variance().tolist(),  # type: ignore
            "stdev": self.covariance.stdev().tolist(),  # type: ignore
        }
        pprint(data1)
        # pd.DataFrame(data1).to_csv(os.path.join(dirpath, filename), index=False)

        ij = [(i, j) for i in range(n_labels) for j in range(n_labels)]
        data2: dict[str, list[str | float]] = {
            "x": [labels[i] for i, _ in ij],
            "y": [labels[j] for _, j in ij],
        }
        for k, v in {  # type: ignore
            "covariance": self.covariance.covariance().detach().cpu(),  # type: ignore
            "correlation": self.covariance.correlation().detach().cpu(),  # type: ignore
        }.items():
            data2[k] = [v[i, j].item() for i, j in ij]  # type: ignore

        pprint(data2)
        dataframe2 = pd.DataFrame(data2)
        dataframe2.to_csv(os.path.join(checkpoint_dirpath, filename), index=False)
        return dataframe2


def get_example_aggregates(
    checkpoint_dirpath: str, model: LanguageModel | Envoy
) -> list[Examples]:
    pair_filename = next(
        (
            filename
            for filename in os.listdir(checkpoint_dirpath)
            if filename.startswith("pair_")
        )
    )
    sae_indices = [
        (int(i), int(j))
        for i, j in pd.read_csv(os.path.join(checkpoint_dirpath, pair_filename))
        .sort_values("count", ascending=False)
        .head(100)[["i", "j"]]
        .values
    ]
    return [
        Examples(model.tokenizer, sae_index_in, sae_index_out)  # type: ignore
        for sae_index_in, sae_index_out in sae_indices
    ]


def main(checkpoint_dirpath: str) -> None:
    with open(os.path.join(checkpoint_dirpath, "cfg.json"), "r") as f:
        cfg = json.load(f)
    model_name = cfg["model_name"]
    run_name = cfg["run_name"]

    examples_config = ExamplesConfig(
        model_name=f"EleutherAI/{model_name}", cache=CacheConfig()
    )

    model, dataloader = get_dataloader(examples_config)

    # aggregates = get_example_aggregates(checkpoint_dirpath, model)
    aggregates: list[Stat] = [Pair()]

    sae_pair, mlp_with_act_grads = load_checkpoint(checkpoint_dirpath, examples_config)

    get_mlp_acts = get_mlp_acts_func(model, sae_pair.cfg.hook_layer)
    get_jacobian = get_jacobian_func(sae_pair, mlp_with_act_grads)

    for batch_index, batch in tqdm(
        enumerate(dataloader),
        desc=f"layer={sae_pair.cfg.hook_layer}",
        total=len(dataloader),
    ):
        # collect MLP activations
        mlp_acts_in_BTI, mlp_acts_out_BTI = get_mlp_acts(
            batch["input_ids"], batch["attention_mask"]
        )

        # TODO: duplicate of utils.run_sandwich

        # compute SAE activations and MLP gradients
        sae_acts_in_BTL, sae_indices_in_BTK = sae_pair.encode(
            mlp_acts_in_BTI, is_output_sae=False, return_topk_indices=True  # type: ignore
        )

        mlp_acts_out_BTI, mlp_act_grads_BTI = mlp_with_act_grads.forward(
            mlp_acts_in_BTI
        )
        mlp_act_grads_BTI = mlp_act_grads_BTI.detach()

        with torch.no_grad():
            sae_acts_out_BTL, sae_indices_out_BTK = sae_pair.encode(
                mlp_acts_out_BTI, is_output_sae=True, return_topk_indices=True  # type: ignore
            )

            # compute Jacobian between SAE activations
            jacobian_BTKK = get_jacobian(
                mlp_act_grads_BTI,
                sae_indices_in_BTK,
                sae_indices_out_BTK,
            )

        activations = Activations(
            input_ids_BT=batch["input_ids"],
            mlp_acts_in_BTI=mlp_acts_in_BTI,
            mlp_acts_out_BTI=mlp_acts_out_BTI,
            mlp_act_grads_BTI=mlp_act_grads_BTI,
            sae_acts_in_BTL=sae_acts_in_BTL,
            sae_indices_in_BTK=sae_indices_in_BTK,
            sae_acts_out_BTL=sae_acts_out_BTL,
            sae_indices_out_BTK=sae_indices_out_BTK,
            jacobian_BTKK=jacobian_BTKK,
        )

        for aggregate in aggregates:
            aggregate.add(batch_index, activations)

    for i, aggregate in enumerate(aggregates):
        dataframe = aggregate.save(examples_config, checkpoint_dirpath)  # type: ignore
        dataframe.to_csv(f"stats{i}_{run_name}.csv", index=False)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", "-p", type=str)
    args = parser.parse_args()
    main(args.path)
