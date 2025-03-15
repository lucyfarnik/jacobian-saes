import argparse
import math
import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch

from jacobian_saes import LanguageModelSAERunnerConfig, SAETrainingRunner
from jacobian_saes.training.sparsity_metrics import sparsity_metrics

d_model_by_size = {
    "14m": 128,
    "31m": 256,
    "70m": 512,
    "160m": 768,
    "gpt2-small": 768,
    "410m": 1024,
    "1b": 2048,
    "1.4b": 2048,
    "2.8b": 2560,
    "6.9b": 4096,
    "12b": 5120,
}

n_layers_by_size = {
    "14m": 6,
    "31m": 6,
    "70m": 6,
    "160m": 12,
    "gpt2-small": 12,
    "410m": 24,
    "1b": 16,
    "1.4b": 24,
    "2.8b": 32,
    "6.9b": 32,
    "12b": 36,
}

parser = argparse.ArgumentParser(description="Train a Jacobian SAE")
parser.add_argument(
    "--accum",
    "--accumulation-steps",
    dest="gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of gradient accumulation steps",
)
parser.add_argument(
    "--always-eval",
    action="store_true",
    help="Run evaluations and wandb logging at every training step (only for debugging)",
)
parser.add_argument("--batch-size", "-b", type=int, default=4096, help="Batch size")
parser.add_argument(
    "--buffer-size",
    type=int,
    default=32,
    help="Buffer size (number of batches in buffer)",
)
parser.add_argument("--context-size", "-c", type=int, default=2048, help="Context size")
parser.add_argument(
    "--expansion-factor", "-e", type=int, default=32, help="Expansion factor"
)
parser.add_argument("--eval-batch-size", type=int, default=8, help="Eval batch size")
parser.add_argument(
    "--eval-n-batches", type=int, default=10, help="Number of eval batches"
)
parser.add_argument(
    "--jacobian-coef", "-j", type=float, default=1, help="Jacobian coefficient"
)
parser.add_argument("-k", type=int, default=32, help="TopK value")
parser.add_argument("--layer", "-l", type=int, default=3, help="Layer to hook")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument(
    "--model-size",
    "-m",
    type=str,
    default="70m",
    help="Pythia model size",
    choices=d_model_by_size.keys(),
)
parser.add_argument(
    "--model-preset",
    type=str,
    help="Pythia model size; also sets other arguments (e.g. Jacobian coef). "
    + "Can cause messy behavior when combined with other args; "
    + "not recommended unless you've read the source code.",
    choices=d_model_by_size.keys(),
)
parser.add_argument(
    "--out-mse-coef",
    dest="mlp_out_mse_coefficient",
    type=float,
    default=1.0,
    help="Coefficient for the post-MLP MSE",
)
parser.add_argument(
    "-p", "--precision", type=int, default=32, help="Floating point precision"
)
parser.add_argument(
    "-r",
    "--randomize-weights",
    action="store_true",
    help="Randomize the weights of the LLM",
)
parser.add_argument(
    "-s",
    "--sparsity-metric",
    type=str,
    choices=sparsity_metrics.keys(),
    default="l1",
    help="Jacobian sparsity metric",
)
parser.add_argument("--store-batch-size", type=int, default=16, help="Store batch size")
parser.add_argument(
    "--tokens",
    "-t",
    type=float,
    default=300_000_000,
    help="Total number of training tokens",
)
parser.add_argument(
    "--wandb-project",
    "-w",
    type=str,
    default="jacobian_saes_test",
    help="Wandb project name",
)
parser.add_argument(
    "--model-deduped",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Whether to use the deduped version of the model (default: True if a deduped version of this model is known to exist).",
)

args = parser.parse_args()

if args.model_deduped is None:
    args.model_deduped = args.model_size not in ["14m", "31m", "gpt2-small"]

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if args.model_preset is not None:
    if args.model_preset == "410m":
        args.model_size = "410m"
        if args.expansion_factor == 32:
            args.expansion_factor = 64
        if args.jacobian_coef == 1:
            # jac coeff should imo be 0.5 k^2 / d_model
            args.jacobian_coef = 0.5
        if args.eval_batch_size == 8:
            args.eval_batch_size = 4
        if args.eval_n_batches == 10:
            args.eval_n_batches = 20

    elif args.model_preset == "1b":
        args.model_size = "1b"
        if args.jacobian_coef == 1:
            args.jacobian_coef = 0.25
        if args.eval_batch_size == 8:
            args.eval_batch_size = 2
        if args.eval_n_batches == 10:
            args.eval_n_batches = 40

    elif args.model_preset == "2.8b":
        args.model_size = "2.8b"
        if args.jacobian_coef == 1:
            args.jacobian_coef = 0.2
        if args.eval_batch_size == 8:
            args.eval_batch_size = 2
        if args.eval_n_batches == 10:
            args.eval_n_batches = 40
        if args.buffer_size == 32:
            args.buffer_size = 8
        if args.store_batch_size == 16:
            args.store_batch_size = 8

    elif args.model_preset == "6.9b":
        args.model_size = "6.9b"
        if args.jacobian_coef == 1:
            args.jacobian_coef = 0.125
        if args.buffer_size == 32:
            args.buffer_size = 2
        if args.store_batch_size == 16:
            args.store_batch_size = 1
        if args.eval_batch_size == 8:
            args.eval_batch_size = 1
        if args.eval_n_batches == 10:
            args.eval_n_batches = 80
        if args.batch_size == 4096:
            args.batch_size = 2048
        if args.gradient_accumulation_steps == 1:
            args.gradient_accumulation_steps = 2
        if args.context_size == 2048:
            args.context_size = 256

    else:
        raise ValueError(f"Model preset not yet supported: {args.model_preset}")

model_name_suffix = "-deduped" if args.model_deduped else ""
model_name = (
    "gpt2-small"
    if args.model_size == "gpt2-small"
    else f"pythia-{args.model_size}{model_name_suffix}"
)

dataset_path = "apollo-research/monology-pile-uncopyrighted-tokenizer-" + (
    "gpt2" if args.model_size == "gpt2-small" else "EleutherAI-gpt-neox-20b"
)

if args.model_preset == "gpt2-small" and args.context_size > 1024:
    args.context_size = 1024


total_training_tokens = int(args.tokens)
batch_size = args.batch_size
total_training_steps = math.ceil(total_training_tokens / batch_size)
total_training_tokens = total_training_steps * batch_size  # to get the exact number

lr_warm_up_steps = total_training_steps // 100  # 1% of training
lr_decay_steps = total_training_steps // 5  # 20% of training
jacobian_warm_up_steps = total_training_steps // 20  # 5% of training

assert (
    args.layer < n_layers_by_size[args.model_size]
), f"Layer {args.layer} does not exist in model size {args.model_size}"

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    # model options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html
    model_name=model_name,
    randomize_llm_weights=args.randomize_weights,
    hook_name=f"blocks.{args.layer}.ln2.hook_normalized",
    hook_layer=args.layer,
    d_in=d_model_by_size[args.model_size],
    activation_fn="topk",
    activation_fn_kwargs={"k": args.k},
    dataset_path=dataset_path,
    streaming=True,
    # SAE Parameters
    expansion_factor=args.expansion_factor,
    b_dec_init_method="zeros",
    apply_b_dec_to_input=False,
    normalize_sae_decoder=True,
    # scale_sparsity_penalty_by_decoder_norm=True,
    # decoder_heuristic_init=True,
    init_encoder_as_decoder_transpose=True,
    # Training Parameters
    lr=args.lr,
    adam_beta1=0.9,
    adam_beta2=0.999,
    lr_scheduler_name="constant",  # constant learning rate with warmup. Could be better schedules out there.
    lr_warm_up_steps=lr_warm_up_steps,  # this can help avoid too many dead features initially.
    lr_decay_steps=lr_decay_steps,  # this will help us avoid overfitting.
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    l1_coefficient=0,  # we're using TopK so we don't need this
    use_jacobian_loss=True,
    sparsity_metric=args.sparsity_metric,
    jacobian_coefficient=args.jacobian_coef,
    jacobian_warm_up_steps=jacobian_warm_up_steps,
    mlp_out_mse_coefficient=args.mlp_out_mse_coefficient,
    train_batch_size_tokens=batch_size,
    context_size=args.context_size,
    # Activation Store Parameters
    n_batches_in_buffer=args.buffer_size,  # controls how many activations we store / shuffle.
    training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
    store_batch_size_prompts=args.store_batch_size,
    eval_batch_size_prompts=args.eval_batch_size,
    n_eval_batches=args.eval_n_batches,
    # Resampling protocol
    use_ghost_grads=False,  # we don't use ghost grads anymore.
    feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
    dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
    dead_feature_threshold=1e-6,  # would effect resampling or ghost grads if we were using it.
    # WANDB
    log_to_wandb=True,
    wandb_project=args.wandb_project,
    wandb_log_frequency=(1 if args.always_eval else 30),
    eval_every_n_wandb_logs=(1 if args.always_eval else 20),
    # Misc
    device=device,
    seed=42,
    n_checkpoints=0,
    checkpoint_path="checkpoints",
    dtype=f"float{args.precision}",
    autocast=(device == "cuda"),
    autocast_lm=(device == "cuda"),
)

sparse_autoencoder = SAETrainingRunner(cfg).run()
