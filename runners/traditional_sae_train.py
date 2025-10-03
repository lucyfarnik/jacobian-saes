# Used to establish a training performance baseline

import argparse
import math
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import torch

from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner

parser = argparse.ArgumentParser(description="Train a traditional SAE")
parser.add_argument("--batch-size", "-b", type=int, default=4096, help="Batch size")
parser.add_argument("--buffer-size", type=int, default=32, help="Buffer size (number of batches in buffer)")
parser.add_argument("--context-size", "-c", type=int, default=2048, help="Context size")
parser.add_argument(
    "--expansion-factor", "-e", type=int, default=32, help="Expansion factor"
)
parser.add_argument("--eval-batch-size", type=int, default=8, help="Eval batch size")
parser.add_argument("-k", type=int, default=32, help="TopK value")
parser.add_argument("--layer", "-l", type=int, default=3, help="Layer to hook")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument(
    "--model-size",
    "-m",
    type=str,
    default="70m",
    help="Pythia model size",
    choices=["14m", "31m", "70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "12b"],
)
parser.add_argument("-p", "--precision", type=int, default=32, help="Floating point precision")
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
args = parser.parse_args()

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Using device:", device)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


total_training_tokens = int(args.tokens)
batch_size = args.batch_size
total_training_steps = math.ceil(total_training_tokens / batch_size)
total_training_tokens = total_training_steps * batch_size  # to get the exact number

lr_warm_up_steps = total_training_steps // 100  # 1% of training
lr_decay_steps = total_training_steps // 5  # 20% of training

d_model_by_size = {
    "14m": 128,
    "31m": 256,
    "70m": 512,
    "160m": 768,
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
    "410m": 24,
    "1b": 16,
    "1.4b": 24,
    "2.8b": 32,
    "6.9b": 32,
    "12b": 36,
}
assert (
    args.layer < n_layers_by_size[args.model_size]
), f"Layer {args.layer} does not exist in model size {args.model_size}"

cfg = LanguageModelSAERunnerConfig(
    # Data Generating Function (Model + Training Distibuion)
    # model options here: https://neelnanda-io.github.io/TransformerLens/generated/model_properties_table.html
    model_name=f"pythia-{args.model_size}-deduped",
    hook_name=f"blocks.{args.layer}.ln2.hook_normalized",
    hook_layer=args.layer,
    d_in=d_model_by_size[args.model_size],
    activation_fn="topk",
    activation_fn_kwargs={"k": args.k},
    dataset_path="apollo-research/monology-pile-uncopyrighted-tokenizer-EleutherAI-gpt-neox-20b",
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
    # l1_coefficient=0,  # we're using TopK so we don't need this
    l1_coefficient=1e-9,  # SAE Lens is silly and crashes if this is zero
    train_batch_size_tokens=batch_size,
    context_size=args.context_size,
    # Activation Store Parameters
    n_batches_in_buffer=args.buffer_size,  # controls how many activations we store / shuffle.
    training_tokens=total_training_tokens,  # 100 million tokens is quite a few, but we want to see good stats. Get a coffee, come back.
    store_batch_size_prompts=args.store_batch_size,
    eval_batch_size_prompts=args.eval_batch_size,
    # Resampling protocol
    use_ghost_grads=False,  # we don't use ghost grads anymore.
    feature_sampling_window=1000,  # this controls our reporting of feature sparsity stats
    dead_feature_window=1000,  # would effect resampling or ghost grads if we were using it.
    dead_feature_threshold=1e-6,  # would effect resampling or ghost grads if we were using it.
    # WANDB
    log_to_wandb=True,
    wandb_project=args.wandb_project,
    wandb_log_frequency=30,
    eval_every_n_wandb_logs=20,
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
