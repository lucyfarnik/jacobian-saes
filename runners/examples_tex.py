import ast
import os
import re
import unicodedata
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

COLORS = {
    "jacobian_positive": "Cyan",
    "jacobian_negative": "RedOrange",
    "sae_acts_in": "Green",
    "sae_acts_out": "Magenta",
}

CATEGORIES = {
    "jacobian": "Jacobian",
    "sae_acts_in": "Input SAE",
    "sae_acts_out": "Output SAE",
}

TOK_SPECIAL_CHARS = {
    "âĢĶ": "—",
    "âĢĵ": "–",
    "âĢĭ": "",
    "âĢľ": '"',
    "âĢĿ": '"',
    "âĢĺ": "'",
    "âĢĻ": "'",
    "Ġ": " ",
    "Ċ": "\n",
    "ĉ": "\t",
}

TEX_SPECIAL_CHARS = {
    "_": "\\_",
    "&": "\\&",
    "%": "\\%",
    "#": "\\#",
    "{": "\\{",
    "}": "\\}",
    "$": "\\$",
    "^": "\\textasciicircum{}",
    "~": "\\textasciitilde{}",
    "<": "\\textless{}",
    ">": "\\textgreater{}",
    "|": "\\textbar{}",
    "*": "\\textasteriskcentered{}",
    "`": "\\textasciigrave{}",
    "'": "\\textquotesingle{}",
    '"': "\\textquotedbl{}",
}


def clean_token_for_latex(token: str) -> str:
    token = token.replace("\\", "\\textbackslash{}")
    for char, repl in TOK_SPECIAL_CHARS.items():
        token = token.replace(char, repl)
    for char, repl in TEX_SPECIAL_CHARS.items():
        token = token.replace(char, repl)
    token = token.replace("Ġ", " ").replace("Ċ", "\\newline ")

    cleaned = ""
    for char in token:
        if ord(char) < 128:
            cleaned += char
        else:
            try:
                cleaned += (
                    unicodedata.normalize("NFKD", char)
                    .encode("ascii", "ignore")
                    .decode("ascii")
                )
            except:  # noqa: E722
                cleaned += " "
    return cleaned


def generate_pair_examples_tex(
    filepath: str, max_rows: int | None = None
) -> str | None:
    info = parse_filename_pair(prettify_filename_pair(filepath))
    if info is None:
        raise ValueError(
            f"Failed to parse filename: {prettify_filename_pair(filepath)}"
        )
    else:
        caption = (
            f"{info.model_name}, layer {info.layer}. "  #
            f"Input SAE latent index {info.sae_index_in}. "  #
            f"Output SAE latent index {info.sae_index_out}. "  #
            f"Jacobian elements sorted by {info.stat}."
        )

    df = pd.read_csv(filepath)
    df = df.sort_values("jacobian_max", ascending=False)

    MAX_VALS = {
        "jacobian": float(df["jacobian_max"].max()),
        "sae_acts_in": float(df["sae_acts_in_max"].max()),
        "sae_acts_out": float(df["sae_acts_out_max"].max()),
    }

    if max_rows is not None:
        if len(df) < max_rows:
            return None
        df = df.head(max_rows)

    for col in [
        "tokens",
        "jacobian",
        "sae_acts_in",
        "sae_acts_out",
    ]:
        df[col] = df[col].apply(ast.literal_eval)

    def get_color(value: float, type_name: str) -> str:
        max_val = MAX_VALS[type_name]
        if type_name == "jacobian":
            abs_max = max(abs(max_val), abs(value))
            normalized = abs(float(value)) / abs_max if abs_max != 0 else 0
            color_key = "jacobian_positive" if value >= 0 else "jacobian_negative"
            return f"{COLORS[color_key]}!{normalized * 100:.3f}"
        else:
            normalized = float(value) / max_val if max_val != 0 else 0
            return f"{COLORS[type_name]}!{normalized * 100:.3f}"

    latex_output = [
        "\\begin{figure}",
        "\\centering",
        "\\begin{longtable}{lrl}",
        "\\toprule",
        "Category & Max. abs. value & Example tokens \\\\",
        "\\midrule",
    ]

    for _, row in df.iterrows():
        tokens = row["tokens"]

        for value_col in ["jacobian", "sae_acts_in", "sae_acts_out"]:
            values = row[value_col]
            if value_col == "jacobian":
                max_value = max(
                    abs(max(float(val) for val in values)),
                    abs(min(float(val) for val in values)),
                )
            else:
                max_value = max(float(val) for val in values)

            colored_tokens = []
            for token, val in zip(tokens, values):
                if token == "<|endoftext|>":
                    continue

                clean_token = clean_token_for_latex(token)
                if clean_token.strip():
                    color = get_color(float(val), value_col)
                    colored_tokens.append(
                        f"\\colorbox{{{color}}}{{\\strut {clean_token}}}"
                    )

            tokens_str = " ".join(colored_tokens)

            latex_output.append(
                f"{CATEGORIES[value_col]} & \\num{{{max_value:.3e}}} & {tokens_str} \\\\"
            )

            if value_col == "sae_acts_out":
                latex_output.append("\\midrule")

    latex_output.pop()  # remove last \midrule
    latex_output.extend(
        [
            "\\bottomrule",
            "\\end{longtable}",
            f"\\caption{{{caption}}}",
            "\\end{figure}",
        ]
    )

    return "\n".join(latex_output)


def generate_latent_examples_tex(
    filepath: str, sae_in: bool, max_rows: int | None = None
) -> str | None:
    info = parse_filename_latent(prettify_filename_latent(filepath))
    if info is None:
        raise ValueError(
            f"Failed to parse filename: {prettify_filename_latent(filepath)}"
        )
    else:
        caption = (
            f"{info.model_name} layer {info.layer}. "  #
            f"{info.sae} SAE latent index {info.index}. "  #
            # f"Jacobian elements sorted by {info.stat}."
        )

    df = pd.read_csv(filepath)
    df = df.sort_values("sae_acts_max", ascending=False)

    if max_rows is not None:
        if len(df) < max_rows:
            return None
        df = df.head(max_rows)

    for col in ["tokens", "sae_acts"]:
        df[col] = df[col].apply(ast.literal_eval)

    type_name = "sae_acts_in" if sae_in else "sae_acts_out"

    def get_color(value: float) -> str:
        max_val = float(df["sae_acts_max"].max())
        normalized = float(value) / max_val if max_val != 0 else 0
        return f"{COLORS[type_name]}!{normalized * 100:.3f}"

    latex_output = [
        "\\begin{subfigure}{\\linewidth}",
        "\\centering",
        "\\begin{longtable}{lr}",
        "\\toprule",
        "Example tokens & Max. activation \\\\",
        "\\midrule",
    ]

    for _, row in df.iterrows():
        tokens = row["tokens"]

        for value_col in ["sae_acts"]:
            values = row[value_col]
            max_value = max(float(val) for val in values)

            colored_tokens = []
            for token, val in zip(tokens, values):
                if token == "<|endoftext|>":
                    continue

                clean_token = clean_token_for_latex(token)
                if clean_token.strip():
                    color = get_color(float(val))
                    colored_tokens.append(
                        f"\\colorbox{{{color}}}{{\\strut {clean_token}}}"
                    )

            tokens_str = " ".join(colored_tokens)

            latex_output.append(f"{tokens_str} & \\num{{{max_value:.3e}}} \\\\")
            latex_output.append("\\midrule")

    latex_output.pop()  # remove last \midrule
    latex_output.extend(
        [
            "\\bottomrule",
            "\\end{longtable}",
            f"\\caption{{{caption}}}",
            "\\end{subfigure}",
        ]
    )

    return "\n".join(latex_output)


def find_examples(top: str = "feature_pairs") -> list[str]:
    dirpaths: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(top):
        for filename in filenames:
            if filename.endswith(".csv"):
                dirpaths.append(os.path.join(dirpath, filename))
    return dirpaths


def prettify_filename_pair(filename: str) -> str:
    return os.path.join(
        "feature_pairs_tex", prettify_filename(filename.replace("feature_pairs_", ""))
    )


def prettify_filename_latent(filename: str) -> str:
    return os.path.join(
        "features_tex", prettify_filename(filename.replace("features_", ""))
    )


def prettify_filename(filename: str) -> str:
    return (
        filename.replace("/", "_")
        .replace("csv", "tex")
        .replace("Layer3-32768-J1-LR5.0e-04-k32-T3.0e+08", "pythia-70m-layer-3")
        .replace("Layer3-32768-J1-LR5.0e-04-Tokens3.0e+08", "pythia-70m-layer-3")
        .replace("Layer7-49152-J1-LR5.0e-04-k32-T3.0e+08", "pythia-160m-layer-7")
        .replace("Layer7-49152-J1-LR5.0e-04-Tokens3.0e+08", "pythia-160m-layer-7")
        .replace("Layer15-65536-J1-LR5.0e-04-k32-T3.0e+08", "pythia-410m-layer-15")
        .replace("Layer15-65536-J1-LR5.0e-04-Tokens3.0e+08", "pythia-410m-layer-15")
        .replace("stas_c4-en-10k,train,", "")
        .replace("examples-", "")
        .replace("batch_size=", "b")
        .replace("ctx_len=", "t")
        .replace(",", "-")
        .replace("_", "-")
    )


@dataclass
class StatsInfo:
    model_name: str
    layer: int
    latents: int


# stats0_Layer3-32768-J1-LR5.0e-04-k32-T3.0e+08_mean.csv
# stats0_Layer7-49152-J1-LR5.0e-04-Tokens3.0e+08_mean.csv
# stats0_Layer15-65536-J1-LR5.0e-04-Tokens3.0e+08_mean.csv
def parse_filename_stats(filename: str) -> StatsInfo | None:
    pattern = r"stats0_Layer(\d+)-(\d+)"
    match = re.match(pattern, os.path.basename(filename))
    if match:
        latents = int(match.group(2))
        return StatsInfo(
            model_name={
                32768: "Pythia-70m",
                49152: "Pythia-160m",
                65536: "Pythia-410m",
            }[latents],
            layer=int(match.group(1)),
            latents=latents,
        )
    return None


@dataclass
class PairInfo:
    model_name: str
    layer: int
    stat: str
    sae_index_in: int
    sae_index_out: int


def parse_filename_pair(filename: str) -> PairInfo | None:
    pattern = (
        r"feature-pairs-(pythia-\d+m)-layer-(\d+)-([\w-]+)-(\d+)-v-(\d+)-b\d+-t\d+\.tex"
    )
    match = re.match(pattern, os.path.basename(filename))
    if match:
        return PairInfo(
            model_name=match.group(1).replace("pythia", "Pythia"),
            layer=int(match.group(2)),
            stat=match.group(3),
            sae_index_in=int(match.group(4)),
            sae_index_out=int(match.group(5)),
        )
    return None


@dataclass
class LatentInfo:
    model_name: str
    layer: int
    stat: str
    sae: str
    index: int


def parse_filename_latent(filename: str) -> LatentInfo | None:
    pattern = (
        r"features-(pythia-\d+m)-layer-(\d+)-([\w-]+)-(in|out)-(\d+)-b\d+-t\d+\.tex"
    )
    match = re.match(pattern, os.path.basename(filename))
    if match:
        return LatentInfo(
            model_name=match.group(1).replace("pythia", "Pythia"),
            layer=int(match.group(2)),
            stat=match.group(3),
            sae=match.group(4).replace("in", "Input").replace("out", "Output"),
            index=int(match.group(5)),
        )
    return None


def generate_tables() -> None:
    max_rows = 12

    os.makedirs("feature_pairs_tex", exist_ok=True)
    for filename in tqdm(find_examples("feature_pairs")):
        latex_table = generate_pair_examples_tex(filename, max_rows)
        if latex_table is None:
            continue
        if "ctx_len=16" in filename:
            for file in [
                filename.replace("csv", "tex"),
                prettify_filename_pair(filename),
            ]:
                with open(file, "w", encoding="utf-8") as f:
                    f.write(latex_table)
        # break  # for testing

    os.makedirs("features_tex", exist_ok=True)
    for filename in tqdm(find_examples("features")):
        sae_in = "examples-in" in filename
        latex_table = generate_latent_examples_tex(filename, sae_in, max_rows)
        if latex_table is None:
            continue
        if "ctx_len=16" in filename:
            for file in [
                filename.replace("csv", "tex"),
                prettify_filename_latent(filename),
            ]:
                with open(file, "w", encoding="utf-8") as f:
                    f.write(latex_table)
        # break  # for testing


def generate_main() -> None:
    texs: list[tuple[str, str, str]] = []
    for dirpath, _dirnames, filenames in os.walk("feature_pairs_tex"):
        for filename in filenames:
            info = parse_filename_pair(filename)
            if info is not None:
                f1 = os.path.join(dirpath, filename)
                prefix = os.path.join(
                    "features_tex",
                    f"features-{info.model_name.lower()}-layer-{info.layer}-{info.stat}",
                )
                f2 = f"{prefix}-in-{info.sae_index_in}-b32-t16.tex"
                f3 = f"{prefix}-out-{info.sae_index_out}-b32-t16.tex"
                texs.append((f1, f2, f3))
    texs = sorted(texs, key=lambda x: x[0])
    with open("main.tex", "w", encoding="utf-8") as f:
        for f1, f2, f3 in texs:
            f.write(f"\\input{{{f1}}}\n")
            f.write("\\clearpage\n")
            f.write(f"\\input{{{f2}}}\n")
            f.write(f"\\input{{{f3}}}\n")
            f.write("\\clearpage\n\n")


def generate_combined(filename_stats: str, max_rows: int = 12) -> None:
    features_dir = filename_stats.replace("stats0_", "features/")
    features_dir = features_dir.replace(".csv", "")
    features_filename = (
        f"{features_dir}/examples-#_stas_c4-en-10k,train,batch_size=32,ctx_len=16.csv"
    )

    info = parse_filename_stats(filename_stats)
    assert info is not None

    df_stats = pd.read_csv(filename_stats)

    texes = []

    for index, row in df_stats.iterrows():
        sae_index_in = int(row["i"])
        sae_index_out = int(row["j"])
        count = int(row["count"])
        mean = float(row["mean"])
        std = float(row["std"])

        features_filename_in = features_filename.replace("#", f"in-{sae_index_in}")
        features_filename_out = features_filename.replace("#", f"out-{sae_index_out}")

        try:
            tex_in = generate_latent_examples_tex(features_filename_in, True, max_rows)
            tex_out = generate_latent_examples_tex(
                features_filename_out, False, max_rows
            )
        except FileNotFoundError:
            continue
        if tex_in is None or tex_out is None:
            continue

        caption = f"""The top 12 examples that produce the maximum latent activations for the input and output SAE latents with indices {sae_index_in} and {sae_index_out}, respectively.
The Jacobian SAE pair was trained on layer {info.layer} of {info.model_name} with an expansion factor of $R=64$ and sparsity $k=32$.
The examples were collected over the first 10K records of the English subset of the C4 text dataset with a context length of 16 tokens.
The corresponding Jacobian element is non-zero for {count} tokens, and has a mean of \\num{{{mean:.3e}}} (rank {index}) and a standard deviation of \\num{{{std:.3e}}}."""

        tex = "\n".join(
            [
                "\\begin{figure}",
                "\\centering",
                tex_in,
                tex_out,
                f"\\caption{{{caption}}}",
                "\\end{figure}",
            ]
        )

        combined_filename = f"combined-{info.model_name.lower()}-layer-{info.layer}-mean-{index:02d}-in-{sae_index_in}-out-{sae_index_out}.tex"
        with open(f"combined_tex/{combined_filename}", "w", encoding="utf-8") as f:
            f.write(tex)
        texes.append(tex + "\n\\clearpage\n")

    all_filename = (
        f"combined_tex/combined-{info.model_name.lower()}-layer-{info.layer}.tex"
    )
    with open(all_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(texes))


if __name__ == "__main__":
    # generate_tables()
    # generate_main()

    os.makedirs("combined_tex", exist_ok=True)
    generate_combined("stats0_Layer3-32768-J1-LR5.0e-04-k32-T3.0e+08_mean.csv")
    generate_combined("stats0_Layer7-49152-J1-LR5.0e-04-Tokens3.0e+08_mean.csv")
    generate_combined("stats0_Layer15-65536-J1-LR5.0e-04-Tokens3.0e+08_mean.csv")
