import ast
import os
import re
import unicodedata

import pandas as pd

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


def generate_pair_examples_tex(filepath: str, max_rows: int | None = None) -> str:
    df = pd.read_csv(filepath)
    df = df.sort_values("jacobian_max", ascending=False)

    MAX_VALS = {
        "jacobian": float(df["jacobian_max"].max()),
        "sae_acts_in": float(df["sae_acts_in_max"].max()),
        "sae_acts_out": float(df["sae_acts_out_max"].max()),
    }

    if max_rows is not None:
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
        "\\begin{table}",
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
            f"\\caption{{{filepath.replace('_', ' ')}}}",
            "\\end{table}",
        ]
    )

    return "\n".join(latex_output)


def generate_latent_examples_tex(
    filepath: str, sae_in: bool, max_rows: int | None = None
) -> str:
    parsed = parse_latent(prettify_latent(filepath))
    if parsed is None:
        caption = filepath.replace("_", " ")
    else:
        model = parsed["model"]
        layer = parsed["layer"]
        stat = parsed["stat"]
        sae = parsed["sae"]
        index = parsed["index"]
        caption = f"{model} layer {layer} {stat} {sae} {index}"

    df = pd.read_csv(filepath)
    df = df.sort_values("sae_acts_max", ascending=False)

    if max_rows is not None:
        df = df.head(max_rows)

    for col in [
        "tokens",
        "sae_acts",
    ]:
        df[col] = df[col].apply(ast.literal_eval)

    type_name = "sae_acts_in" if sae_in else "sae_acts_out"

    def get_color(value: float) -> str:
        max_val = float(df["sae_acts_max"].max())
        normalized = float(value) / max_val if max_val != 0 else 0
        return f"{COLORS[type_name]}!{normalized * 100:.3f}"

    latex_output = [
        "\\begin{table}",
        "\\centering",
        "\\begin{longtable}{rl}",
        "\\toprule",
        "Max. abs. value & Example tokens \\\\",
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

            latex_output.append(f"\\num{{{max_value:.3e}}} & {tokens_str} \\\\")

            latex_output.append("\\midrule")

    latex_output.pop()  # remove last \midrule
    latex_output.extend(
        [
            "\\bottomrule",
            "\\end{longtable}",
            f"\\caption{{{caption}}}",
            "\\end{table}",
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


def prettify_pair(filename: str) -> str:
    return os.path.join(
        "feature_pairs_tex",
        filename.replace("/", "_")
        .replace("csv", "tex")
        .replace("feature_pairs_", "")
        .replace("Layer3-32768-J1-LR5.0e-04-k32-T3.0e+08", "pythia-70m-layer-3")
        .replace("Layer7-49152-J1-LR5.0e-04-k32-T3.0e+08", "pythia-160m-layer-7")
        .replace("Layer15-65536-J1-LR5.0e-04-k32-T3.0e+08", "pythia-410m-layer-15")
        .replace("stas_c4-en-10k,train,", "")
        .replace("examples-", "in-")
        .replace("-v-", "-out-")
        .replace("batch_size=", "b")
        .replace("ctx_len=", "t")
        .replace(",", "-")
        .replace("_", "-"),
    )


def prettify_latent(filename: str) -> str:
    return os.path.join(
        "features_tex",
        filename.replace("/", "_")
        .replace("csv", "tex")
        .replace("features_", "")
        .replace("Layer3-32768-J1-LR5.0e-04-k32-T3.0e+08", "pythia-70m-layer-3")
        .replace("Layer7-49152-J1-LR5.0e-04-Tokens3.0e+08", "pythia-160m-layer-7")
        .replace("Layer15-65536-J1-LR5.0e-04-Tokens3.0e+08", "pythia-410m-layer-15")
        .replace("stas_c4-en-10k,train,", "")
        .replace("examples-", "")
        .replace("batch_size=", "b")
        .replace("ctx_len=", "t")
        .replace(",", "-")
        .replace("_", "-"),
    )


def parse_latent(filename: str) -> dict | None:
    pattern = r"(pythia-\d+m)-layer-(\d+)-([\w-]+)-(in|out)-(\d+)-b\d+-t\d+\.tex"
    match = re.match(pattern, os.path.basename(filename))
    if match:
        return {
            "model": match.group(1),
            "layer": int(match.group(2)),
            "stat": match.group(3),
            "sae": match.group(4),
            "index": int(match.group(5)),
        }
    return None


if __name__ == "__main__":
    max_rows = 12

    os.makedirs("feature_pairs_tex", exist_ok=True)
    # for filename in find_examples("feature_pairs"):
    #     latex_table = generate_pair_examples_tex(filename, max_rows)
    #     print(filename)
    #     if "ctx_len=16" in filename:
    #         for file in [filename.replace("csv", "tex"), prettify_pair(filename)]:
    #             with open(file, "w", encoding="utf-8") as f:
    #                 f.write(latex_table)

    os.makedirs("features_tex", exist_ok=True)
    for filename in find_examples("features"):
        sae_in = "examples-in" in filename
        latex_table = generate_latent_examples_tex(filename, sae_in, max_rows)
        if "ctx_len=16" in filename:
            for file in [filename.replace("csv", "tex"), prettify_latent(filename)]:
                with open(file, "w", encoding="utf-8") as f:
                    f.write(latex_table)
