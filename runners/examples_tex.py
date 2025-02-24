import ast
import os
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


def generate_examples_tex(filepath: str, max_rows: int | None = None) -> str:
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


def find_examples(top: str = "feature_pairs") -> list[str]:
    dirpaths: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(top):
        for filename in filenames:
            if filename.endswith(".csv"):
                dirpaths.append(os.path.join(dirpath, filename))
    return dirpaths


def prettify(filename: str) -> str:
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


if __name__ == "__main__":
    max_rows = 12
    for filename in find_examples():
        latex_table = generate_examples_tex(filename, max_rows)
        print(filename)
        if "ctx_len=16" in filename:
            for file in [filename.replace("csv", "tex"), prettify(filename)]:
                with open(file, "w", encoding="utf-8") as f:
                    f.write(latex_table)
