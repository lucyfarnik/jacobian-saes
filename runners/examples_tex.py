import ast
import os
import unicodedata

import pandas as pd

COLORS = {
    "jacobian": "Red",
    "sae_acts_in": "Cyan",
    "sae_acts_out": "Green",
}

CATEGORIES = {
    "jacobian": "$\\mat{J}$",
    "sae_acts_in": "In",
    "sae_acts_out": "Out",
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

    if max_rows is not None:
        df = df.head(max_rows)

    for col in [
        "tokens",
        "jacobian",
        "sae_acts_in",
        "sae_acts_out",
    ]:
        df[col] = df[col].apply(ast.literal_eval)

    def get_color(value: float, max_val: float, type_name: str) -> str:
        normalized = float(value) / max_val if max_val != 0 else 0
        return f"{COLORS[type_name]}!{normalized * 100:.3f}"

    def clean_token_for_latex(token: str) -> str:
        token = token.replace("\\", "\\textbackslash{}")
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
        # "\\documentclass{article}",
        # "\\usepackage[utf8]{inputenc}",
        # "\\usepackage[T1]{fontenc}",
        # "\\usepackage[dvipsnames]{xcolor}",
        # "\\usepackage{colortbl}",
        # "\\usepackage{booktabs}",
        # "\\usepackage{longtable}",
        # "\\usepackage{times}",
        # "\\usepackage{icml2025}",
        # "\\begin{document}",
        # "\\renewcommand{\\arraystretch}{0.5}",
        # "\\renewcommand{\\fboxsep}{0pt}",
        # "\\onecolumn",
        "\\begin{table}",
        "\\centering",
        "\\begin{longtable}{lrl}",
        "\\toprule",
        "Category & Max. value & Example tokens \\\\",
        "\\midrule",
    ]

    for _, row in df.iterrows():
        tokens = row["tokens"]

        for value_col in ["jacobian", "sae_acts_in", "sae_acts_out"]:
            values = row[value_col]
            max_value = max(float(val) for val in values)

            colored_tokens = []
            for token, val in zip(tokens, values):
                if token == "<|endoftext|>":
                    continue

                clean_token = clean_token_for_latex(token)
                if clean_token.strip():
                    color = get_color(float(val), max_value, value_col)
                    colored_tokens.append(
                        f"\\colorbox{{{color}}}{{\\strut {clean_token}}}"
                    )

            tokens_str = " ".join(colored_tokens)

            latex_output.append(
                f"{CATEGORIES[value_col]} & {max_value:.3f} & {tokens_str} \\\\"
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


if __name__ == "__main__":
    max_rows = 8
    for filename in find_examples():
        latex_table = generate_examples_tex(filename, max_rows)
        print(filename)
        for file in [
            filename.replace("csv", "tex"),
            filename.replace("/", "_").replace("csv", "tex"),
        ]:
            with open(file, "w", encoding="utf-8") as f:
                f.write(latex_table)
