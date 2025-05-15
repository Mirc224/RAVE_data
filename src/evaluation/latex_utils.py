import pandas as pd
from evaluation.experiment_evaluation import *

class TableHeaderCell:
    def __init__(self, code_name: str, header_name: str|None = None, can_bold: bool = True):
        self._code_name = code_name
        self._header_name = header_name if header_name is not None else code_name
        self._can_bold = can_bold
    
    @property
    def code_name(self) -> str:
        return self._code_name
    
    @property
    def header_name(self) -> str:
        return self._header_name
    
    @property
    def can_bold(self) -> bool:
        return self._can_bold

def add_thead(text: str) -> str:
    return rf"\thead{{{text}}}"

def add_bold(text: str) -> str:
    return rf"\textbf{{{text}}}"

def row_to_latex_table_line(row: pd.Series, ordered_experiment_headers: list[TableHeaderCell], precision: int, bold_max:bool|None=True) -> str:
    result = f"{add_bold(row.name)} & "
    values = [row[header_name.code_name] for header_name in ordered_experiment_headers]
    ids_to_bold = []
    valid_values_to_search = [row[header_name.code_name] for header_name in ordered_experiment_headers if header_name.can_bold]
    if bold_max is not None and valid_values_to_search:
        edge_value = max(valid_values_to_search) if bold_max else min(valid_values_to_search)
        ids_to_bold = [i for i, value in enumerate(values) if value == edge_value and ordered_experiment_headers[i].can_bold]
    result_values = []
    for value in values:
        if int(value) == round(value, precision):
            result_values.append(str(int(value)))
            continue
        result_values.append(f"{value:.{precision}f}")
    # values = [f"{value:.{precision}f}" for value in values]
    for id_to_bold in ids_to_bold:
        result_values[id_to_bold] = add_bold(result_values[id_to_bold])
    result += " & ".join(result_values)
    result += r" \\ \hline"
    return result

def get_latex_table_lines(
        result_df: pd.DataFrame, 
        metric_to_use: str, 
        ordered_model_names: list[str], 
        ordered_experiment_table_heads: list[TableHeaderCell],
        metric_label: str,
        caption: str,
        table_label: str,
        metric_precision: int = 3,
        bold_max: bool|None = None) -> list[str]:
    columns = [ExperimentResults.MODEL_NAME_W_SIZE, ExperimentResults.EXPERIMENT_CODE_NAME, metric_to_use]
    pivot_results = result_df[columns].pivot(index=ExperimentResults.MODEL_NAME_W_SIZE, columns=ExperimentResults.EXPERIMENT_CODE_NAME, values=metric_to_use)
    experiment_code_names_headers = [add_bold(add_thead(header_name.header_name.replace("_", r"\_"))) for header_name in ordered_experiment_table_heads]
    number_of_experiments = len(ordered_experiment_table_heads)

    result_table: list[str] = []
    result_table.append(r"\begin{table}[ht]")
    result_table.append(r"\centering")
    result_table.append(r"\rowcolors{2}{white}{gray!25}")
    result_table.append(rf"\begin{{tabular}}{{{"|" + "c|" * (number_of_experiments + 1)}}}")
    result_table.append(r"\hline")
    result_table.append(r"\hiderowcolors")
    result_table.append(rf"\multirow{{2}}{{*}}{{{add_bold(add_thead("Model"))}}} & \multicolumn{{{number_of_experiments}}}{{c|}}{{{add_bold(add_thead(metric_label))}}} \\ \cline{{2-{number_of_experiments+1}}}")
    result_table.append("& " + " & ".join(experiment_code_names_headers) + r" \\ \hline")
    result_table.append(r"\showrowcolors")
    for model_name in ordered_model_names:
        row = pivot_results.loc[model_name]
        latex_table_row = row_to_latex_table_line(row, ordered_experiment_table_heads, metric_precision, bold_max)
        result_table.append(latex_table_row)
    result_table.append(r"\end{tabular}")
    result_table.append(rf"\caption{{{caption}}}")
    result_table.append(rf"\label{{{table_label}}}")
    result_table.append(r"\end{table}")
    return result_table