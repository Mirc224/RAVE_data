import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from evaluation.experiment_evaluation import *
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle


def plot_experiment_metric_for_each_model(experiment_results_df: pd.DataFrame, experiment_code_name: str, metric_to_use: str, metric_label:str, plot_title: str, decimal_places:int=3):
    ordered_experiment_df = experiment_results_df.loc[experiment_results_df[ExperimentResults.EXPERIMENT_CODE_NAME] == experiment_code_name].sort_values(by=metric_to_use, ascending=False)
    plt.figure(figsize=(16, 8))
    plt.barh(ordered_experiment_df[ExperimentResults.MODEL_NAME_W_SIZE], ordered_experiment_df[metric_to_use], color=ordered_experiment_df[ExperimentResults.MODEL_FAMILY_COLOR], edgecolor="black")

    # Add labels and title
    plt.xlabel(metric_label, fontsize=12, fontweight='bold')
    plt.ylabel('Modely', fontsize=12, fontweight='bold')
    plt.title(plot_title, fontweight='bold')

    # Add the F1 score values next to the bars
    for i, score in enumerate(ordered_experiment_df[metric_to_use]):
        plt.text(score, i, f'{score:.{decimal_places}f}', va='center')
    plt.show()


def plot_multiple_experiment_metrics_for_each_model(
        experiment_results_df: pd.DataFrame, 
        experiment_code_name: str):
    experiment_rows =  experiment_results_df.loc[experiment_results_df[ExperimentResults.EXPERIMENT_CODE_NAME] == experiment_code_name]
    ordered_metrics = [
        (f"micro_{ExperimentEvaluator.F1_SCORE}", 2), 
        (ExperimentEvaluator.AVERAGE_RESPONSE_TIME, 2), 
        (ExperimentEvaluator.AVERAGE_RETRIES, 3), 
        (ExperimentEvaluator.INVALID_OUTPUTS_COUNT, 0)
    ]
    metrics_labels = {
        f"micro_{ExperimentEvaluator.F1_SCORE}" : "F1 score", 
        ExperimentEvaluator.AVERAGE_RESPONSE_TIME : "Time (s)", 
        ExperimentEvaluator.AVERAGE_RETRIES : "Number of retries", 
        ExperimentEvaluator.INVALID_OUTPUTS_COUNT : "Invalid outputs count"
    }

    metrics_titles = {
        f"micro_{ExperimentEvaluator.F1_SCORE}" : f"Resulting F1 score - [{experiment_code_name}]", 
        ExperimentEvaluator.AVERAGE_RESPONSE_TIME : f"Resulting average response time - [{experiment_code_name}]", 
        ExperimentEvaluator.AVERAGE_RETRIES : f"Resulting average number of retires - [{experiment_code_name}]", 
        ExperimentEvaluator.INVALID_OUTPUTS_COUNT : f"Resulting invalind outputs count - [{experiment_code_name}]"
    }

    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 13.7))

    for i, metric_tuple in enumerate(ordered_metrics):
        metric, decimal_places = metric_tuple
        x_i, y_i = (i, 0)
        metric_label = metrics_labels[metric]
        metric_title = metrics_titles[metric]
        ordered_rows_by_metric = experiment_rows.sort_values(by=metric, ascending=False)
        axs[x_i].barh(
            ordered_rows_by_metric[ExperimentResults.MODEL_NAME_W_SIZE], 
            ordered_rows_by_metric[metric], 
            color=ordered_rows_by_metric[ExperimentResults.MODEL_FAMILY_COLOR], 
            edgecolor="black")
        axs[x_i].set_xlabel(metric_label, fontsize=8)
        axs[x_i].set_title(metric_title, fontsize=10)
        max_metric = ordered_rows_by_metric[metric].max()
        axs[x_i].set_xlim(0, max_metric + max_metric * 0.08)
        
        additional = max_metric / 100
        for i, score in enumerate(ordered_rows_by_metric[metric]):
            axs[x_i].text(score + additional, i, f'{score:.{decimal_places}f}', va='center')

    plt.tight_layout(pad=1.1, w_pad=0, h_pad=0.3)
    plt.show()

def plot_metric_for_each_model_grouped_by_size(experiment_results_df: pd.DataFrame, metric_to_use: str, y_label: str, plot_title: str, take_max:bool = True, precision: int=3):
    grouped_rows = experiment_results_df.groupby(ExperimentResults.MODEL_NAME_W_SIZE)[metric_to_use]
    rows_to_select = grouped_rows.idxmax() if take_max else grouped_rows.idxmin()
    rows_per_model_and_experiment = experiment_results_df.loc[rows_to_select].sort_values(by=metric_to_use, ascending=take_max)
    model_results_by_size = rows_per_model_and_experiment.groupby(by=ExperimentResults.MODEL_SIZE)

    groups = []
    model_sizes = list(sorted(model_results_by_size.groups.keys()))  # Model sizes
    for model_size in model_sizes:
        ordered_group = model_results_by_size.get_group(model_size).sort_values(by=ExperimentResults.MODEL_WITH_EXPERIMENT)
        group_name = ordered_group[ExperimentResults.MODEL_NORMALIZED_SIZE].unique()[0]
        model_experiment_names = ordered_group[ExperimentResults.MODEL_WITH_EXPERIMENT].to_list()
        model_experiment_metric_value = ordered_group[metric_to_use].to_list()
        model_experiment_color = ordered_group[ExperimentResults.MODEL_FAMILY_COLOR].to_list()
        group_dict = {
            "groupLabel": group_name,
            "values": model_experiment_metric_value,
            "labels": model_experiment_names,
            "colors": model_experiment_color
        }
        groups.append(group_dict)


    bar_width = 1  # Width of each bar
    group_gap = 2  # Larger gap between groups
    bar_gap = 0.15 # Gap between bars within a group
    box_margin = 0.2

    # Calculate bar positions
    x_positions = []
    x_ticks = []
    group_labels = []
    group_positions = []
    group_boundaries = []
    current_x = 0

    for group in groups:
        n_bars = len(group["values"])
        group_width = (n_bars - 1) * (bar_width + bar_gap)  # Width of the group
        group_start = current_x + group_width / 2 # Center the group
        x_positions.extend(np.arange(current_x, current_x + group_width + bar_width, bar_width + bar_gap)[:n_bars])
        x_ticks.append(group_start)
        group_positions.append(group_start)
        group_labels.append(group["groupLabel"])
        group_boundaries.append((current_x - box_margin, current_x + group_width + box_margin))
        current_x += group_width + group_gap  # Add gap after the group

    # Flatten data for plotting
    all_values = [value for group in groups for value in group["values"]]
    colors = [value for group in groups for value in group["colors"]]
    bar_labels = [value for group in groups for value in group["labels"]]

    plt.figure(figsize=(16, 8))
    # Plot
    plt.bar(x_positions, all_values, width=bar_width, color=colors, edgecolor="black")

    for x, y in zip(x_positions, all_values):
        plt.text(x, y , f"{y:.{precision}f}", ha='center', va='bottom', fontsize=10)

    plt.xticks(x_positions, bar_labels, rotation=45, ha='right', fontsize=10)

    additional_percentage = 0.1
    max_value = max(all_values)
    group_additional = max_value * 0.05
    # Add group labels above the bars
    for pos, label in zip(group_positions, group_labels):
        plt.text(pos, max_value + group_additional, label, ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add boxes around groups
    ax = plt.gca()
    for boundary in group_boundaries:
        start, end = boundary
        rect = Rectangle((start - bar_width / 2, 0),  # Starting position (adjust for bar width)
                        end - start + bar_width,     # Width of the rectangle
                        max_value + group_additional,      # Height of the rectangle
                        fill=False,                 # No fill
                        edgecolor='grey',          # Box color
                        linewidth=1)              # Box thickness
        ax.add_patch(rect)

    plt.xlabel("Model + Settings", fontsize=14, fontweight='bold')
    plt.ylabel(y_label, fontsize=14, fontweight='bold')
    # plt.title(plot_title,  fontweight='bold')

    family_color_df = experiment_results_df[[ExperimentResults.MODEL_NAME, ExperimentResults.MODEL_FAMILY_COLOR]].drop_duplicates(subset=[ExperimentResults.MODEL_NAME, ExperimentResults.MODEL_FAMILY_COLOR])
    legend_colors = family_color_df[ExperimentResults.MODEL_FAMILY_COLOR].unique()
    legend_labels = family_color_df[ExperimentResults.MODEL_NAME].unique()
    handles = [mpatches.Patch(facecolor=color, edgecolor='black', label=label) for label, color in zip(legend_labels, legend_colors)]

    # Add the legend
    plt.legend(handles=handles, title="Model family", bbox_to_anchor=(1, 1))

    # Show the plot
    plt.ylim(0, max_value + max_value * additional_percentage)
    plt.tight_layout()
    plt.show()