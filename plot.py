from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ALL_MODELS_PATH = Path("tables/performance_table_all_datasets.csv")
LLM_PATH = Path("tables/performance_table_llm_ek_c.csv")
OUTPUT_PATH = Path("plots/ek_c_unnormalized_metrics_llm_mlp_rbf.png")
EK_C_NAME = "ek_c_kitosan_kitosan_tio2_partikulleri_veri_seti.csv"


def main() -> None:
    all_df = pd.read_csv(ALL_MODELS_PATH)
    llm_df = pd.read_csv(LLM_PATH)

    # Keep only ek_c rows for classical models.
    ek_c_df = all_df[all_df["Dataset"] == EK_C_NAME].copy()
    mlp_row = ek_c_df[ek_c_df["Model"] == "MLP model"].iloc[0]
    rbf_row = ek_c_df[ek_c_df["Model"] == "RBF model"].iloc[0]

    # Use LLM row that corresponds to LLM predictions on subset.
    llm_row = llm_df[llm_df["Model"] == "LLM + retrieved examples"].iloc[0]

    plot_df = pd.DataFrame([
        {
            "Model": "MLP",
            "MSE": float(mlp_row["MSE"]),
            "MAE": float(mlp_row["MAE"]),
            "RMSE": float(mlp_row["RMSE"]),
            "R2": float(mlp_row["R²"]),
        },
        {
            "Model": "RBF",
            "MSE": float(rbf_row["MSE"]),
            "MAE": float(rbf_row["MAE"]),
            "RMSE": float(rbf_row["RMSE"]),
            "R2": float(rbf_row["R²"]),
        },
        {
            "Model": "LLM",
            "MSE": float(llm_row["MSE"]),
            "MAE": float(llm_row["MAE"]),
            "RMSE": float(llm_row["RMSE"]),
            "R2": float(llm_row["R2"]),
        },
    ])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()
    metrics = ["MSE", "MAE", "RMSE", "R2"]
    model_colors = ["tab:blue", "tab:orange", "tab:green"]

    for ax, metric in zip(axes, metrics):
        ax.bar(plot_df["Model"], plot_df[metric], color=model_colors)
        ax.set_title(f"ek_c: {metric}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Value")
        ax.grid(axis="y", alpha=0.25)
        if metric == "R2":
            ax.set_ylim(0, 1)

    fig.suptitle("LLM vs MLP vs RBF (ek_c dataset)", fontsize=13)
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=180)
    plt.close(fig)

    print(f"Saved plot: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
