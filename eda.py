from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = Path("data/ek_c_kitosan_kitosan_tio2_partikulleri_veri_seti.csv")
TARGET = "Particle_Diameter_nm"
OUTPUT_DIR = Path("eda_plots/ek_c_feature_analysis")
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df


def save_correlation_matrix(df: pd.DataFrame) -> None:
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
    )
    plt.title("Correlation Matrix (ek_c dataset)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_matrix_heatmap.png", dpi=200)
    plt.close()


def save_target_correlation_bar(df: pd.DataFrame) -> None:
    corr_with_target = (
        df.corr(numeric_only=True)[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    plt.figure(figsize=(8, 5))
    corr_with_target.plot(kind="bar", color="tab:blue")
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel(f"Correlation with {TARGET}")
    plt.title("Feature Correlation with Target (ek_c dataset)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_target_correlation_bar.png", dpi=200)
    plt.close()


def save_shap_plots(df: pd.DataFrame) -> None:
    features = [c for c in df.columns if c != TARGET]
    X = df[features]
    y = df[TARGET]

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary Plot (ek_c dataset)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Bar) (ek_c dataset)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_summary_bar.png", dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    save_correlation_matrix(df)
    save_target_correlation_bar(df)
    save_shap_plots(df)
    print(f"Saved EDA plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
