from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

TARGET = "Particle_Diameter_nm"
TEST_SIZE = 0.2
RANDOM_STATE = 42
PLOTS_DIR = Path("plots")
RESULTS_DIR = Path("results_table")

def build_models() -> dict[str, Pipeline]:
    return {
        "MLP model": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", max_iter=10000, early_stopping=True, random_state=RANDOM_STATE,)),
        ]),
        "RBF model": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)),
        ]),
        "Random Forest model": Pipeline([
            ("model", RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1,)),
        ]),
    }

def unnormalized_metrics(y_true: pd.Series, y_pred: np.ndarray):
    y_true_np = y_true.values
    mse = mean_squared_error(y_true_np, y_pred)
    mae = mean_absolute_error(y_true_np, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_true_np, y_pred)
    return mse, mae, rmse, r2

def scale_target(y, y_scaler: MinMaxScaler):
    return y_scaler.transform(np.asarray(y).reshape(-1, 1)).ravel()

def plot_regression_output_vs_target(ax, y_true, y_pred, title: str, fit_color: str, y_scaler: MinMaxScaler):
    """Fig. 5 style: predicted (output) vs actual (target), normalized."""
    target = scale_target(y_true, y_scaler)
    output = scale_target(y_pred, y_scaler)
    target_std = float(np.std(target))
    output_std = float(np.std(output))
    if target_std > 0 and output_std > 0:
        r = float(np.corrcoef(target, output)[0, 1])
        slope, intercept = np.polyfit(target, output, 1)
        fit_note = f"R={r:.5f}"
    else:
        # Guard against degenerate folds where one side is constant.
        r = np.nan
        slope, intercept = 1.0, 0.0
        fit_note = "R=n/a (constant values)"

    ax.scatter(target, output, facecolors="none", edgecolors=fit_color, s=40)
    lo = min(target.min(), output.min())
    hi = max(target.max(), output.max())
    margin = 0.05 * (hi - lo) if hi > lo else 0.05
    lo, hi = lo - margin, hi + margin
    ax.plot([lo, hi], [lo, hi], "k:", linewidth=1.2, label="Y = T")
    xs = np.linspace(lo, hi, 100)
    ax.plot(xs, slope * xs + intercept, color=fit_color, linewidth=1.5, label="Fit")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Target")
    ax.set_ylabel("Output")
    ax.set_title(
        f"{title}: {fit_note}\nOutput ~= {slope:.2f} * Target + {intercept:.3f}"
    )
    ax.legend(loc="lower right", fontsize=8)

def save_scatter_plot(df: pd.DataFrame, features: list[str], dataset_name: str):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    type_labels = {0: "Chitosan", 1: "Chitosan/TiO2"}
    type_colors = {0: "tab:blue", 1: "tab:orange"}

    for ax, col in zip(axes, features):
        type_series = pd.to_numeric(df["Particle_Type_Code"], errors="coerce")
        for particle_type in sorted(type_series.dropna().unique()):
            mask = type_series == particle_type
            color = type_colors.get(int(particle_type), "tab:gray")
            label = type_labels.get(int(particle_type), f"Type {int(particle_type)}")
            ax.scatter(
                df.loc[mask, col],
                df.loc[mask, TARGET],
                alpha=0.6,
                edgecolors="none",
                color=color,
                label=label,
            )
        ax.legend(loc="best", fontsize=8)
        ax.set_xlabel(col)
        ax.set_ylabel(TARGET)
        ax.set_title(f"{TARGET} vs {col}")
    fig.tight_layout()
    fig.savefig(f"plots/scatter_plot.png", dpi=150)
    plt.close(fig)

def plot_target_vs_all_models(y_true: pd.Series, predictions: dict[str, np.ndarray], y_scaler: MinMaxScaler, out_file: Path):
    y_true_scaled = y_scaler.transform(y_true.values.reshape(-1, 1)).ravel()
    x = range(len(y_true_scaled))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y_true_scaled, "k-*", label="Target", markersize=6)
    style_cycle = ["r-o", "b-s", "g-^", "m-d", "c-v"]
    for idx, (model_name, y_pred) in enumerate(predictions.items()):
        y_pred_scaled = y_scaler.transform(y_pred.reshape(-1, 1)).ravel()
        ax.plot(
            x,
            y_pred_scaled,
            style_cycle[idx % len(style_cycle)],
            label=model_name,
            markersize=5,
            markerfacecolor="none",
            markeredgewidth=1.2,
        )
    ax.set_xlabel("Test set Output")
    ax.set_ylabel("Normalized Output")
    ax.legend(loc="upper right")
    ax.set_xlim(0, max(len(y_true_scaled) - 1, 1))
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)

def save_model_regression_plots(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_scaler: MinMaxScaler,
    dataset_name: str,
):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    plot_regression_output_vs_target(
        axes[0], y_train, y_pred_train, "Training", "C0", y_scaler
    )
    plot_regression_output_vs_target(
        axes[1], y_test, y_pred_test, "Validation", "C2", y_scaler
    )
    fig.suptitle(f"{model_name} — output vs target ({dataset_name})", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"plots/{dataset_name}/regression_{model_name}.png", dpi=150)
    plt.close(fig)


def save_univariate_regression_plots(df: pd.DataFrame, features: list[str], y: pd.Series, y_scaler: MinMaxScaler, dataset_name: str):
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4))
    if len(features) == 1:
        axes = [axes]
    for ax, feature in zip(axes, features):
        X_one = df[[feature]]
        X_tr, X_te, y_tr, y_te = train_test_split(X_one, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        uni_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                max_iter=5000,
                early_stopping=True,
                random_state=RANDOM_STATE,
            )),
        ])
        uni_model.fit(X_tr, y_tr)
        plot_regression_output_vs_target(ax, y_te, uni_model.predict(X_te), f"{feature}", "C2", y_scaler)
    fig.suptitle(f"Validation regression — univariate (each axis: one feature)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"plots/{dataset_name}/regression_univariate_validation_all_features.png", dpi=150)
    plt.close(fig)

def train_one_dataset(dataset_path: Path, output_rows: list[dict]):
    Path(f"plots/{dataset_path.stem}").mkdir(parents=True, exist_ok=True)
    dataset_name = dataset_path.stem
    df = pd.read_csv(dataset_path)
    df.columns = [c.strip() for c in df.columns]

    features = [c for c in df.columns if c != TARGET]
    X = df[features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    y_scaler = MinMaxScaler()
    y_scaler.fit(y_train.values.reshape(-1, 1))
    
    if 'ek_c' in dataset_name:
        save_scatter_plot(df, features[1:], dataset_path.stem)
        save_univariate_regression_plots(df, features[1:], y, y_scaler, dataset_name)
    
    models = build_models()
    fitted_models = {}
    model_predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        y_pred = model.predict(X_test)
        model_predictions[name] = y_pred
        mse, mae, rmse, r2 = unnormalized_metrics(y_test, y_pred)
        output_rows.append({
            "Dataset": dataset_path.name,
            "Model": name,
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2,
        })

        save_model_regression_plots(
            model,
            name,
            X_train,
            X_test,
            y_train,
            y_test,
            y_scaler,
            dataset_name,
        )

    plot_target_vs_all_models(
        y_test,
        model_predictions,
        y_scaler,
        f"plots/{dataset_name}/test_output_all_models.png",
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (name, model) in zip(axes, fitted_models.items()):
        plot_regression_output_vs_target(
            ax,
            y_test,
            model.predict(X_test),
            f"{name} (validation)",
            "C0",
            y_scaler,
        )
    fig.suptitle(f"Validation regression — all models ({dataset_path.name})", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"plots/{dataset_name}/regression_all_models_validation.png", dpi=150)
    plt.close(fig)

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_rows = []
    dataset_paths = sorted(Path("data").glob("ek_*.csv"))

    for dataset_path in dataset_paths:
        train_one_dataset(dataset_path, output_rows)

    results = pd.DataFrame(output_rows)
    results.to_csv(RESULTS_DIR / "performance_table_all_datasets.csv", index=False)

if __name__ == "__main__":
    main()