import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

DATASET_NAME = "TiO2 doped chitosan"

df = pd.read_csv("data.csv", delimiter="\t")

target = "Particle Diameter (nm)"
features = [c for c in df.columns if c != target]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.ravel()

for ax, col in zip(axes, features):
    ax.scatter(df[col], df[target], alpha=0.6, edgecolors="none")
    ax.set_xlabel(col)
    ax.set_ylabel(target)
    ax.set_title(f"{target} vs {col}")

plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.close()

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "MLP model": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=10000,
            early_stopping=True,
            random_state=42,
        )),
    ]),
    "RBF model": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)),
    ]),
}

y_scaler = MinMaxScaler()
y_scaler.fit(y_train.values.reshape(-1, 1))


def normalized_metrics(y_true, y_pred):
    y_true_scaled = y_scaler.transform(y_true.values.reshape(-1, 1)).ravel()
    y_pred_scaled = y_scaler.transform(y_pred.reshape(-1, 1)).ravel()
    mse = mean_squared_error(y_true_scaled, y_pred_scaled)
    mae = mean_absolute_error(y_true_scaled, y_pred_scaled)
    rmse = mse**0.5
    r2 = r2_score(y_true_scaled, y_pred_scaled)
    return mse, mae, rmse, r2


def scale_target(y):
    return y_scaler.transform(np.asarray(y).reshape(-1, 1)).ravel()


def plot_regression_output_vs_target(ax, y_true, y_pred, title, fit_color):
    """Fig. 5 style: predicted (output) vs actual (target), normalized."""
    target = scale_target(y_true)
    output = scale_target(y_pred)
    r = np.corrcoef(target, output)[0, 1]
    slope, intercept = np.polyfit(target, output, 1)

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
        f"{title}: R={r:.5f}\nOutput ~= {slope:.2f} * Target + {intercept:.3f}"
    )
    ax.legend(loc="lower right", fontsize=8)


def save_model_regression_plots(model, model_name):
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    plot_regression_output_vs_target(
        axes[0], y_train, y_pred_train, "Training", "C0"
    )
    plot_regression_output_vs_target(
        axes[1], y_test, y_pred_test, "Validation", "C2"
    )
    fig.suptitle(
        f"{model_name} — output vs target ({DATASET_NAME})",
        fontsize=11,
    )
    fig.tight_layout()
    slug = model_name.lower().replace(" ", "_")
    fig.savefig(f"regression_{slug}.png", dpi=150)
    plt.close(fig)


def save_univariate_regression_plots():
    """Fig. 5 style plots using one input feature at a time."""
    for feature in features:
        X_one = df[[feature]]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_one, y, test_size=0.2, random_state=42
        )
        uni_model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                max_iter=5000,
                early_stopping=True,
                random_state=42,
            )),
        ])
        uni_model.fit(X_tr, y_tr)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        plot_regression_output_vs_target(
            axes[0], y_tr, uni_model.predict(X_tr), "Training", "C0"
        )
        plot_regression_output_vs_target(
            axes[1], y_te, uni_model.predict(X_te), "Validation", "C2"
        )
        fig.suptitle(f"MLP — input: {feature}", fontsize=10)
        fig.tight_layout()
        slug = feature.lower().replace(" ", "_").replace(":", "").replace("/", "")
        fig.savefig(f"regression_feature_{slug}.png", dpi=150)
        plt.close(fig)


def plot_target_vs_model(y_true, y_pred, model_name, filename):
    y_true_scaled = y_scaler.transform(y_true.values.reshape(-1, 1)).ravel()
    y_pred_scaled = y_scaler.transform(y_pred.reshape(-1, 1)).ravel()
    x = range(len(y_true_scaled))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y_true_scaled, "k-*", label="Target", markersize=6)
    ax.plot(
        x,
        y_pred_scaled,
        "r-o",
        label=model_name,
        markersize=5,
        markerfacecolor="none",
        markeredgewidth=1.2,
    )
    ax.set_xlabel("Test set Output")
    ax.set_ylabel(f"{model_name} Output")
    ax.legend(loc="upper right")
    ax.set_xlim(0, max(len(x) - 1, 1))
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)


rows = []
fitted_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    fitted_models[name] = model
    y_pred = model.predict(X_test)
    mse, mae, rmse, r2 = normalized_metrics(y_test, y_pred)
    rows.append({
        "Dataset": DATASET_NAME,
        "Model": name,
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
    })

    slug = name.lower().replace(" ", "_")
    plot_target_vs_model(y_test, y_pred, name, f"test_output_{slug}.png")
    save_model_regression_plots(model, name)

save_univariate_regression_plots()

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, model) in zip(axes, fitted_models.items()):
    plot_regression_output_vs_target(
        ax, y_test, model.predict(X_test), f"{name} (validation)", "C0"
    )
fig.suptitle(f"Validation regression — all models ({DATASET_NAME})", fontsize=11)
fig.tight_layout()
fig.savefig("regression_all_models_validation.png", dpi=150)
plt.close(fig)

results = pd.DataFrame(rows)
print(f"\nTable 3 — Best performance indices ({DATASET_NAME}, test set, normalized output)\n")
print(results.to_string(index=False, formatters={
    "MSE": "{:.4f}".format,
    "MAE": "{:.4f}".format,
    "RMSE": "{:.4f}".format,
    "R²": "{:.4f}".format,
}))
results.to_csv("performance_table_tio2.csv", index=False)
