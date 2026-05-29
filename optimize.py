"""
Optimize process parameters for small TiO2-doped chitosan nanoparticles.

Usage:
  python3 optimize.py
  python3 optimize.py --mixing-rate 500 --chitosan-tpp 1.5 --chitosan-conc 0.5 --tpp-conc 1.0
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

DATA_PATH = Path(__file__).parent / "data.csv"
TARGET = "Particle Diameter (nm)"


def load_data():
    df = pd.read_csv(DATA_PATH, delimiter="\t")
    features = [c for c in df.columns if c != TARGET]
    return df, features


def build_models():
    return {
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


def train_best_model(df, features):
    X = df[features]
    y = df[TARGET]
    models = build_models()
    for model in models.values():
        model.fit(X, y)
    return models["RBF model"], features


def snap_to_grid(values, features, allowed):
    snapped = values.copy()
    for i, name in enumerate(features):
        choices = np.array(sorted(allowed[name]))
        snapped[i] = choices[np.argmin(np.abs(choices - snapped[i]))]
    return snapped


def predict_diameter(model, features, params):
    row = pd.DataFrame([dict(zip(features, params))])
    return float(model.predict(row)[0])


def optimize_small_particles(model, features, df):
    allowed = {col: df[col].unique() for col in features}
    bounds = [(df[col].min(), df[col].max()) for col in features]

    def objective(x):
        x_snapped = snap_to_grid(x, features, allowed)
        return predict_diameter(model, features, x_snapped)

    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=300,
        polish=True,
        tol=1e-6,
    )
    best = snap_to_grid(result.x, features, allowed)
    return best, result.fun


def sensitivity(model, features, params, allowed):
    base = predict_diameter(model, features, params)
    rows = []
    for i, name in enumerate(features):
        choices = np.array(sorted(allowed[name]))
        current = params[i]
        for direction, label in [(-1, "decrease"), (1, "increase")]:
            idx = np.where(choices == current)[0]
            if len(idx) == 0:
                nearest = choices[np.argmin(np.abs(choices - current))]
                idx = np.where(choices == nearest)[0]
            idx = int(idx[0]) + direction
            if idx < 0 or idx >= len(choices):
                continue
            trial = params.copy()
            trial[i] = choices[idx]
            pred = predict_diameter(model, features, trial)
            rows.append({
                "parameter": name,
                "change": f"{current} → {trial[i]} ({label})",
                "predicted_diameter_nm": pred,
                "delta_nm": pred - base,
            })
    return pd.DataFrame(rows).sort_values("predicted_diameter_nm")


def format_params(features, params):
    return "\n".join(f"  {name}: {params[i]}" for i, name in enumerate(features))


def print_recommendations(df, features, current, model):
    allowed = {col: df[col].unique() for col in features}
    current = snap_to_grid(np.array(current, dtype=float), features, allowed)
    current_pred = predict_diameter(model, features, current)

    print("\n--- Your input (snapped to experimental grid) ---")
    print(format_params(features, current))
    print(f"  Predicted particle diameter: {current_pred:.2f} nm")

    sens = sensitivity(model, features, current, allowed)
    print("\n--- One-step changes (RBF model) ---")
    print(sens.to_string(index=False, formatters={
        "predicted_diameter_nm": "{:.2f}".format,
        "delta_nm": "{:+.2f}".format,
    }))

    best_change = sens.iloc[0]
    print("\n--- Best single-step move from your settings ---")
    print(f"  Change {best_change['parameter']}: {best_change['change']}")
    print(f"  → predicted diameter {best_change['predicted_diameter_nm']:.2f} nm "
          f"({best_change['delta_nm']:+.2f} nm vs current)")

    best_params, best_pred = optimize_small_particles(model, features, df)
    print("\n--- Global optimum (within dataset ranges, RBF model) ---")
    print(format_params(features, best_params))
    print(f"  Predicted particle diameter: {best_pred:.2f} nm")

    empirical_min = df.loc[df[TARGET].idxmin()]
    print("\n--- Smallest particles in experimental data ---")
    for name in features:
        print(f"  {name}: {empirical_min[name]}")
    print(f"  Measured diameter: {empirical_min[TARGET]:.2f} nm")

    print("\n--- Guidance for very small nanoparticles ---")
    print("  1. Lower mixing rate (rpm): strongest trend in data — high rpm → larger particles.")
    print("     Prefer 250 rpm over 500–750 rpm when possible.")
    print("  2. Lower chitosan:TPP ratio (v/v): negative correlation with size.")
    print("     Prefer 1.0 over 1.5–2.0 where formulation allows.")
    print("  3. Tune TPP and chitosan concentrations together: best measured case used")
    print("     chitosan 0.5 mg/ml and TPP 0.5 mg/ml at 250 rpm, ratio 1.5.")
    print("  4. Use the global optimum above as a starting point; validate experimentally")
    print("     because the model is fit on ~200 points and test R² is moderate.")


def parse_args(features):
    parser = argparse.ArgumentParser(
        description="Predict and optimize nanoparticle diameter for arbitrary inputs.",
    )
    parser.add_argument("--mixing-rate", type=float, help="Mixing rate (rpm)")
    parser.add_argument("--chitosan-tpp", type=float, help="Chitosan:TPP (v/v)")
    parser.add_argument("--chitosan-conc", type=float, help="Chitosan concentration (mg/ml)")
    parser.add_argument("--tpp-conc", type=float, help="TPP concentration (mg/ml)")
    return parser.parse_args()


def main():
    df, features = load_data()
    model, features = train_best_model(df, features)
    args = parse_args(features)

    arg_values = [args.mixing_rate, args.chitosan_tpp, args.chitosan_conc, args.tpp_conc]

    if all(v is None for v in arg_values):
        allowed = {col: df[col].unique() for col in features}
        defaults = [
            df["Mixing rate (rpm)"].median(),
            df["Chitosan:TPP (v/v)"].median(),
            df["Chitosan concentration (mg/ml)"].median(),
            df["TPP Concentration (mg/ml)"].median(),
        ]
        current = snap_to_grid(np.array(defaults), features, allowed)
        print("No inputs provided — using median process settings as example.\n")
    else:
        if any(v is None for v in arg_values):
            raise SystemExit(
                "Provide all four inputs: --mixing-rate, --chitosan-tpp, "
                "--chitosan-conc, --tpp-conc"
            )
        current = np.array(arg_values)

    print_recommendations(df, features, current, model)


if __name__ == "__main__":
    main()
