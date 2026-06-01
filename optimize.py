"""
Search-based formulation optimizer for nanoparticle diameter.

Supports:
1) Minimize size: find formulations with smallest predicted diameter.
2) Target size: find formulations closest to an arbitrary target nm.

Example:
  python3 optimize.py --mode minimize --top-n 10
  python3 optimize.py --mode target --target-nm 220 --top-n 15
  python3 optimize.py --mode minimize --particle-type-code 1
"""

import argparse
from itertools import product
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

DATA_PATH = Path(__file__).parent / "data" / "ek_c_kitosan_kitosan_tio2_partikulleri_veri_seti.csv"
TARGET = "Particle_Diameter_nm"
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in {DATA_PATH}.")
    return df


def build_models() -> dict[str, Pipeline]:
    return {
        "MLP model": Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                max_iter=10000,
                early_stopping=True,
                random_state=RANDOM_STATE,
            )),
        ]),
        "RBF model": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=100, gamma="scale", epsilon=0.1)),
        ]),
    }


def train_model(df: pd.DataFrame, features: list[str], model_name: str):
    models = build_models()
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(models)}")
    model = models[model_name]
    model.fit(df[features], df[TARGET])
    return model


def predict_diameter(model, features: list[str], params: tuple) -> float:
    row = pd.DataFrame([dict(zip(features, params))])
    return float(model.predict(row)[0])


def search_formulations(
    model,
    df: pd.DataFrame,
    features: list[str],
    mode: str,
    target_nm: Optional[float],
    top_n: int,
) -> pd.DataFrame:
    allowed = {f: sorted(df[f].unique()) for f in features}

    rows = []
    for combo in product(*(allowed[f] for f in features)):
        pred = predict_diameter(model, features, combo)
        if mode == "minimize":
            score = pred
            target_error = None
        else:
            if target_nm is None:
                raise ValueError("target_nm is required in target mode.")
            target_error = abs(pred - target_nm)
            # tie-break toward smaller particles when equally close
            score = target_error + 1e-3 * pred

        row = {f: v for f, v in zip(features, combo)}
        row.update({
            "predicted_diameter_nm": pred,
            "target_error_nm": target_error,
            "score": score,
        })
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("score").head(top_n).reset_index(drop=True)
    result.index = result.index + 1
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search formulation settings for minimum or target nanoparticle size.",
    )
    parser.add_argument(
        "--mode",
        choices=["minimize", "target"],
        default="minimize",
        help="Optimization mode.",
    )
    parser.add_argument(
        "--target-nm",
        type=float,
        default=None,
        help="Required when --mode target (desired particle diameter in nm).",
    )
    parser.add_argument("--top-n", type=int, default=10, help="How many best rows to return.")
    parser.add_argument(
        "--particle-type-code",
        type=int,
        choices=[0, 1],
        default=None,
        help="Optional filter for ek_c dataset: 0=Chitosan, 1=Chitosan/TiO2.",
    )
    parser.add_argument(
        "--model",
        choices=["RBF model", "MLP model"],
        default="RBF model",
        help="Model used to score formulations.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_data()

    if args.particle_type_code is not None:
        if "Particle_Type_Code" not in df.columns:
            raise ValueError("Particle_Type_Code not present in selected dataset.")
        df = df[df["Particle_Type_Code"] == args.particle_type_code].copy()
        if df.empty:
            raise ValueError("No rows left after Particle_Type_Code filtering.")

    features = [c for c in df.columns if c != TARGET]
    model = train_model(df, features, args.model)

    if args.mode == "target" and args.target_nm is None:
        raise SystemExit("For target mode, provide --target-nm (e.g., --target-nm 220).")

    result = search_formulations(
        model=model,
        df=df,
        features=features,
        mode=args.mode,
        target_nm=args.target_nm,
        top_n=args.top_n,
    )

    print(f"\nDataset: {DATA_PATH.name}")
    if args.particle_type_code is not None:
        print(f"Particle_Type_Code filter: {args.particle_type_code}")
    print(f"Model: {args.model}")
    print(f"Mode: {args.mode}")
    if args.mode == "target":
        print(f"Target: {args.target_nm:.2f} nm")

    print("\nTop recommended formulations:")
    print(result.to_string(index=True, formatters={
        "predicted_diameter_nm": "{:.2f}".format,
        "target_error_nm": (lambda x: "" if pd.isna(x) else f"{x:.2f}"),
        "score": "{:.4f}".format,
    }))

    print("\nBest recommendation:")
    best = result.iloc[0]
    for f in features:
        print(f"  {f}: {best[f]}")
    print(f"  Predicted diameter: {best['predicted_diameter_nm']:.2f} nm")
    if args.mode == "target":
        print(f"  Target error: {best['target_error_nm']:.2f} nm")


if __name__ == "__main__":
    main()
