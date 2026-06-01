import json
import re
from dotenv import load_dotenv
import os
import requests
from pathlib import Path
load_dotenv()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, StandardScaler

TARGET = "Particle_Diameter_nm"
TEST_SIZE = 0.2
K_NEIGHBORS = 8
MAX_LLM_SAMPLES = 40

def normalized_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2}

def weighted_knn_predict(X_train_scaled, y_train, X_query_scaled, k):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X_train_scaled)
    distances, indices = knn.kneighbors(X_query_scaled, return_distance=True)
    weights = 1.0 / (distances + 1e-9)
    pred = np.sum(weights * y_train[indices], axis=1) / np.sum(weights, axis=1)
    return pred, distances, indices

def call_openai_compatible_chat(system_prompt: str, user_prompt: str):
    endpoint = f"https://api.openai.com/v1/chat/completions"
    payload = {
        "model": "gpt-4o-mini",
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    response = requests.post(
        endpoint,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        },
        json=payload,
    )
    result = response.json()
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"Unexpected API response format: {result}")

def parse_llm_json(text):
    # Handle answers with markdown code fences.
    if "```" in text:
        fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced[0]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as err:
        raise ValueError(f"Could not parse model JSON response: {text}") from err
    if "predicted_size_nm" not in parsed:
        raise ValueError(f"Missing 'predicted_size_nm' in response: {parsed}")
    return parsed


def parse_llm_batch_json(text):
    if "```" in text:
        fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced[0]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as err:
        raise ValueError(f"Could not parse model JSON response: {text}") from err
    if "predictions" not in parsed or not isinstance(parsed["predictions"], list):
        raise ValueError(f"Missing 'predictions' list in response: {parsed}")
    return parsed["predictions"]


def build_llm_prompt_batch(
    feature_names: list[str],
    batch_samples: list[dict],
) -> tuple[str, str]:
    system_prompt = (
        "You are a scientific numeric regression assistant. "
        "Use nearest-example interpolation to estimate particle diameter for each sample. "
        "Return ONLY valid JSON."
    )

    sample_sections = []
    for sample in batch_samples:
        query_lines = "\n".join(
            f"- {name}: {value}" for name, value in zip(feature_names, sample["query_values"])
        )
        sample_sections.append(
            f"""
Sample ID: {sample["sample_id"]}
Query sample:
{query_lines}

Nearest known samples (CSV):
{sample["neighbor_csv"]}
"""
        )

    user_prompt = f"""
Predict particle size in nm from process parameters for ALL samples below.

{''.join(sample_sections)}

Return only JSON with this exact structure:
{{
  "predictions": [
    {{
      "sample_id": <int>,
      "predicted_size_nm": <float>,
      "lower_bound_nm": <float>,
      "upper_bound_nm": <float>,
      "method_note": "<short string>"
    }}
  ]
}}
"""
    return system_prompt, user_prompt


def save_target_prediction_plot(
    y_true: np.ndarray,
    retrieval_pred: np.ndarray,
    llm_pred: np.ndarray,
    y_scaler: MinMaxScaler,
    out_file: str,
) -> None:
    out_file = Path(out_file)
    y_true_scaled = y_scaler.transform(y_true.reshape(-1, 1)).ravel()
    retrieval_scaled = y_scaler.transform(retrieval_pred.reshape(-1, 1)).ravel()
    llm_scaled = y_scaler.transform(llm_pred.reshape(-1, 1)).ravel()
    x = range(len(y_true_scaled))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x, y_true_scaled, "k-*", label="Target", markersize=6)
    ax.plot(
        x,
        retrieval_scaled,
        "b-s",
        label="Retrieval-kNN",
        markersize=5,
        markerfacecolor="none",
        markeredgewidth=1.2,
    )
    ax.plot(
        x,
        llm_scaled,
        "r-o",
        label="LLM prediction",
        markersize=5,
        markerfacecolor="none",
        markeredgewidth=1.2,
    )
    ax.set_xlabel("Test set Output")
    ax.set_ylabel("Normalized Output")
    ax.legend(loc="upper right")
    ax.set_xlim(0, max(len(y_true_scaled) - 1, 1))
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=150)
    plt.close(fig)


def save_target_prediction_plot_unnormalized(
    y_true: np.ndarray,
    retrieval_pred: np.ndarray,
    llm_pred: np.ndarray,
    out_file: str,
) -> None:
    out_file = Path(out_file)
    x = range(len(y_true))

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x, y_true, "k-*", label="Target", markersize=6)
    ax.plot(
        x,
        retrieval_pred,
        "b-s",
        label="Retrieval-kNN",
        markersize=5,
        markerfacecolor="none",
        markeredgewidth=1.2,
    )
    ax.plot(
        x,
        llm_pred,
        "r-o",
        label="LLM prediction",
        markersize=5,
        markerfacecolor="none",
        markeredgewidth=1.2,
    )
    ax.set_xlabel("Test set Output")
    ax.set_ylabel("Particle Diameter (nm)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, max(len(y_true) - 1, 1))
    fig.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)

def run():
    df = pd.read_csv("data/ek_c_kitosan_kitosan_tio2_partikulleri_veri_seti.csv")
    df.columns = [c.strip() for c in df.columns]
    feature_cols = [c for c in df.columns if c != TARGET]
    X = df[feature_cols].astype(float)
    y = df[TARGET].astype(float).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    y_scaler = MinMaxScaler()
    y_scaler.fit(y_train.reshape(-1, 1))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    retrieval_pred_all, distances_all, indices_all = weighted_knn_predict(X_train_scaled, y_train, X_test_scaled, k=K_NEIGHBORS)
    retrieval_all_metrics = normalized_metrics(y_test, retrieval_pred_all)

    eval_count = min(MAX_LLM_SAMPLES, len(X_test))
    X_eval = X_test.iloc[:eval_count].copy()
    y_eval = y_test[:eval_count].copy()
    retrieval_eval_pred = retrieval_pred_all[:eval_count]
    distances_eval = distances_all[:eval_count]
    indices_eval = indices_all[:eval_count]

    batch_samples = []
    for i in range(eval_count):
        query_values = X_eval.iloc[i].tolist()
        neighbor_idx = indices_eval[i]
        neighbor_rows = X_train.iloc[neighbor_idx].copy()
        neighbor_rows[TARGET] = y_train[neighbor_idx]
        neighbor_rows["distance_in_scaled_space"] = distances_eval[i]
        neighbor_rows = neighbor_rows.sort_values("distance_in_scaled_space")
        batch_samples.append({
            "sample_id": i,
            "query_values": query_values,
            "neighbor_csv": neighbor_rows.to_csv(index=False),
        })

    system_prompt, user_prompt = build_llm_prompt_batch(feature_cols, batch_samples)
    print(system_prompt)
    print(user_prompt)
    content = call_openai_compatible_chat(system_prompt=system_prompt, user_prompt=user_prompt)
    parsed_predictions = parse_llm_batch_json(content)

    parsed_by_id = {int(item["sample_id"]): item for item in parsed_predictions if "sample_id" in item}
    llm_preds = []
    llm_lows = []
    llm_highs = []
    raw_responses = []
    for i in range(eval_count):
        item = parsed_by_id.get(i, {})
        fallback = float(retrieval_eval_pred[i])
        llm_preds.append(float(item.get("predicted_size_nm", fallback)))
        llm_lows.append(float(item.get("lower_bound_nm", np.nan)))
        llm_highs.append(float(item.get("upper_bound_nm", np.nan)))
        raw_responses.append(content)

    retrieval_eval_metrics = normalized_metrics(y_eval, retrieval_eval_pred)
    llm_metrics = normalized_metrics(y_eval, llm_preds)

    summary = pd.DataFrame([
        {"Model": "Retrieval-kNN (all test)", **retrieval_all_metrics, "Samples": len(y_test)},
        {"Model": "Retrieval-kNN (LLM subset)", **retrieval_eval_metrics, "Samples": eval_count},
        {"Model": "LLM + retrieved examples", **llm_metrics, "Samples": eval_count},
    ])
    summary_path = "results_table/performance_table_llm_ek_c.csv"
    summary.to_csv(summary_path, index=False)

    detail = X_eval.copy()
    detail["Actual_nm"] = y_eval
    detail["Retrieval_Pred_nm"] = retrieval_eval_pred
    detail["LLM_Pred_nm"] = llm_preds
    detail["LLM_Lower_nm"] = llm_lows
    detail["LLM_Upper_nm"] = llm_highs
    detail["LLM_Raw_Response"] = raw_responses
    detail_path = "results_table/llm_predictions_ek_c.csv"
    detail.to_csv(detail_path, index=False)

    plot_path = "plots/test_output_llm_vs_retrieval.png"
    save_target_prediction_plot(
        y_true=y_eval,
        retrieval_pred=np.array(retrieval_eval_pred),
        llm_pred=np.array(llm_preds),
        y_scaler=y_scaler,
        out_file=plot_path,
    )
    raw_plot_path = "plots/test_output_llm_vs_retrieval_raw_nm.png"
    save_target_prediction_plot_unnormalized(
        y_true=np.array(y_eval),
        retrieval_pred=np.array(retrieval_eval_pred),
        llm_pred=np.array(llm_preds),
        out_file=raw_plot_path,
    )

    print("\nLLM regression comparison on ek_c dataset\n")
    print(summary.to_string(index=False, formatters={
        "MSE": "{:.4f}".format,
        "MAE": "{:.4f}".format,
        "RMSE": "{:.4f}".format,
        "R2": "{:.4f}".format,
    }))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved predictions: {detail_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved plot: {raw_plot_path}")

if __name__ == "__main__":
    run()
