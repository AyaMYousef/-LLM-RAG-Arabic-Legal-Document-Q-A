"""Log RAGAS evaluation results to MLflow and run the faithfulness alert."""

import json
import subprocess
import sys
from pathlib import Path

import mlflow

RESULTS_FILE = Path("data/evaluation/ragas_results.json")

mlflow.set_experiment("arabic-legal-rag")

with RESULTS_FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

metrics = data["metrics"]

with mlflow.start_run(run_name="ragas-baseline"):
    mlflow.log_metrics({
        "faithfulness":      metrics["faithfulness"],
        "context_precision": metrics["context_precision"],
        "context_recall":    metrics["context_recall"],
        "answer_relevancy":  metrics["answer_relevancy"],
    })

    mlflow.log_param("embedding_model", "intfloat/multilingual-e5-base")
    mlflow.log_param("llm_model",       "qwen/qwen3.8-27b")
    mlflow.log_param("num_samples",     data["num_samples"])
    mlflow.log_param("evaluation_type", "RAGAS baseline")

    mlflow.log_artifact(str(RESULTS_FILE))

print("RAGAS metrics logged to MLflow.")

# ── Faithfulness alert (threshold 0.80) ───────────────────────────────────
# Exits with code 1 when faithfulness < 0.80, propagating the failure
# to CI or any calling process.
alert_script = Path(__file__).resolve().parent / "faithfulness_alert.py"
result = subprocess.run(
    [sys.executable, str(alert_script), "--threshold", "0.80"],
    check=False,
)
sys.exit(result.returncode)
