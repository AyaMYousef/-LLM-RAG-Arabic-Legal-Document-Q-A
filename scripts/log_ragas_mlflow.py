from pathlib import Path
import json
import mlflow

RESULTS_FILE = Path("data/evaluation/ragas_results.json")

mlflow.set_experiment("arabic-legal-rag")

with RESULTS_FILE.open("r", encoding="utf-8") as f:
    data = json.load(f)

metrics = data["metrics"]

with mlflow.start_run(run_name="ragas-baseline"):
    mlflow.log_metrics({
        "faithfulness": metrics["faithfulness"],
        "context_precision": metrics["context_precision"],
        "context_recall": metrics["context_recall"],
    })

    mlflow.log_param("embedding_model", "intfloat/multilingual-e5-base")
    mlflow.log_param("llm_model", "qwen/qwen3.8-27b")
    mlflow.log_param("num_samples", data["num_samples"])
    mlflow.log_param("evaluation_type", "RAGAS baseline")

    mlflow.log_artifact(str(RESULTS_FILE))

print("RAGAS metrics logged to MLflow.")