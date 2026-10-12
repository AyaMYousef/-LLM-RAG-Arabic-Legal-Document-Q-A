import os
from pathlib import Path

# Set before importing torch.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import nn

MODEL_NAME = "intfloat/multilingual-e5-base"
OUTPUT_PATH = Path("data/processed/e5_onnx/model.onnx")


class E5EmbeddingModel(nn.Module):
    """Transformer + attention-mask mean pooling + L2 normalization."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer = transformer

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )

        token_embeddings = outputs[0]

        mask = attention_mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)

        summed_embeddings = (token_embeddings * mask).sum(dim=1)
        token_counts = mask.sum(dim=1).clamp(min=1e-9)
        sentence_embeddings = summed_embeddings / token_counts

        return F.normalize(sentence_embeddings, p=2, dim=1)


def main():
    torch.manual_seed(42)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME} on CPU...")

    sentence_model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    transformer = sentence_model[0].auto_model.to("cpu").eval()
    tokenizer = sentence_model.tokenizer

    model = E5EmbeddingModel(transformer).to("cpu").eval()

    assert next(model.parameters()).device.type == "cpu"
    print("PASS: model is on CPU")

    sample_texts = [
        "passage: Egyptian Civil Code article about contracts.",
        "query: What are the rules governing contracts?",
    ]

    encoded = tokenizer(
        sample_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to("cpu")
    attention_mask = encoded["attention_mask"].to("cpu")

    # Export with dynamic batch and sequence dimensions.
    with torch.no_grad():
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            str(OUTPUT_PATH),
            input_names=["input_ids", "attention_mask"],
            output_names=["embeddings"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {
                    0: "batch_size",
                    1: "sequence_length",
                },
                "embeddings": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )

    print(f"Exported model: {OUTPUT_PATH}")

    # 1. Validate the ONNX graph.
    onnx_model = onnx.load(str(OUTPUT_PATH))
    onnx.checker.check_model(onnx_model, full_check=True)
    print("PASS: onnx.checker.check_model")

    # 2. Validate dynamic batch inference.
    session = ort.InferenceSession(
        str(OUTPUT_PATH),
        providers=["CPUExecutionProvider"],
    )

    test_texts = [
        "query: What is a contract?",
        "passage: Contracts are governed by the applicable law.",
        "query: Explain contractual obligations.",
    ]

    onnx_embeddings = None

    for batch_size in (1, 2, 3):
        batch_texts = test_texts[:batch_size]

        test_inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )

        result = session.run(
            ["embeddings"],
            {
                "input_ids": test_inputs["input_ids"].astype(np.int64),
                "attention_mask": test_inputs["attention_mask"].astype(np.int64),
            },
        )[0]

        assert result.shape == (batch_size, 768), (
            f"Unexpected output shape for batch {batch_size}: {result.shape}"
        )
        assert np.isfinite(result).all()

        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(
            norms,
            np.ones(batch_size),
            rtol=1e-4,
            atol=1e-4,
        )

        print(f"PASS: batch size {batch_size}, shape {result.shape}")

        if batch_size == 3:
            onnx_embeddings = result

    # 3. Compare ONNX with the same PyTorch wrapper and inputs.
    reference_inputs = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    with torch.no_grad():
        reference_embeddings = (
            model(
                reference_inputs["input_ids"].cpu(),
                reference_inputs["attention_mask"].cpu(),
            )
            .cpu()
            .numpy()
        )

    max_abs_error = np.max(np.abs(onnx_embeddings - reference_embeddings))
    print(f"Maximum absolute embedding error: {max_abs_error:.8f}")

    np.testing.assert_allclose(
        onnx_embeddings,
        reference_embeddings,
        rtol=1e-3,
        atol=1e-4,
    )

    print("PASS: ONNX embeddings match the PyTorch wrapper")
    print("ONNX export and verification completed successfully.")


if __name__ == "__main__":
    main()
