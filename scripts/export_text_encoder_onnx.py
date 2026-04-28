#!/usr/bin/env python3
"""
Export sentence-transformers/all-MiniLM-L6-v2 to ONNX format.

Produces a model that takes tokenized input (input_ids, attention_mask)
and outputs 384-dim sentence embeddings using mean pooling.

Strategy: Use torch.onnx.export with dynamo=False (legacy TorchScript exporter)
which properly handles dynamic_axes. Work around the new transformers masking
issue by pre-computing the extended attention mask before tracing.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModel

OUTPUT_PATH = os.environ.get("STRIFF_DATA_DIR", "./data") + "/models/text_encoder.onnx"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceEncoderForExport(nn.Module):
    """BertModel + mean pooling, with pre-computed attention mask expansion.

    The new transformers library (v5+) has masking logic that doesn't trace well.
    We pre-compute the extended attention mask (4D) and pass it through,
    bypassing the problematic _create_attention_masks method.
    """

    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.embeddings = self.bert.embeddings
        self.encoder = self.bert.encoder
        self.pooler = self.bert.pooler

    def _prepare_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Convert 2D attention mask to 4D extended attention mask.

        This replicates what BertModel does internally but in a traceable way.
        """
        # attention_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
        extended = attention_mask[:, None, None, :]
        # Convert 0/1 mask to 0/-inf
        extended = extended.to(dtype=torch.float32)
        extended = (1.0 - extended) * torch.finfo(torch.float32).min
        return extended

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor) -> torch.Tensor:
        # Embeddings
        embedding_output = self.embeddings(input_ids=input_ids)

        # Extended attention mask
        extended_attention_mask = self._prepare_attention_mask(attention_mask)

        # Encoder
        encoder_output = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
        )
        sequence_output = encoder_output.last_hidden_state  # (batch, seq, 384)

        # Mean pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
        summed = torch.sum(sequence_output * mask_expanded, dim=1)
        counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        embeddings = summed / counts  # (batch, 384)

        return embeddings


def main():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    encoder = SentenceEncoderForExport(MODEL_NAME)
    encoder.eval()

    # Prepare dummy input
    test_sentence = "This is a test sentence for ONNX export validation."
    encoded = tokenizer(test_sentence, return_tensors="pt", padding="max_length",
                        max_length=128, truncation=True)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    # Export using legacy TorchScript exporter with proper dynamic axes
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"Exporting ONNX model to: {OUTPUT_PATH}")

    torch.onnx.export(
        encoder,
        (input_ids, attention_mask),
        OUTPUT_PATH,
        input_names=["input_ids", "attention_mask"],
        output_names=["embeddings"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "attention_mask": {0: "batch_size", 1: "seq_len"},
            "embeddings": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print("Export complete.")

    # Report file size
    file_size = os.path.getsize(OUTPUT_PATH)
    print(f"\nFile size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

    # Validate: PyTorch inference
    with torch.no_grad():
        pytorch_output = encoder(input_ids, attention_mask).numpy()

    # Validate: ONNX Runtime inference
    session = ort.InferenceSession(OUTPUT_PATH, providers=["CPUExecutionProvider"])
    onnx_output = session.run(["embeddings"], {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
    })[0]

    # Compare
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_output - onnx_output))
    print(f"\n=== Validation ===")
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"ONNX output shape:    {onnx_output.shape}")
    print(f"Max absolute diff:    {max_diff:.2e}")
    print(f"Mean absolute diff:   {mean_diff:.2e}")

    # Report ONNX model input/output details
    print(f"\n=== ONNX Model Details ===")
    print("Inputs:")
    for inp in session.get_inputs():
        print(f"  name={inp.name}, shape={inp.shape}, type={inp.type}")
    print("Outputs:")
    for out in session.get_outputs():
        print(f"  name={out.name}, shape={out.shape}, type={out.type}")

    # Test dynamic batch sizes and sequence lengths
    print(f"\n=== Dynamic Shape Validation ===")
    for batch in [1, 2, 3, 5]:
        for seq in [16, 64, 128]:
            ids = np.random.randint(0, 1000, (batch, seq)).astype(np.int64)
            mask = np.ones((batch, seq), dtype=np.int64)
            out = session.run(["embeddings"], {"input_ids": ids, "attention_mask": mask})[0]
            print(f"  batch={batch}, seq={seq} -> output shape: {out.shape}")
            assert out.shape == (batch, 384), f"Expected ({batch}, 384), got {out.shape}"

    # Verify against the original sentence-transformers output
    print(f"\n=== Sentence-Transformers Parity Check ===")
    from transformers import AutoModel as AM2

    orig_model = AM2.from_pretrained(MODEL_NAME)
    orig_model.eval()
    with torch.no_grad():
        orig_output = orig_model(input_ids=input_ids, attention_mask=attention_mask)
        # Manual mean pooling (same as sentence-transformers)
        tok_emb = orig_output.last_hidden_state
        mask_exp = attention_mask.unsqueeze(-1).expand(tok_emb.size()).float()
        orig_pooled = torch.sum(tok_emb * mask_exp, dim=1) / torch.clamp(mask_exp.sum(dim=1), min=1e-9)

    parity_diff = np.max(np.abs(orig_pooled.numpy() - onnx_output))
    print(f"Original model vs ONNX max diff: {parity_diff:.2e}")
    assert parity_diff < 1e-4, f"Parity diff {parity_diff} exceeds 1e-4"
    print("Parity check passed.")

    # Final summary
    print(f"\n{'='*60}")
    print(f"=== Summary ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Output: {OUTPUT_PATH}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"Inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.type} shape={inp.shape}")
    print(f"Outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.type} shape={out.shape}")
    print(f"PyTorch vs ONNX max diff: {max_diff:.2e}")
    print(f"Original vs ONNX max diff: {parity_diff:.2e}")
    print(f"Dynamic shapes: batch_size (any >= 1), seq_len (any >= 1)")
    print(f"Pooling: mean pooling over token embeddings")
    print(f"Embedding dim: 384")
    print(f"{'='*60}")

    print("\nDone. ONNX text encoder exported successfully.")


if __name__ == "__main__":
    main()
