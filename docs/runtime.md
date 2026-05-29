# Runtime Setup Notes

The detector uses two runtime components:

- spaCy runs the lightweight English parsing pipeline on CPU.
- Sentence Transformers can use CUDA when a compatible PyTorch installation is available.

XGBoost defaults to CPU because the engineered feature matrix is compact. This avoids unnecessary GPU memory pressure when MiniLM embeddings are already using the GPU.
