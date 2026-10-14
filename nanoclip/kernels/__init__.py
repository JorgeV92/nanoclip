"""Optional GPU kernels. Add future softmax/LayerNorm/matmul experiments here.

Keep imports lazy at the operation boundary so CPU and MPS require no Triton install.
"""
