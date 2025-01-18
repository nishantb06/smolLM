# [SmolLM](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)
Reverse Engineering SmolLM 135M
Writing the model from scratch, by analysing its config and code. 
Training it on the CosmoCorpus datasetf from scratch

## Deductions from the model info above
1. The model is a GPT-2 like model with 30 layers, 9 heads, and 576 embedding dimensions.
2. The model has activation function of Silu.
3. The model has a vocabulary size of 49152.
4. Max positional embeddings is 8192
4. The model has a attention dropout rate of 0.0
5. The model uses RMSNorm for layer normalization instead of LayerNorm.
6. Model uses Rotary Embeddings for positional embeddings
7. the Hidden dimension of the mlp is 1536