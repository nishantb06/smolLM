# [SmolLM](https://huggingface.co/HuggingFaceTB/SmolLM2-135M)

Reverse Engineering SmolLM 135M
Writing the model from scratch, by analysing its config and code.
Training it on the CosmoCorpus datasetf from scratch

## Deductions from the model info above

1. The model is a GPT-2 like model with 30 layers, 9 heads, and 576 embedding dimensions.
2. The model has activation function of Silu.
3. The model has a vocabulary size of 49152.
4. Max positional embeddings is 8192
5. The model has a attention dropout rate of 0.0
6. The model uses RMSNorm for layer normalization instead of LayerNorm.
7. Model uses Rotary Embeddings for positional embeddings
8. the Hidden dimension of the mlp is 1536

## Original model architecture

```
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(49152, 576)
    (layers): ModuleList(
      (0-29): 30 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=576, out_features=576, bias=False)
          (k_proj): Linear(in_features=576, out_features=192, bias=False)
          (v_proj): Linear(in_features=576, out_features=192, bias=False)
          (o_proj): Linear(in_features=576, out_features=576, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
          (up_proj): Linear(in_features=576, out_features=1536, bias=False)
          (down_proj): Linear(in_features=1536, out_features=576, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((576,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((576,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((576,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)
```

## My model architecture- v1

```
SmolLM(
  (wte): Embedding(49152, 576)
  (wpe): Embedding(1024, 576)
  (drop): Dropout(p=0.1, inplace=False)
  (blocks): ModuleList(
    (0-29): 30 x DecoderBlockWithLayerNorm(
      (ln_1): LayerNorm((576,), eps=1e-05, elementwise_affine=True)
      (attn): CausalMultiHeadAttention(
        (c_attn): Linear(in_features=576, out_features=1728, bias=True)
        (c_proj): Linear(in_features=576, out_features=576, bias=True)
        (attn_dropout): Dropout(p=0.0, inplace=False)
        (resid_dropout): Dropout(p=0.1, inplace=False)
      )
      (ln_2): LayerNorm((576,), eps=1e-05, elementwise_affine=True)
      (mlp): MLP(
        (c_fc): Linear(in_features=576, out_features=1536, bias=True)
        (silu): SiLU()
        (c_proj): Linear(in_features=1536, out_features=576, bias=True)
      )
    )
  )
  (ln_f): LayerNorm((576,), eps=1e-05, elementwise_affine=True)
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)
```

- Clear differences are in the attention , mlp layers and that I havent implemented the rotary embeddings (Brings a diff of 0.5M parameters)
- The original model projects the K and V to 192 dimensions and Q to 576 dimensions, whereas I have projected all of them 576 dimensions.
- There are also differences in the implementation of MLP layers. the original model uses a gate and up and down projection, whereas I have used a single linear layer. Need to figure out what gate and up and down projection is doing.

### My Model Architecture - v2

- I have also implemented the gate and up and down projection in the MLP layer.
- I have also implemented the RMSNorm for layer normalization.
- I have also implemented the weight sharing between the input embeddings and the output linear layer.
- TODO : implementing the rotary embeddings which will bring down the number of parameters to 134.67M

**FINAL Number of parameters: 135.26M**

```
SmolLM(
  (wte): Embedding(49152, 576)
  (wpe): Embedding(1024, 576)
  (drop): Dropout(p=0.1, inplace=False)
  (blocks): ModuleList(
    (0-29): 30 x DecoderBlockWithRMSNorm(
      (rms_1): RMSNorm()
      (attn): CausalMultiHeadAttention(
        (w_q): Linear(in_features=576, out_features=576, bias=True)
        (w_k): Linear(in_features=576, out_features=192, bias=True)
        (w_v): Linear(in_features=576, out_features=192, bias=True)
        (c_proj): Linear(in_features=576, out_features=576, bias=True)
        (resid_dropout): Dropout(p=0.1, inplace=False)
      )
      (rms_2): RMSNorm()
      (mlp): LlamaMLP(
        (w1): Linear(in_features=576, out_features=1536, bias=True)
        (w2): Linear(in_features=1536, out_features=576, bias=True)
        (w3): Linear(in_features=576, out_features=1536, bias=True)
      )
    )
  )
  (rms_norm): RMSNorm()
  (lm_head): Linear(in_features=576, out_features=49152, bias=False)
)
```

### Future work

- Implementing the rotary embeddings
- Implementing KV cache for the attention layer

## Overfitting on a single batch

Starting loss : 10.89
Ending loss : 0.40

Overfitting a small batch is a crucial debugging step when building deep learning models. Here's why:

Model Validation: Ensures that your model architecture, loss function, and data pipeline are set up correctly. If your model can't overfit a tiny dataset, it likely won't scale to larger datasets.
Hyperparameter Sanity Check: Helps identify if learning rates, weight initializations, or other hyperparameters are preventing effective learning.
Gradient Flow: Confirms gradients are flowing properly through the network, and there are no issues like exploding/vanishing gradients.
Baseline Benchmark: Provides a minimum performance target to confirm the model’s ability to learn.
The goal of overfitting is to see if the model can drive the loss close to zero on a small batch of data. Failure to overfit often indicates bugs or misconfigurations in the model setup.
