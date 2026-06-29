# building-llm-from-scratch

Working through *Build a Large Language Model (From Scratch)* by Sebastian Raschka —
implementing a GPT-style model end to end: tokenization, attention, pretraining,
and fine-tuning.

Each chapter lives in its own directory with a `main.ipynb` notebook.

## Key takeaways

### Ch 2 — Working with text data
- LLMs can't read raw text; text is split into **tokens**, mapped to integer IDs, then to
  **embedding vectors** that are learned during training.
- **Byte-pair encoding (BPE)** handles unknown words by breaking them into subword units,
  so the vocabulary stays fixed while still covering any input.
- Training data is generated with a **sliding window**: each input chunk is paired with the
  same chunk shifted one token to the right (the next-token targets).
- **Token embeddings + positional embeddings** are summed so the model knows both *what* a
  token is and *where* it sits in the sequence.

### Ch 3 — Attention mechanisms
- **Self-attention** lets every token weigh the importance of every other token, producing
  context-aware representations.
- Scaled dot-product attention uses learned **query, key, and value** projections; scores are
  scaled by √dₖ and softmaxed into attention weights.
- **Causal (masked) attention** hides future tokens so the model can only attend backward —
  essential for autoregressive generation.
- **Multi-head attention** runs several attention computations in parallel, letting each head
  focus on different relationships, then concatenates the results.

### Ch 4 — Implementing the GPT model
- A GPT model is a stack of identical **transformer blocks**: multi-head attention +
  feed-forward network, each wrapped with layer norm and a residual (shortcut) connection.
- **Layer normalization** stabilizes training; **GELU** is the activation used in the
  feed-forward layers.
- **Residual connections** let gradients flow through deep stacks without vanishing.
- The full 124M-parameter GPT-2 architecture is assembled and can already generate text —
  just incoherently, because it's untrained.

### Ch 5 — Pretraining on unlabeled data
- The training objective is **next-token prediction**, measured with **cross-entropy loss**
  (and perplexity as a readable proxy).
- A standard training loop (forward → loss → backprop → AdamW step) is enough to make the
  model produce coherent text.
- Decoding strategies — **temperature scaling** and **top-k sampling** — trade off between
  predictable and diverse generations.
- Pretraining from scratch is expensive, so we **load OpenAI's pretrained GPT-2 weights**
  into our own implementation instead.

### Ch 6 — Fine-tuning for classification
- A pretrained LLM is adapted to a task (spam vs. ham) by **replacing the output head** with a
  classification layer and fine-tuning.
- You don't need to retrain everything — fine-tuning the **last transformer block + final
  norm + new head** is often enough.
- Classification uses the logits of the **last token** in the sequence.
- Even a modest model reaches high accuracy quickly, showing the power of transfer learning.

### Ch 7 — Fine-tuning to follow instructions
- **Instruction fine-tuning** turns a base model into an assistant by training on
  (instruction, input, response) triples formatted with a consistent prompt template
  (Alpaca-style).
- Batches are built with **custom collation**: padding to equal length and **masking padding
  tokens** (-100) so they don't contribute to the loss.
- The model is trained on the full formatted prompt but evaluated on its generated responses.
- Output quality can be scored automatically by prompting another (larger) LLM as a judge.

## Setup

```bash
poetry install
poetry run jupyter lab
```

Datasets and model weights are downloaded by the notebooks at runtime and are git-ignored.

## Verdict

Finished reading and implementing a working LLM. **4/5** — a good introduction. It doesn't go
deep into the theoretical details but covers a solid amount, and the implementation and code
details are clean.
