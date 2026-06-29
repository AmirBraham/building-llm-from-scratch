"""Generates main.ipynb for Chapter 7 (instruction fine-tuning)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Chapter 7 — Instruction Fine-Tuning

Teach pretrained GPT-2 to **follow instructions**. Unlike Ch6 (classification),
we keep the full LM head and train on next-token prediction over a fixed
prompt template — but mask padding so the loss focuses on the response.

Pipeline:
1. Download the 1,100-example instruction dataset, split train/val/test
2. Format each example with the Alpaca-style template
3. `InstructionDataset` + a **custom collate** (per-batch padding, target shift, `-100` masking)
4. Load pretrained GPT-2 (124M) — no architecture change
5. Train (next-token loss), tracking + plotting train/val curves
6. Generate responses on the held-out test set
""")

md("## 1. Setup")
code("""
import json, urllib.request, ssl, certifi, functools, time
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import tiktoken

from chapter04 import GPTModel, generate_text_simple
from gpt_download import load_pretrained_gpt2

torch.manual_seed(123)
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")
device
""")

md("## 2. Data: download + split (85% / 5% / 10%)")
code("""
file_path = "instruction-data.json"
url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/"
       "ch07/01_main-chapter-code/instruction-data.json")
try:
    data = json.load(open(file_path))
except FileNotFoundError:
    ctx = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, context=ctx) as r:
        data = json.loads(r.read().decode())
    json.dump(data, open(file_path, "w"), indent=2)

n = len(data)
train_portion = int(n * 0.85)
test_portion  = int(n * 0.10)
train_data = data[:train_portion]
test_data  = data[train_portion:train_portion + test_portion]
val_data   = data[train_portion + test_portion:]
len(train_data), len(val_data), len(test_data)
""")

md("""
## 3. Prompt template

Alpaca-style. The `### Input:` block is only added when the example has an input.
The **response** is appended separately (we need to know where it starts).
""")
code("""
def format_input(entry):
    text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\\n\\n### Instruction:\\n{entry['instruction']}"
    )
    if entry["input"]:
        text += f"\\n\\n### Input:\\n{entry['input']}"
    return text

print(format_input(train_data[50]) + f"\\n\\n### Response:\\n{train_data[50]['output']}")
""")

md("""
## 4. Dataset + custom collate

`InstructionDataset` pre-tokenizes the full `prompt + response` string.
`custom_collate`:
- pads each batch to its **own** longest sequence (dynamic padding)
- builds targets by shifting inputs left by one
- keeps the first `<|endoftext|>` as a stop signal, masks the rest with `-100`
""")
code("""
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded = []
        for entry in data:
            full = format_input(entry) + f"\\n\\n### Response:\\n{entry['output']}"
            self.encoded.append(tokenizer.encode(full))
    def __getitem__(self, i): return self.encoded[i]
    def __len__(self): return len(self.data)

def custom_collate(batch, pad_token_id=50256, ignore_index=-100,
                   allowed_max_length=1024, device="cpu"):
    batch_max = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new = item + [pad_token_id]
        padded = new + [pad_token_id] * (batch_max - len(new))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        idx = torch.nonzero(mask).squeeze()
        if idx.numel() > 1:
            targets[idx[1:]] = ignore_index      # mask all pads except the first
        inputs = inputs[:allowed_max_length]
        targets = targets[:allowed_max_length]
        inputs_lst.append(inputs); targets_lst.append(targets)
    return torch.stack(inputs_lst).to(device), torch.stack(targets_lst).to(device)
""")
code("""
collate = functools.partial(custom_collate, device=device, allowed_max_length=1024)
batch_size = 8

train_loader = DataLoader(InstructionDataset(train_data, tokenizer), batch_size=batch_size,
                          collate_fn=collate, shuffle=True,  drop_last=True)
val_loader   = DataLoader(InstructionDataset(val_data, tokenizer), batch_size=batch_size,
                          collate_fn=collate, shuffle=False, drop_last=False)
test_loader  = DataLoader(InstructionDataset(test_data, tokenizer), batch_size=batch_size,
                          collate_fn=collate, shuffle=False, drop_last=False)

xb, yb = next(iter(train_loader))
print("inputs:", xb.shape, "targets:", yb.shape)
print("targets row (note -100 padding mask):\\n", yb[0])
""")

md("""
## 5. Load pretrained GPT-2 (124M)

No architecture surgery — we fine-tune the whole model with its original LM head.
(The book uses the 355M model for nicer answers; 124M trains faster and still learns the format.)
""")
code("""
model = load_pretrained_gpt2().to(device)
sum(p.numel() for p in model.parameters())
""")

md("## 6. Loss — cross-entropy with `-100` ignored automatically")
code("""
def calc_loss_batch(input_batch, target_batch, model, device):
    logits = model(input_batch.to(device))
    return torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.to(device).flatten()
    )  # ignore_index=-100 by default -> padded positions skipped

def calc_loss_loader(loader, model, device, num_batches=None):
    total = 0.0
    n = min(num_batches or len(loader), len(loader))
    for i, (x, y) in enumerate(loader):
        if i >= n: break
        total += calc_loss_batch(x, y, model, device).item()
    return total / n

with torch.no_grad():
    print("initial val loss:", calc_loss_loader(val_loader, model, device, 5))
""")

md("## 7. Training")
code("""
from tqdm.auto import tqdm

def train_model(model, train_loader, val_loader, optimizer, device,
                num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, track_steps = [], [], []
    step = -1
    total_steps = num_epochs * len(train_loader)
    pbar = tqdm(total=total_steps, desc="training")   # live progress + ETA
    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(x, y, model, device)
            loss.backward()
            optimizer.step()
            step += 1
            pbar.update(1)
            if step % eval_freq == 0:
                model.eval()
                with torch.no_grad():
                    tl = calc_loss_loader(train_loader, model, device, eval_iter)
                    vl = calc_loss_loader(val_loader, model, device, eval_iter)
                model.train()
                train_losses.append(tl); val_losses.append(vl); track_steps.append(step)
                pbar.set_postfix(epoch=epoch+1, train=f"{tl:.3f}", val=f"{vl:.3f}")
    pbar.close()
    return train_losses, val_losses, track_steps
""")
code("""
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
t0 = time.time()
train_losses, val_losses, track_steps = train_model(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=2, eval_freq=20, eval_iter=5)
print(f"training time: {(time.time()-t0)/60:.1f} min")
""")

md("## 8. Plot the loss curves")
code("""
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(track_steps, train_losses, label="train loss")
ax.plot(track_steps, val_losses, label="val loss", linestyle="--")
ax.set_xlabel("training step"); ax.set_ylabel("cross-entropy loss")
ax.set_title("Instruction fine-tuning loss"); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
""")

md("""
## 9. Generate responses on held-out test instructions

Format the prompt exactly as in training, generate, then slice off everything
up to `### Response:` to read just the model's answer.
""")
code("""
def generate_response(entry, model, tokenizer, device, max_new_tokens=256):
    prompt = format_input(entry)
    ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
    out = generate_text_simple(model, ids, max_new_tokens=max_new_tokens, context_size=1024)
    full = tokenizer.decode(out[0].tolist())
    return full[len(prompt):].replace("### Response:", "").strip()

model.eval()
for entry in test_data[:3]:
    print("INSTRUCTION:", entry["instruction"])
    if entry["input"]: print("INPUT:", entry["input"])
    print("EXPECTED :", entry["output"])
    print("MODEL    :", generate_response(entry, model, tokenizer, device))
    print("-" * 70)
""")

md("""
## 10. Save model + generated answers

The book then scores these answers with another LLM (e.g. Llama via Ollama).
We save the responses so that evaluation can run separately.
""")
code("""
out_path = "instruction-data-with-response.json"
results = []
model.eval()
for entry in test_data:
    e = dict(entry)
    e["model_response"] = generate_response(entry, model, tokenizer, device)
    results.append(e)
json.dump(results, open(out_path, "w"), indent=2)
torch.save(model.state_dict(), "gpt2-instruct.pth")
print("saved", out_path, "and gpt2-instruct.pth")
""")

nb["cells"] = cells
with open("main.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote main.ipynb with", len(cells), "cells")
