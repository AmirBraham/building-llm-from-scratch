"""Generates main.ipynb for Chapter 6 (classification fine-tuning)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Chapter 6 — Fine-tuning for Classification

Turn the pretrained GPT-2 into a **spam classifier**. Pipeline:
1. Download + balance + split the SMS spam dataset
2. Tokenize and pad into fixed-length tensors (`Dataset` / `DataLoader`)
3. Load pretrained GPT-2 weights into our from-scratch `GPTModel`
4. **Surgery:** replace the output head (768 → 2) and freeze most of the body
5. Define classification loss + accuracy (read off the **last token**)
6. Train and evaluate
""")

md("## 1. Setup")
code("""
import torch
import torch.nn as nn
import pandas as pd
import tiktoken
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")
device
""")

md("""
## 2. Data: download, balance, split

The raw data is ~6.5:1 imbalanced (4825 ham / 747 spam). We **undersample ham**
to 747 so the model can't win by always guessing the majority class.
""")
code("""
import urllib.request, zipfile, os, ssl, certifi
from pathlib import Path

data_file = Path("sms_spam_collection/SMSSpamCollection.tsv")
if not data_file.exists():
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    ctx = ssl.create_default_context(cafile=certifi.where())  # verified TLS
    with urllib.request.urlopen(url, context=ctx) as r:
        open("sms_spam_collection.zip","wb").write(r.read())
    with zipfile.ZipFile("sms_spam_collection.zip") as z:
        z.extractall("sms_spam_collection")
    os.rename("sms_spam_collection/SMSSpamCollection", data_file)

df = pd.read_csv(data_file, sep="\\t", header=None, names=["Label","Text"])
df["Label"].value_counts()
""")
code("""
def balance(df):
    n_spam = (df["Label"] == "spam").sum()
    ham = df[df["Label"]=="ham"].sample(n_spam, random_state=123)
    return pd.concat([ham, df[df["Label"]=="spam"]])

balanced = balance(df)
balanced["Label"] = balanced["Label"].map({"ham": 0, "spam": 1})
balanced["Label"].value_counts()
""")
code("""
def split(df, train_frac=0.7, val_frac=0.1):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    n = len(df); a = int(n*train_frac); b = a + int(n*val_frac)
    return df[:a], df[a:b], df[b:]

train_df, val_df, test_df = split(balanced)
for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
    d.to_csv(f"{name}.csv", index=None)
len(train_df), len(val_df), len(test_df)
""")

md("""
## 3. Dataset & DataLoader

Tokenize each message, then pad to `max_length` with the `<|endoftext|>` token
(50256). **`max_length` is derived from the *training* set only** — val/test are
padded/truncated to that same length (no leakage from the data we evaluate on).
""")
code("""
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded = [tokenizer.encode(t) for t in self.data["Text"]]
        if max_length is None:
            self.max_length = max(len(e) for e in self.encoded)
        else:
            self.max_length = max_length
            self.encoded = [e[:max_length] for e in self.encoded]      # truncate
        self.encoded = [e + [pad_token_id]*(self.max_length - len(e))   # pad
                        for e in self.encoded]
    def __getitem__(self, i):
        return (torch.tensor(self.encoded[i]),
                torch.tensor(self.data.iloc[i]["Label"]))
    def __len__(self):
        return len(self.data)

train_ds = SpamDataset("train.csv", tokenizer)
val_ds   = SpamDataset("val.csv",  tokenizer, max_length=train_ds.max_length)
test_ds  = SpamDataset("test.csv", tokenizer, max_length=train_ds.max_length)
print("max_length (from train):", train_ds.max_length)

train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=8, shuffle=False, drop_last=False)

xb, yb = next(iter(train_loader))
xb.shape, yb.shape
""")

md("""
## 4. Load pretrained GPT-2 + surgery

`load_pretrained_gpt2()` (in `gpt_download.py`) maps the official GPT-2 (124M)
weights into our `GPTModel`. Then:
- **Freeze** every parameter
- **Replace** `out_head` with `Linear(768, 2)` — random, trainable
- **Unfreeze** the final transformer block + final LayerNorm

Only ~5.7% of params end up trainable.
""")
code("""
from gpt_download import load_pretrained_gpt2

model = load_pretrained_gpt2()
for p in model.parameters():
    p.requires_grad = False

model.out_head = nn.Linear(768, 2)
for p in model.trf_blocks[-1].parameters():
    p.requires_grad = True
for p in model.final_norm.parameters():
    p.requires_grad = True

model.to(device)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
""")

md("""
## 5. Loss & accuracy — read off the LAST token

Causal attention means only the **last** token has attended over the whole
message, so its hidden state is the classification summary.

- **Loss:** `cross_entropy(logits[:, -1, :], target)` — differentiable, we backprop it.
- **Accuracy:** `argmax(logits[:, -1, :])` vs target — *not* differentiable, monitor only.
  (No softmax needed: argmax of logits == argmax of softmax.)
""")
code("""
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]          # last-token logits -> [batch, 2]
    return torch.nn.functional.cross_entropy(logits, target_batch)

def calc_loss_loader(loader, model, device, num_batches=None):
    total = 0.0
    n = min(num_batches or len(loader), len(loader))
    for i, (x, y) in enumerate(loader):
        if i >= n: break
        total += calc_loss_batch(x, y, model, device).item()
    return total / n

@torch.no_grad()
def calc_accuracy_loader(loader, model, device, num_batches=None):
    model.eval()
    correct = seen = 0
    n = min(num_batches or len(loader), len(loader))
    for i, (x, y) in enumerate(loader):
        if i >= n: break
        x, y = x.to(device), y.to(device)
        preds = model(x)[:, -1, :].argmax(dim=-1)
        correct += (preds == y).sum().item()
        seen += y.numel()
    return correct / seen
""")
code("""
# Baseline BEFORE training — random head, expect ~50%
print("init train acc:", calc_accuracy_loader(train_loader, model, device, 5))
print("init val   acc:", calc_accuracy_loader(val_loader,   model, device, 5))
""")

md("## 6. Training loop")
code("""
def train_classifier(model, train_loader, val_loader, optimizer, device,
                     num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(x, y, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += x.shape[0]
            global_step += 1
            if global_step % eval_freq == 0:
                model.eval()
                tl = calc_loss_loader(train_loader, model, device, eval_iter)
                vl = calc_loss_loader(val_loader,   model, device, eval_iter)
                model.train()
                train_losses.append(tl); val_losses.append(vl)
                print(f"ep {epoch+1} step {global_step:04d} | "
                      f"train loss {tl:.3f} | val loss {vl:.3f}")
        ta = calc_accuracy_loader(train_loader, model, device, eval_iter)
        va = calc_accuracy_loader(val_loader,   model, device, eval_iter)
        train_accs.append(ta); val_accs.append(va)
        print(f"== epoch {epoch+1}: train acc {ta:.3f} | val acc {va:.3f} ==")
    return train_losses, val_losses, train_accs, val_accs
""")
code("""
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
results = train_classifier(model, train_loader, val_loader, optimizer, device,
                           num_epochs=5, eval_freq=50, eval_iter=5)
""")

md("## 7. Final evaluation on the test set (touched once)")
code("""
print("train acc:", calc_accuracy_loader(train_loader, model, device))
print("val   acc:", calc_accuracy_loader(val_loader,   model, device))
print("test  acc:", calc_accuracy_loader(test_loader,  model, device))
""")

md("## 8. Use it")
code("""
def classify(text, model, tokenizer, device, max_length):
    model.eval()
    ids = tokenizer.encode(text)[:max_length]
    ids += [50256] * (max_length - len(ids))
    x = torch.tensor(ids, device=device).unsqueeze(0)
    with torch.no_grad():
        pred = model(x)[:, -1, :].argmax(dim=-1).item()
    return "spam" if pred == 1 else "not spam"

print(classify("You are a winner! Claim your FREE prize now, text WIN to 80086",
               model, tokenizer, device, train_ds.max_length))
print(classify("Hey, are we still meeting for lunch tomorrow?",
               model, tokenizer, device, train_ds.max_length))
""")

nb["cells"] = cells
with open("main.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote main.ipynb with", len(cells), "cells")
