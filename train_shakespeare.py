"""
train_shakespeare.py

Train a small character-level autoregressive Transformer on the repository's input.txt (Shakespeare).

Usage examples:
  python train_shakespeare.py --input input.txt --epochs 5 --batch_size 64 --seq_len 128 --lr 3e-4

Outputs:
  - model checkpoint (model.pt)
  - vocab mapping (vocab.json)

Requirements: torch >= 1.9
"""

import argparse
import json
import math
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class CharVocab:
    def __init__(self, text=None, vocab=None):
        if vocab is not None:
            self.stoi = vocab
            self.itos = {i: s for s, i in vocab.items()}
        else:
            chars = sorted(list(set(text)))
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, indices):
        return ''.join(self.itos[i] for i in indices)


class CharDataset(Dataset):
    def __init__(self, data, seq_len):
        # data is a list/array of token ints
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        # return contiguous chunk for simplicity
        start = idx
        chunk = self.data[start:start + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def generate_square_subsequent_mask(sz):
    # causal mask for transformer: (sz, sz)
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = ~mask
    mask = mask.float().masked_fill(mask, float('-inf')).masked_fill(~mask, float(0.0))
    return mask


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, seq_len, n_embd=256, n_layer=4, n_head=8, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(seq_len, n_embd)
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head, dim_feedforward=4 * n_embd, dropout=dropout, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.seq_len = seq_len
        self.n_embd = n_embd

        # initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, attn_mask=None):
        # idx: (B, S)
        b, s = idx.size()
        pos = torch.arange(0, s, dtype=torch.long, device=idx.device).unsqueeze(0).expand(b, s)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        # Transformer expects (S, B, E)
        x = x.transpose(0, 1)
        if attn_mask is None:
            attn_mask = generate_square_subsequent_mask(s).to(x.device)
        x = self.transformer(x, mask=attn_mask)
        x = x.transpose(0, 1)  # (B, S, E)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # read data
    with open(args.input, 'r', encoding='utf-8') as f:
        text = f.read()

    vocab = CharVocab(text=text)
    data = vocab.encode(text)

    # create dataset
    dataset = CharDataset(data, seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = TinyGPT(vocab_size=vocab.vocab_size, seq_len=args.seq_len, n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head, dropout=args.dropout)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)  # (B, S, V)
            B, S, V = logits.size()
            loss = criterion(logits.view(B * S, V), y.view(B * S))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % args.log_interval == 0:
                avg = running_loss / args.log_interval
                elapsed = time.time() - start_time
                print(f"Epoch {epoch} step {global_step} | avg_loss {avg:.4f} | lr {args.lr:.2e} | time {elapsed:.1f}s")
                running_loss = 0.0

        # save checkpoint each epoch
        ckpt = {
            'model_state_dict': model.state_dict(),
            'args': vars(args),
            'vocab': vocab.stoi,
        }
        torch.save(ckpt, args.save_path)
        print(f"Saved checkpoint to {args.save_path}")

    # save vocab separately too
    with open(args.vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab.stoi, f, ensure_ascii=False, indent=2)
    print(f"Saved vocab to {args.vocab_path}")


def generate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load vocab
    with open(args.vocab_path, 'r', encoding='utf-8') as f:
        stoi = json.load(f)
    vocab = CharVocab(vocab=stoi)

    # load checkpoint
    ckpt = torch.load(args.save_path, map_location=device)
    model_args = ckpt.get('args', {})
    seq_len = model_args.get('seq_len', args.seq_len)
    model = TinyGPT(vocab_size=vocab.vocab_size, seq_len=seq_len, n_embd=model_args.get('n_embd', 256), n_layer=model_args.get('n_layer', 4), n_head=model_args.get('n_head', 8), dropout=model_args.get('dropout', 0.1))
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    # seed
    context = args.prompt
    idx = torch.tensor([vocab.encode(context[-seq_len:])], dtype=torch.long, device=device)

    generated = context
    with torch.no_grad():
        for _ in range(args.generate):
            if idx.size(1) < seq_len:
                pad_len = seq_len - idx.size(1)
                inp = torch.cat([torch.zeros((1, pad_len), dtype=torch.long, device=device), idx], dim=1)
            else:
                inp = idx[:, -seq_len:]
            logits = model(inp)  # (B, S, V)
            logits = logits[:, -1, :] / args.temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ch = vocab.decode([int(next_id)])
            generated += ch
            idx = torch.cat([idx, next_id], dim=1)
    print(generated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='input.txt', help='path to input.txt')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--n_embd', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_path', type=str, default='model.pt')
    parser.add_argument('--vocab_path', type=str, default='vocab.json')

    # generate subcommand
    subparsers = parser.add_subparsers(dest='command')
    gen_p = subparsers.add_parser('generate')
    gen_p.add_argument('--prompt', type=str, default='To be, or not to be', help='seed text')
    gen_p.add_argument('--generate', type=int, default=200, help='number of chars to generate')
    gen_p.add_argument('--temperature', type=float, default=1.0)

    args = parser.parse_args()

    if args.command == 'generate':
        generate(args)
    else:
        train(args)


if __name__ == '__main__':
    main()
