"""
schema_fusion_pt.py -- Deep Memory-Fusion decoder (PyTorch prototype).

Architecture (replaces the late-fusion dual-encoder SchemaRAGNet):
  - ONE causal decoder over [Question][SEP][SQL][EOS].
  - A global memory bank (schema elements + synonyms + jargon terms, built by
    prepare_fusion.py) is cross-attended in EVERY decoder block:
        pre-LN: self-attn -> cross-attn(memory) -> FFN
    This is "attention as differentiable RAG" applied at depth, instead of one
    retrieval at the last layer.
  - Pointer-generator head over a COMPACT vocab: real BPE tokens [0, V_bpe)
    via the generator, atomic schema/jargon IDs [V_bpe, V) via the pointer.
    Atomic IDs expand deterministically at detokenization -- never spelled.

Blocks are hand-rolled (not nn.TransformerDecoderLayer) to mirror the C++
layer inventory (SelfAttention + PointerAttention + Dense FFN, pre-LN) so the
Phase-B port is mechanical.

Usage:
  python scripts/schema_fusion_pt.py                   # train
  python scripts/schema_fusion_pt.py --epochs 300 --bs 8   # mini overfit
  python scripts/schema_fusion_pt.py --eval            # greedy-decode eval
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import struct
import array
import math
import os
import re
import sys
import time
import argparse

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    """Pre-LN: self-attn -> cross-attn(memory) -> FFN. Mirrors the planned C++
    DecoderBlock (SelfAttention + PointerAttention + 2x Dense)."""

    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.cross_attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln3 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, 4 * d)
        self.ff2 = nn.Linear(4 * d, d)

    def forward(self, x, mem, causal_mask):
        h = self.ln1(x)
        a, _ = self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        x = x + a
        h = self.ln2(x)
        a, w = self.cross_attn(h, mem, mem, average_attn_weights=True)
        x = x + a
        h = self.ln3(x)
        x = x + self.ff2(F.relu(self.ff1(h)))
        return x, w  # w: [B, T, M] cross-attention weights


class DeepFusionNet(nn.Module):
    def __init__(self, V, V_bpe, max_len, d=256, heads=8, depth=4):
        super().__init__()
        self.V = V
        self.V_bpe = V_bpe
        # ONE embedding: input tokens, memory keys, and tied output projection.
        self.tok_emb = nn.Embedding(V, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(DecoderBlock(d, heads) for _ in range(depth))
        self.final_ln = nn.LayerNorm(d)
        self.p_gen_proj = nn.Linear(d, 1)
        # Bias > 0 keeps p_gen high early so the flat pointer distribution
        # doesn't drown the generator. With the compact vocab P_vocab is far
        # less diluted than at V=50k -- monitor mean p_gen at copy positions
        # and lower this to 1.0 if it stays pinned > 0.9 after warmup.
        nn.init.constant_(self.p_gen_proj.bias, 2.5)
        # GPT-style small init: the default N(0,1) embedding init makes the
        # tied-softmax logits explode (std ~ sqrt(d)) and saturates training.
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def encode_memory(self, mem_tokens):
        # mem_tokens: [M, max_mem_toks] -> masked mean-pool -> [M, d].
        # No self-attention over memory rows: keeps each row's identity crisp.
        emb = self.tok_emb(mem_tokens)
        mask = (mem_tokens != 0).float().unsqueeze(-1)
        return (emb * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def forward(self, X, mem_tokens, mem_emit_ids):
        B, T = X.shape
        M = mem_tokens.shape[0]
        mem = self.encode_memory(mem_tokens).unsqueeze(0).expand(B, -1, -1)
        pos = torch.arange(T, device=X.device)
        x = self.tok_emb(X) + self.pos_emb(pos).unsqueeze(0)
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=X.device)
        for blk in self.blocks:
            x, w = blk(x, mem, causal)  # keep last block's w as pointer dist

        ln_out = self.final_ln(x)

        # Generator branch (tied embeddings), computed in fp32 for stability.
        vocab_logits = F.linear(ln_out, self.tok_emb.weight).float()
        vocab_logits[:, :, self.V_bpe:] = -1e9  # atomic IDs only via pointer
        P_vocab = F.softmax(vocab_logits, dim=-1)

        p_gen = torch.sigmoid(self.p_gen_proj(ln_out)).float()  # [B, T, 1]

        # Pointer branch: scatter last block's cross-attn weights into vocab
        # space. Multiple memory rows (synonyms/jargon) may share an emit ID;
        # scatter_add accumulates them.
        w = w.float()
        P_mem = torch.zeros(B, T, self.V, device=X.device)
        idx = mem_emit_ids.view(1, 1, M).expand(B, T, M)
        P_mem.scatter_add_(2, idx, w)

        # +eps instead of clamp: clamp has zero gradient below the floor, which
        # silently kills the learning signal for any target the model currently
        # assigns ~0 probability.
        P_final = p_gen * P_vocab + (1.0 - p_gen) * P_mem
        return torch.log(P_final + 1e-9), p_gen.squeeze(-1)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_dataset(bin_path):
    with open(bin_path, "rb") as f:
        (n_train, n_val, seq_len, V, V_bpe, M, max_mem_toks, S, J) = struct.unpack(
            "9i", f.read(36))

        mem_emit_ids = array.array('f')
        mem_emit_ids.fromfile(f, M)
        mem_emit_ids = torch.tensor(mem_emit_ids, dtype=torch.long)

        mem_tokens = array.array('f')
        mem_tokens.fromfile(f, M * max_mem_toks)
        mem_tokens = torch.tensor(mem_tokens, dtype=torch.long).view(M, max_mem_toks)

        def read_set(n):
            X = array.array('f')
            X.fromfile(f, n * seq_len)
            Y = array.array('f')
            Y.fromfile(f, n * seq_len)
            return (torch.tensor(X, dtype=torch.long).view(n, seq_len),
                    torch.tensor(Y, dtype=torch.long).view(n, seq_len))

        train_X, train_Y = read_set(n_train)
        val_X, val_Y = read_set(n_val)

    assert int(mem_emit_ids.min()) >= V_bpe and int(mem_emit_ids.max()) < V
    for Y in (train_Y, val_Y):
        valid = Y[Y != -100]
        assert int(valid.min()) >= 0 and int(valid.max()) < V, "target id out of range"

    return {
        "n_train": n_train, "n_val": n_val, "seq_len": seq_len,
        "V": V, "V_bpe": V_bpe, "M": M, "max_mem_toks": max_mem_toks,
        "S": S, "J": J,
        "mem_emit_ids": mem_emit_ids, "mem_tokens": mem_tokens,
        "train": (train_X, train_Y), "val": (val_X, val_Y),
    }


def load_expansions():
    exp = {}
    with open(os.path.join(BASE, "data", "fusion_expansions.txt"), encoding="utf-8") as f:
        for line in f:
            vid, _, text = line.rstrip("\n").partition("|")
            exp[int(vid)] = text
    return exp


# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = load_dataset(os.path.join(BASE, "data", "fusion.bin"))
    print(f"V={data['V']} V_bpe={data['V_bpe']} M={data['M']} "
          f"train={data['n_train']} val={data['n_val']}")

    train_ds = torch.utils.data.TensorDataset(*data["train"])
    val_ds = torch.utils.data.TensorDataset(*data["val"])
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.bs)

    mem_tokens = data["mem_tokens"].to(device)
    mem_emit_ids = data["mem_emit_ids"].to(device)

    model = DeepFusionNet(data["V"], data["V_bpe"], data["seq_len"],
                          d=256, heads=8, depth=4).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-6)
    criterion = nn.NLLLoss(ignore_index=-100)
    # fp32 only. The pointer-generator head computes log(P_final + 1e-9); its
    # gradient is 1/(P+eps) which can spike to ~1e9 on a confident-but-wrong
    # token. fp16 (autocast) overflows that to inf -> nan and poisons the
    # weights (this is what killed the run at epoch 43). At 4.9M params fp16
    # saves nothing on a T4, so we drop it entirely.
    use_amp = False

    log_path = os.path.join(BASE, "loss_log_fusion.csv")
    # Append mode: resuming a run continues the same log.
    if not os.path.exists(log_path):
        with open(log_path, "w") as lf:
            lf.write("epoch,train_loss,val_loss,top1,top5,copy_acc,pgen_copy,lr\n")

    best_top5 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        if epoch <= args.warmup:
            lr = 1e-6 + (args.max_lr - 1e-6) * (epoch / args.warmup)
        else:
            # Cosine annealing from max_lr down to lr_min (1% of peak) over the
            # post-warmup epochs. Tames the late-stage oscillation / overfit
            # tail seen when LR was held flat at the peak.
            lr_min = args.max_lr * 0.01
            progress = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
            lr = lr_min + 0.5 * (args.max_lr - lr_min) * (1 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g['lr'] = lr

        total_loss = 0.0
        n_steps = 0
        t0 = time.time()
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                log_probs, _ = model(X, mem_tokens, mem_emit_ids)
            loss = criterion(log_probs.view(-1, data["V"]), Y.view(-1))
            # Insurance for an unattended run: never let a non-finite loss
            # poison the weights -- drop that step and carry on.
            if not torch.isfinite(loss):
                print(f"  [warn] non-finite loss at epoch {epoch}, step {n_steps} -- skipped")
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_steps += 1
        train_loss = total_loss / max(1, n_steps)

        # ---- validation ----
        model.eval()
        val_loss = c1 = c5 = total = 0
        copy_c1 = copy_total = 0
        pgen_copy_sum = pgen_copy_n = 0.0
        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    log_probs, p_gen = model(X, mem_tokens, mem_emit_ids)
                lp = log_probs.view(-1, data["V"])
                Yf = Y.view(-1)
                val_loss += criterion(lp, Yf).item()

                valid = Yf != -100
                tgt = Yf[valid]
                if len(tgt):
                    vlp = lp[valid]
                    _, top5 = vlp.topk(5, dim=-1)
                    c1 += (top5[:, 0] == tgt).sum().item()
                    c5 += (top5 == tgt.unsqueeze(1)).sum().item()
                    total += len(tgt)
                    # Diagnostics on copy positions (atomic-ID targets):
                    is_copy = tgt >= data["V_bpe"]
                    if is_copy.any():
                        copy_c1 += (top5[:, 0][is_copy] == tgt[is_copy]).sum().item()
                        copy_total += is_copy.sum().item()
                        pgen_copy_sum += p_gen.view(-1)[valid][is_copy].sum().item()
                        pgen_copy_n += is_copy.sum().item()

        val_loss /= len(val_loader)
        top1 = 100 * c1 / max(1, total)
        top5a = 100 * c5 / max(1, total)
        copy_acc = 100 * copy_c1 / max(1, copy_total)
        pgen_copy = pgen_copy_sum / max(1, pgen_copy_n)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Top1: {top1:.2f}% | Top5: {top5a:.2f}% | "
              f"CopyAcc: {copy_acc:.2f}% | pgen@copy: {pgen_copy:.3f} | "
              f"LR: {lr:.2e} | Time: {elapsed:.2f}s")

        with open(log_path, "a") as lf:
            lf.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},"
                     f"{top1:.4f},{top5a:.4f},{copy_acc:.4f},{pgen_copy:.6f},{lr:.2e}\n")

        if top5a > best_top5:
            best_top5 = top5a
            os.makedirs(os.path.join(BASE, "weights"), exist_ok=True)
            path = os.path.join(BASE, "weights", "schema_fusion_pt.pt")
            torch.save(model.state_dict(), path)
            print(f"    [checkpoint] new best Top5 {top5a:.2f}% -> {path}")


# --------------------------------------------------------------------------
# Greedy-decode eval
# --------------------------------------------------------------------------
def well_formed(sql):
    s = sql.strip().lower()
    if not (s.startswith("select") or s.startswith("with") or s.startswith("exec")):
        return False
    if s.count("'") % 2 != 0:
        return False
    if s.count("(") != s.count(")"):
        return False
    return True


def evaluate(args):
    from tokenizers import Tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(os.path.join(BASE, "data", "fusion.bin"))
    tokenizer = Tokenizer.from_file(os.path.join(BASE, "data", "bpe_tokenizer.json"))
    expansions = load_expansions()
    eos_id = tokenizer.token_to_id("[eos]")
    V_bpe = data["V_bpe"]

    def detok(ids):
        parts, buf = [], []
        for i in ids:
            if i >= V_bpe:
                if buf:
                    # keep_special so [valN] slot tokens stay visible
                    parts.append(tokenizer.decode(buf, skip_special_tokens=False))
                    buf = []
                parts.append(expansions[i])
            else:
                buf.append(i)
        if buf:
            parts.append(tokenizer.decode(buf, skip_special_tokens=False))
        return " ".join(parts)

    def squash(s):
        return re.sub(r"\s+", "", s).lower()

    model = DeepFusionNet(data["V"], data["V_bpe"], data["seq_len"],
                          d=256, heads=8, depth=4).to(device)
    path = os.path.join(BASE, "weights", "schema_fusion_pt.pt")
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    mem_tokens = data["mem_tokens"].to(device)
    mem_emit_ids = data["mem_emit_ids"].to(device)
    val_X, val_Y = data["val"]
    seq_len = data["seq_len"]

    # Per-jargon breakdown driven by jargon_fusion.json so it covers every
    # term we actually trained on (procedures, time expressions, synonyms),
    # not just the original 5. Each entry is labelled by its first key and
    # matched against the prompt by ANY of its key phrases.
    import json
    jpath = os.path.join(BASE, "jargon_fusion.json")
    if os.path.exists(jpath):
        with open(jpath, encoding="utf-8") as jf:
            jentries = json.load(jf)
        jargon_specs = [(e["keys"][0],
                         [re.compile(r'(?<!\w)' + re.escape(k.lower()) + r'(?!\w)')
                          for k in e["keys"]])
                        for e in jentries]
    else:
        jargon_specs = [(t, [re.compile(r'\b' + t.lower() + r'\b')])
                        for t in ["MPY", "ASM", "RSM", "SSM", "NSM"]]
    stats = {label: [0, 0, 0] for label, _ in jargon_specs}  # label -> [exact, value_blind, total]
    exact = exact_vb = wf = tok_exact = n = 0
    shown = 0

    def value_blind(s):
        # Mask quoted literals and bare numbers, then squash. Isolates
        # "got the SQL structure + schema right" from "transcribed the
        # literal dates/ids right" -- two very different failure modes.
        s = re.sub(r"'[^']*'", "'<V>'", s)
        s = re.sub(r"\d+", "<N>", s)
        return squash(s)

    with torch.no_grad():
        for i in range(len(val_X)):
            Y = val_Y[i]
            sup = (Y != -100).nonzero().view(-1)
            if len(sup) == 0:
                continue
            p = int(sup[0])  # first supervised position; X[:p+1] = prompt + [sep]
            tgt_ids = [int(t) for t in Y[sup] if int(t) != eos_id]

            seq = val_X[i][:p + 1].tolist()
            gen = []
            for _ in range(seq_len - len(seq)):
                X = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
                log_probs, _ = model(X, mem_tokens, mem_emit_ids)
                nxt = int(log_probs[0, -1].argmax())
                if nxt == eos_id:
                    break
                gen.append(nxt)
                seq.append(nxt)

            pred_sql = detok(gen)
            gold_sql = detok(tgt_ids)
            prompt_txt = tokenizer.decode(val_X[i][:p].tolist(),
                                          skip_special_tokens=False)

            ok = squash(pred_sql) == squash(gold_sql)
            ok_vb = value_blind(pred_sql) == value_blind(gold_sql)
            exact += ok
            exact_vb += ok_vb
            tok_exact += gen == tgt_ids
            wf += well_formed(pred_sql)
            n += 1
            plo = prompt_txt.lower()
            for label, pats in jargon_specs:
                if any(p.search(plo) for p in pats):
                    stats[label][2] += 1
                    stats[label][0] += ok
                    stats[label][1] += ok_vb

            if args.show_jargon:
                want = any(any(p.search(plo) for p in pats)
                           for label, pats in jargon_specs
                           if label.lower() == args.show_jargon.lower())
                if want and not ok and shown < args.show:
                    shown += 1
                    print(f"--- example {i} [{args.show_jargon}] ---")
                    print("Q   :", prompt_txt[:140])
                    print("GOLD:", gold_sql[:200])
                    print("PRED:", pred_sql[:200])
            elif shown < args.show and not ok:
                shown += 1
                print(f"--- example {i} ---")
                print("Q   :", prompt_txt[:120])
                print("GOLD:", gold_sql[:160])
                print("PRED:", pred_sql[:160])

    print(f"\nVal examples: {n}")
    print(f"Exact match (whitespace-normalized): {100 * exact / max(1, n):.2f}%")
    print(f"Exact match VALUE-BLIND (literals masked): {100 * exact_vb / max(1, n):.2f}%")
    print(f"Token-sequence exact match:          {100 * tok_exact / max(1, n):.2f}%")
    print(f"Well-formed SQL:                     {100 * wf / max(1, n):.2f}%")
    print("Per-jargon exact / value-blind (val prompts containing the term):")
    for label, (e, evb, tot) in sorted(stats.items(), key=lambda kv: -kv[1][2]):
        if tot:
            print(f"  {label:32s}: {e}/{tot} ({100 * e / tot:.1f}%)  "
                  f"| value-blind {evb}/{tot} ({100 * evb / tot:.1f}%)")


# Probe questions for --ask. Three tiers, in increasing difficulty:
#   A) in-distribution phrasings (should be solid),
#   B) PARAPHRASED jargon in unseen sentences (the real test of whether
#      similarity-based retrieval generalizes beyond exact string match),
#   C) synonyms / multi-jargon (hardest -- multiple copies + a join).
PROBE_QUESTIONS = [
    # --- A: in-distribution ---
    "What is the number of sales for MPY",
    "Give all the OBDs invoiced in the month of april 2025",
    "Who is the ASM for customer code ABC123",
    "Vehicle Utilization for the period between 2025-01-01 to 2025-01-31",
    # --- B: paraphrased jargon (generalization) ---
    "How many sales were made for MPY in April 2025?",
    "List all OBDs dispatched after 2025-05-01",
    "Which salesperson is the ASM for customer code XYZ789?",
    "Show me the vehicle utilization for March 2025",
    # --- C: synonyms / multi-jargon ---
    "What is the Sales Order Number for OBD ABC123",
    "Who is the RSM for OBD Number ABC123",
    "Count the OBDs for the Marine business unit",
]


def ask(args):
    """Interactive inference: English question -> generated SQL. Either runs
    the built-in PROBE_QUESTIONS or a single --q "..." question. Literal
    values are delexicalized to [valN] slots before encoding and substituted
    back into the generated SQL (see value_slots.py)."""
    from tokenizers import Tokenizer
    from value_slots import extract_slots, relex
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_dataset(os.path.join(BASE, "data", "fusion.bin"))
    tokenizer = Tokenizer.from_file(os.path.join(BASE, "data", "bpe_tokenizer.json"))
    expansions = load_expansions()
    eos_id = tokenizer.token_to_id("[eos]")
    sep_id = tokenizer.token_to_id("[sep]")
    V_bpe = data["V_bpe"]
    seq_len = data["seq_len"]

    def detok(ids):
        parts, buf = [], []
        for i in ids:
            if i >= V_bpe:
                if buf:
                    parts.append(tokenizer.decode(buf, skip_special_tokens=False))
                    buf = []
                parts.append(expansions[i])
            else:
                buf.append(i)
        if buf:
            parts.append(tokenizer.decode(buf, skip_special_tokens=False))
        return " ".join(parts)

    model = DeepFusionNet(data["V"], data["V_bpe"], data["seq_len"],
                          d=256, heads=8, depth=4).to(device)
    model.load_state_dict(torch.load(
        os.path.join(BASE, "weights", "schema_fusion_pt.pt"), map_location=device))
    model.eval()
    mem_tokens = data["mem_tokens"].to(device)
    mem_emit_ids = data["mem_emit_ids"].to(device)

    questions = [args.q] if args.q else PROBE_QUESTIONS
    with torch.no_grad():
        for q in questions:
            delexed, slot_values = extract_slots(q)
            seq = tokenizer.encode(delexed).ids + [sep_id]
            gen = []
            for _ in range(seq_len - len(seq)):
                X = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)
                log_probs, _ = model(X, mem_tokens, mem_emit_ids)
                nxt = int(log_probs[0, -1].argmax())
                if nxt == eos_id:
                    break
                gen.append(nxt)
                seq.append(nxt)
            print(f"Q   : {q}")
            if slot_values:
                print(f"slots: {dict((f'[val{i+1}]', v) for i, v in enumerate(slot_values))}")
            print(f"SQL : {relex(detok(gen), slot_values)}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--ask", action="store_true", help="run built-in probe questions")
    ap.add_argument("--q", type=str, default=None, help="ask a single question")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max-lr", type=float, default=2.5e-4)
    ap.add_argument("--show", type=int, default=5, help="failed examples to print in --eval")
    ap.add_argument("--show-jargon", type=str, default=None,
                    help="in --eval, only print failures whose prompt matches this jargon label")
    args = ap.parse_args()
    if args.eval:
        evaluate(args)
    elif args.ask or args.q:
        ask(args)
    else:
        train(args)
