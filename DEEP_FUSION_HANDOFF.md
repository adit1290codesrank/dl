# Deep Memory-Fusion Text-to-SQL — Handoff

End-to-end enterprise text-to-SQL: business jargon → exact, executable T-SQL.
Built twice — a PyTorch reference (`scripts/`) and a from-scratch C++/CUDA port
(`models/`, `examples/`, `src/`) — sharing one data pipeline and `fusion.bin`.

## The idea (why this works where vanilla seq2seq fails)
Attention KQV *is* a differentiable RAG. Selection of schema/jargon is learned
by similarity (cross-attention into a memory bank); **emission is
deterministic** — schema columns and jargon SQL fragments are atomic IDs that
expand to exact strings at decode, so they can never be misspelled. Literal
values (dates/ids/numbers) are delexicalized to `[valN]` slots and substituted
back deterministically. The model only has to *point* and *compose*, never
spell.

## Architecture (frozen production config: d=512, heads=8, depth=8, dropout=0.15)
- Single causal decoder over `[Question][SEP][SQL]`.
- A global **memory bank** (schema cols/tables + ALT_SYNONYMS + jargon terms,
  ~859 rows) is cross-attended in **every** decoder block (deep fusion), keys =
  masked mean-pool of sub-tokens + a type embedding (table/column/fragment).
- **Dedicated pointer head** (own Q/K, decoupled from block attention) →
  pointer distribution over memory rows, scattered to atomic vocab IDs.
- Pointer-generator head: `P = p_gen·P_vocab + (1-p_gen)·P_mem`, generator
  masked to real BPE ids `[0,V_bpe)`, copy covers `[V_bpe,V)`.
- Greedy decode only (beam search *hurt* — an overfit model assigns higher
  global likelihood to frequent training templates than to the correct query).

## Results (held-out *phrasings*, never seen in training)
| | PyTorch | C++ |
|---|---|---|
| teacher-forced top1 | 97.5% | 94.75% |
| copy_acc (schema linking) | 91.3% | 81.4% |
| greedy exact-match (full query) | 74% overall / 82% business | (run `infer_deep_fusion`) |
| well-formed SQL | 99% | — |

The ~3pt C++ gap is the framework's **per-tensor grad clipping** (each weight
clipped to norm 1.0 independently) vs PyTorch's global-norm clip — it throttles
the tied-embedding's sharpening gradient. Closeable by switching to global-norm
clipping; cosmetic otherwise (the model works).

## WHAT IT CAN DO (trained query families — 45)
Counts/filters per SMU, distinct RSM/SSM/ASM lookups, SONum↔OBD, OBDs invoiced
in a month, top-N customers by volume, largest SMU by volume, business units per
warehouse, RSM-for-OBD joins, vehicle tracking + utilization (period/month/
transporter/LR), transporter dashboard / DOT / POD-upload, OBD availability,
mass-update (set/cancel/reset/mark), pending-status + pending-dispatch reports,
the create→picking / picking/invoicing→dispatch time reports, single-date
thresholds, MPY (incl. MPY-in-month), plus general SELECT/WHERE/TOP-N/GROUP-BY/
DATEDIFF/ISNULL/JOIN over arbitrary columns. Jargon (MPY/ASM/RSM/SSM/OBD/POD/
transporter names…) maps to exact columns/fragments; dates & ids flow through
slots.

## Coverage vs the seed queries
**Value-blind exact-match audit** (every seed's SQL, literals masked, checked
against the actual generated training set — the strongest offline guarantee):
**all 45 seed query types covered.** Families added to close gaps: largest-SMU
by volume, specific-transporter (`@transporter='id'`), transporter+vehicle-type
combo. `ANDashBoardVehicleUtilization` was standardized on the **`'_CB_'` leading
context-arg form** (matches seeds #28–32 and every other proc's convention —
`'_CHATBOT_'`/`'_VIEWING_'`/`'_CB_USERNAME_'`); seeds #10–12 use the older no-arg
form and are intentionally deprecated to that single form (their query *types* —
period-only, period+transporter, period+by-transporter — are all covered in the
`'_CB_'` form). Caveat: the audit checks query shape/params/columns value-blind,
not every phrasing — "type covered" ≠ "every wording correct" (see #2–4).

## WHAT IT CANNOT DO (known limits — tell the user)
0. **Untrained PARAMETER COMBINATIONS within a known procedure.** Subtler than
   missing a procedure: e.g. vehicle-utilization with a *specific* transporter,
   or transporter+vehicle-type together, each needed its own family even though
   the procedure was "covered." Adding a proc isn't enough — each param combo
   users actually ask must be a family. (All seed combos are now in; novel ones
   won't be.)
1. **Query shapes outside the trained families.** No nested subqueries, `HAVING`,
   3+-table joins, `UNION`, window functions, `CASE`, or arbitrary date math
   beyond month/`>=`. It will confidently emit a *plausible but wrong* query for
   these (e.g. early "largest SMU" before its family existed). The model
   generates only shapes it has seen.
2. **Novel jargon / synonyms not in `jargon_fusion.json` or `ALT_SYNONYMS`.**
   Selection generalizes across *phrasings*, but a genuinely new business term
   or column nickname won't map.
3. **Near-twin column disambiguation** under unusual phrasing (e.g. Invoicing-
   vs Picking-to-Dispatch, ActDeliveryDate vs InvoiceDate). C++ copy_acc ~81%
   means roughly 1 in 5 hard copy decisions can pick the wrong twin.
4. **Garbled / pure-letter ids** ("customer code hz g an x") and typo'd month
   names — unslottable noise; dropped from training, unreliable at inference.
5. **Values it must compute, not copy** — handled only where a slot rule exists
   (month→start/end, "after DATE"). Anything needing real arithmetic/reasoning
   over values is out of scope.
6. It is a **4.9M–35M param model trained from scratch on synthetic data** — it
   has no world knowledge or language understanding beyond the ~317 templates.
   Robustness comes from data diversity, not pretraining.

## How to extend (this is the data flywheel)
Add a family to `scripts/generate_paraphrase_dataset.py` (5–8 phrasings + the
SQL template; last 2 phrasings are auto held-out for honest eval), then retrain.
Real production questions should be fed back as new families/phrasings over
time. **Already queued for next retrain:** the "largest SMU by volume" family
and (optional) global-norm grad clip.

## Data pipeline & files
- `scripts/generate_paraphrase_dataset.py` → `data/synthetic_dataset.json`
  (train block then held-out val block; sequential 80/20 split lands on the
  boundary — do NOT pass `--dedup` for the real run).
- `scripts/train_bpe.py` → `data/bpe_tokenizer.json` + `bpe_vocab.txt` +
  `bpe_merges.txt`. **NB:** merges must be written `"a b"` not list-repr —
  HF≥0.20 stores pairs; getting this wrong silently breaks the C++ tokenizer
  into char-level (was the inference "garbage SQL" bug).
- `scripts/prepare_fusion.py` → `data/fusion.bin` + `fusion_expansions.txt`
  (atomic id → exact string) + `fusion_memory.txt`. Delexicalizes literals
  (`scripts/value_slots.py`), builds the memory bank, repairs/drops noisy labels.

### `fusion.bin` layout
`9×int32 header(n_train,n_val,seq_len,V,V_bpe,M,max_mem_toks,S,J)` then
`mem_emit_ids[M]`, `mem_tokens[M*max_mem_toks]`, `mem_types[M]`,
`X_train`, `Y_train`, `X_val`, `Y_val` (all float32; Y uses -100 = ignore).
Memory is stored ONCE (global), not per example.

## Build & run (Linux/cloud; T4 = sm_75, the Makefile default)
```bash
# data (run once; re-run only when families/jargon change)
python scripts/generate_paraphrase_dataset.py
python scripts/train_bpe.py
python scripts/prepare_fusion.py

# C++ train (args: dim depth epochs batch; converges ~epoch 20)
make MAIN_SRC=examples/train_deep_fusion.cpp -j$(nproc)
./train_deep_fusion 512 8 30 64          # OOMs at bs128 on a 16GB T4 — use 64

# C++ inference (REPL)
make MAIN_SRC=examples/infer_deep_fusion.cpp -j$(nproc)
./infer_deep_fusion

# PyTorch reference
python scripts/schema_fusion_pt.py --dim 512 --depth 8 --epochs 30
python scripts/schema_fusion_pt.py --eval        # greedy exact-match + per-jargon
python scripts/schema_fusion_pt.py --ask         # probe questions
```
**Gotchas:** editing a header (`*.h`) does NOT trigger recompile (Makefile has
no header dep tracking) — use `make clean` after header changes. Per-epoch
timing + `copy_acc` are logged to `loss_log_fusion_cpp.csv`.

## Performance notes (C++)
~511s/epoch for 512/8 on a T4 (~4.3h/30ep). It's **compute-bound**, not buggy —
this hand-rolled fp32 framework runs ~5× slower/step than PyTorch. cuBLAS
matmuls, the tensor pool, and adam are all fine. Grad-clip was made fully
on-device (was doing cudaMalloc+sync+D2H+free per call, ~170×/batch).

## If you rebuild the framework (how to actually BEAT PyTorch)
PyTorch is Python wrappers over cuBLAS/cuDNN — a clean C++/CUDA impl making the
same library calls with less overhead *should* be faster, not 5× slower. The
current code is slow because it's naively built, not because it's C++. The
concrete debts a rebuild must fix (roughly by impact):

1. **Tensor cores (biggest).** Everything runs fp32. A T4/A100/H100 has fp16/
   bf16 tensor cores giving 8–16× matmul throughput. Use `cublasGemmEx`/
   cuBLASLt in fp16 (or bf16) with fp32 accumulation + loss scaling. This alone
   likely flips the 5×-slower into faster-than-PyTorch.
2. **Operator fusion.** Every op is a separate kernel launch (LayerNorm, scale,
   mask, softmax, residual all separate). Fuse: LN+residual, the whole attention
   score→scale→mask→softmax→·V into a flash-attention-style kernel. PyTorch fuses
   via cuDNN/compile; the hand framework launches hundreds of tiny kernels/step.
3. **CUDA graphs.** The per-step kernel-launch sequence is static — capture it
   once as a CUDA graph to erase per-launch CPU overhead (huge when you have
   thousands of small launches/step).
4. **The 64× redundant memory K/V projection** (~43% of forward). The memory
   bank is identical across the batch but re-projected per batch-element in every
   block. Project K/V once on `[M,d]`, broadcast — needs `PointerAttention`
   forward+backward rewrite (+ re-parity check).
5. **One-thread-per-row elementwise/softmax kernels** → low occupancy. Use
   warp/block reductions, vectorized loads.
6. **Per-layer clip→Adam is atomic.** Restructure to a two-phase backward
   (accumulate all grads → one global-norm clip → step). Fixes BOTH speed
   (batch the Adam/clip kernels) AND the ~3pt quality gap (global-norm clipping
   = PyTorch parity; per-tensor norm-1.0 currently throttles the tied embedding).
7. **Real arena allocator + async.** The pool keys on exact byte size; the
   memory broadcast does 64 synchronous `cudaMemcpy`s; `calculate_accuracy`
   still mallocs per call. Move everything to one stream, async copies, no
   per-call malloc.

Target: with 1–3 alone, this model should train in well under PyTorch's ~90 min
on the same T4. Keep the architecture and `fusion.bin` format; rebuild the
kernel/exec layer underneath.
