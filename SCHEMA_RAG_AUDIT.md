# SchemaRAGNet — Full Audit & Fix Plan

Instructions for the code editor (Gemini): each item below is an independent change. Apply them in
the order given (Tier 1 first — highest impact on accuracy). Each item lists the **file**, the
**problem**, **why it matters**, and the **exact fix**. Do not skip the "why" — several of these look
optional but are the reason the model plateaus at ~30% top-1 / ~56% top-5.

Context: vocab_size = 2000, seq_len = 256, dim = 256, heads = 8, depth = 4, batch = 256,
~12k train examples. Task = text→T-SQL with a schema-linking cross-attention ("pointer") layer.
Symptom: smooth descent to val_loss ≈ 1.19 then a hard plateau, plus periodic loss spikes
(epochs 31/41/52) — one of which (52) permanently knocked the run into a worse basin.

Diagnosis summary: the architecture is **not** broken (smooth descent proves it learns). The model
is **underfitting** (train_loss ≈ val_loss, val even slightly lower), and the training signal is
heavily **diluted and mis-targeted**. Fix the signal and the optimizer hygiene first; scale capacity
and data last.

---

## TIER 1 — Training signal (do these first; biggest accuracy wins, low risk)

### 1. No loss masking — the model is trained on padding and on the English prompt
- **Files:** `models/transformers/schema_rag_net.h` (fit loop + `backward`), `src/core/cuda_ops.cu`
  (`cross_entropy_kernel`, `cross_entropy_loss_kernel`), `scripts/prepare_breakwalls.py`.
- **Problem:** In `fit()` the one-hot target `bY` is built for **all** `seq_len` positions, including
  the `[PAD]` run and the English-prompt tokens (`prepare_breakwalls.py:95` builds
  `combined = inp_ids + [sep] + out_ids`, and every position is teacher-forced). `cross_entropy_kernel`
  / `cross_entropy_loss_kernel` divide by `n = bs*seq_len` (all tokens). With a ~40–80 token real
  sequence padded to 256, **roughly 70–80% of every gradient and of the reported loss is PAD→PAD**
  (trivial) plus prompt-token prediction (noise).
- **Why it matters:** This is the single biggest reason for the plateau. val_loss ≈ 1.19 is *diluted*
  by easy padding, so it looks low while real per-SQL-token accuracy stays at 30%. The model spends
  most of its capacity learning "PAD follows PAD" and trying to predict the user's free-text question
  (unlearnable). Reported top-k already skips PAD, which is why loss and accuracy disagree.
- **Fix:**
  1. Introduce an **ignore label** convention. Stop materializing dense one-hot targets (see item #9).
     Pass integer target IDs to the GPU; use `-100` (or `pad_id`) to mark "ignore".
  2. In the data prep, mark **every prompt position and every PAD position as ignore** in the target,
     so the loss is computed **only on the SQL tokens after `[SEP]`**. Concretely, when building
     `y_seq`, set the label to the ignore value for indices `< sep_position` and for padding.
  3. In `cross_entropy_kernel` / `cross_entropy_loss_kernel`, skip positions whose target == ignore
     (write zero gradient, add zero loss) and divide by the **count of non-ignored tokens**, not
     `bs*seq_len`. Compute that count on the GPU (atomic counter) or pass it in from the host.
- **Acceptance:** reported val_loss will jump *up* (it stops being diluted) — that is correct and
  expected. Top-1/Top-5 should rise materially within a few epochs.

### 2. Schema is encoded as a single token per element (schema linking is blind)
- **File:** `scripts/prepare_breakwalls.py:92`.
- **Problem:** `schema_ids = [toks[0] if toks else unk_id for toks in schema_tokens_list]` — each
  schema element (e.g. table+column+synonyms) is reduced to **only its first BPE subword**. The
  schema encoder therefore sees one token per column and cannot represent `customer_email` vs
  `customer_id`.
- **Why it matters:** For a schema-linking task this throws away almost all of the discriminative
  signal the cross-attention layer needs. The "RAG" half of the model is running on near-zero
  information.
- **Fix:** Encode the **full token sequence** of each schema element. Change the schema tensor from
  `[n, schema_size]` (one id per element) to `[n, schema_size, max_schema_tok_len]` (or flatten to
  `[n, schema_size * max_schema_tok_len]` with a fixed per-element length and PAD). Update the schema
  encoder forward to consume the real sub-sequences (pool per element, e.g. mean over each element's
  tokens, to get one vector per schema slot for the cross-attention keys). If full sequences are too
  invasive right now, at minimum **mean-pool the embeddings of all sub-tokens** of each element
  instead of taking `toks[0]`.

### 3. The pointer / copy mechanism is implemented but never used in training
- **Files:** `models/transformers/schema_rag_net.h` (`forward`), `src/layers/pointer_attention.cu`
  (`pointer_scatter_add_cuda` exists), `include/layers/pointer_attention.h`.
- **Problem:** `forward()` computes `pointer_layer->forward_dual(...)`, takes only `out.first`
  (the context vector), does `context + Q_emb → vocab_proj → softmax`, and **discards the attention
  weights**. `pointer_scatter_add_cuda` (the copy distribution `p_final = p_gen*P_vocab +
  (1-p_gen)*P_schema`) is dead code.
- **Why it matters:** In SQL, table/column names must be **copied** from the schema. A copy head is
  exactly the mechanism that makes those tokens near-deterministic and spikes top-k. Leaving it off
  caps achievable accuracy on schema-specific tokens.
- **Fix:** Wire the pointer-generator into the training forward and loss:
  1. Add a learned scalar gate `p_gen ∈ (0,1)` per output position (a `Dense(dim,1)` + sigmoid on the
     cross-attention context).
  2. Final distribution = `p_gen * softmax(vocab_logits) + (1 - p_gen) * scatter(attn_weights →
     schema token vocab ids)`, using `pointer_scatter_add_cuda` (extend it to batched, not just
     batch=1 as the current comment notes).
  3. Compute CE on this mixed distribution. Backprop through both branches and through `p_gen`.
  - This depends on #2 (real schema tokens) to be useful.

### 4. Identical RNG seed across every layer → broken initialization symmetry
- **Files:** `src/layers/self_attention.cpp:25`, `src/layers/dense.cpp:19`,
  `src/layers/embedding.cpp:13`, `src/layers/pointer_attention.cu:158`.
- **Problem:** Every layer constructs `std::mt19937 gen(42)` with the **same fixed seed**. Within one
  layer the draws differ, but **across layers they are identical**: every Transformer block's
  attention starts with the exact same wQ/wK/wV/wO, and the query encoder and schema encoder start
  with identical analogous weights.
- **Why it matters:** Identical blocks reduce effective capacity early and slow convergence (the
  depth-4 stack starts as 4 copies of the same function). It's not fatal because gradients differ,
  but it wastes the first chunk of training.
- **Fix:** Seed each layer uniquely. Pass a `seed` (or a global atomic counter) into each
  constructor, or derive from a static incrementing counter, e.g. `static std::atomic<int> ctr;
  std::mt19937 gen(42 + ctr++);`. Keep it deterministic for reproducibility.

---

## TIER 2 — Optimizer stability (prevents the spikes / the epoch-52 collapse)

### 5. No true global-norm gradient clipping; per-element ±5 clamp distorts direction
- **Files:** `src/core/cuda_ops.cu` (`adam_kernel:191-194`), all trainable layers,
  `models/transformers/schema_rag_net.h`.
- **Problem:** The only clipping is a per-element clamp to ±5 inside `adam_kernel`, applied per layer
  as gradients are produced. Per-element clamping bounds each component but **not the total step
  norm**, and when many components saturate at once (a spike) the step is still huge. There is no
  coordination across layers.
- **Why it matters:** This is the cause of the periodic spikes and the epoch-52 permanent damage.
- **Fix (true global-norm clipping — requires decoupling backward from the Adam update):**
  1. `Layer` interface: split into `backward(grad)` (compute & **store** grads as members) and
     `step(lr, scale)` (apply Adam with grads pre-scaled by `scale`), plus
     `float grad_norm_sq()` (sum of squares of this layer's grads via a CUDA reduction).
  2. Promote all local grad tensors (`dW`, `db`, …) to persistent members where they aren't already
     (PointerAttention/SelfAttention already store `dwQ…`; Dense/LayerNorm/Embedding compute locally
     — promote those).
  3. Move each layer's `t++` from `backward` into `step`, or bias-correction desyncs.
  4. In `SchemaRAGNet::step(lr)`: accumulate `global_norm_sq` over **every** trainable param
     (be exhaustive — all blocks, both encoders, pointer, vocab_proj), `global_norm = sqrt(...)`,
     then:
     ```cpp
     float scale;
     if (!isfinite(global_norm))      scale = 0.0f;          // skip poisoned step (NaN/Inf guard)
     else if (global_norm > max_norm) scale = max_norm/global_norm;
     else                             scale = 1.0f;
     ```
     with `max_norm = 1.0f`. Then call `step(lr, scale)` on each sub-module. The `scale = 0` branch
     is what turns "epoch 52 destroyed the run" into "epoch 52 skipped one update."
  - **Lighter alternative if the refactor is too risky:** per-tensor norm clipping — each layer scales
    its own grad to `max_norm` right before its existing `adam_cuda` call. No interface change, ~90%
    of the benefit, but not globally coordinated. Use this if you want the spikes gone without the
    15-file refactor.
- **Keep** the ±5 per-element clamp as a cheap backstop; with norm clipping it will rarely trigger.

---

## TIER 3 — Performance (time) & memory

### 6. Dense one-hot targets built on CPU and uploaded every batch
- **File:** `models/transformers/schema_rag_net.h` (fit loop, both train & val), `src/core/loss.cpp`,
  `src/core/cuda_ops.cu` (CE kernels).
- **Problem:** Each batch allocates and zeroes `bY` of size `bs * seq_len * vocab = 256*256*2000 ≈
  131M floats ≈ 524 MB` on the **host**, fills a handful of 1.0s, and uploads all 524 MB to the GPU.
  `grad` and `pred` are the same shape and resident on device. So ~1.5 GB of vocab-sized tensors plus
  a 524 MB host→device copy **per batch**.
- **Why it matters:** This is a major time sink (host alloc/zero + PCIe copy every step) and a large
  chunk of VRAM, all to represent a one-hot that is 99.95% zeros.
- **Fix:** Pass **integer target IDs** `[bs*seq_len]` to the GPU (a few hundred KB). Rewrite the CE
  forward/backward kernels to index the target class directly:
  `dy[i*vocab + j] = (pred - (j==target_id ? 1 : 0)) / n_valid` with the ignore handling from #1.
  This deletes the one-hot entirely (host vector, the fill loop, and the 524 MB copy). Combine with #1.

### 7. CPU top-1/top-5 accuracy loop in validation
- **File:** `models/transformers/schema_rag_net.h:262-298`.
- **Problem:** For every val token it copies `pred` to host and runs an O(vocab) insertion sort for
  top-5 → O(n_val · seq_len · vocab · 5) on the CPU each epoch. (This is the "67-second epoch.")
- **Why it matters:** Pure wall-clock waste. **Note:** it runs *after* `backward` and affects **only
  metric speed — it has zero effect on training stability or accuracy values.** Do it for speed, not
  to fix spikes.
- **Fix:** Write a `calculate_accuracy_kernel` — one thread per token, scan `vocab` keeping the top-5,
  `atomicAdd` into `top1`, `top5`, `total` counters; download three ints at the end. Skip ignore/PAD
  tokens in-kernel.

### 8. CPU round-trip inside PointerAttention backward (every training step)
- **File:** `src/layers/pointer_attention.cu:376-379`.
- **Problem:** `dInput_k.download()` + host loop adding `cached_gate` + `cudaMemcpy` back — a full
  device→host→device round-trip and CPU loop **every backward step**.
- **Why it matters:** Forces a sync and a CPU loop over `N*T_k*D` elements each step. Numerically
  correct, purely a speed bug.
- **Fix:** Replace with on-GPU add: `dInput_k = matrix_add(dInput_k, dSchema_gate)` (the
  `dSchema_gate` currently stashed in `cached_gate`). Remove the download/loop/upload.

### 9. Redundant `cudaDeviceSynchronize()` after every pointer kernel
- **File:** `src/layers/pointer_attention.cu` (`add_gated_k_frozen_cuda`, `sigmoid_inplace_cuda`,
  `sigmoid_backward_inplace_cuda`, `pointer_scatter_add_cuda`, `backward_gated_k_frozen_cuda`).
- **Problem:** Each wrapper calls `cudaDeviceSynchronize()` after launch, serializing the GPU and
  killing kernel overlap. (The core `cuda_ops.cu` kernels mostly do **not** sync — good — so this is
  inconsistent.)
- **Why it matters:** Each sync stalls the pipeline; across a deep model this adds up.
- **Fix:** Remove the per-kernel `cudaDeviceSynchronize()`. Rely on stream ordering; sync only where a
  host read actually needs the result (e.g. before downloading loss/accuracy). Add one
  `cudaDeviceSynchronize()` (or a `cudaGetLastError()` check) at the end of a step in debug builds only.

### 10. seq_len = 256 makes every layer pay O(T²) on mostly-padding
- **Files:** `scripts/prepare_breakwalls.py:67`, attention scores in `self_attention.cpp` /
  `pointer_attention.cu`.
- **Problem:** Attention scores are `N*H*T*T = 256*8*256*256 ≈ 134M` floats ≈ 537 MB **per scores
  tensor**, and there are several per forward/backward. With real content ~40–80 tokens, most of that
  T² is padding attending to padding.
- **Why it matters:** Quadratic compute and memory on garbage. Shrinking seq_len is a near-linear
  speed+memory win across the whole network.
- **Fix:** Measure the actual token-length distribution and set `seq_len` to a tight cap (e.g. 96 or
  128, or length-bucket batches). Combined with loss masking (#1) the trailing padding stops
  contributing at all; capping seq_len stops it consuming compute.

### 11. (Minor) `matrix_multiply` allocates a fresh output Tensor every call
- **File:** `src/core/cuda_ops.cu:16-36` and all callers.
- **Problem:** Every GEMM allocates a new `Tensor C` (cudaMalloc/free churn via shared_ptr) — dozens
  per step.
- **Why it matters:** Allocation overhead and fragmentation; not a correctness issue.
- **Fix (optional, later):** Add a simple device memory pool / arena, or pre-allocate reusable
  workspaces for the hot shapes. Low priority versus #6/#10.

### 12. (Minor) Embedding Adam updates the entire vocab×dim table every step
- **File:** `src/layers/embedding.cpp:37-43`.
- **Problem:** `adam_cuda` runs over all `size*dim` rows even though only the tokens in the batch have
  non-zero grad. At vocab 2000 this is small, but it also means Adam moment decay is applied to rows
  that didn't appear (slightly wrong for unused rows).
- **Why it matters:** Minor now; would matter if vocab grows. Also a subtle correctness nit (decaying
  unused rows).
- **Fix (optional):** Sparse embedding update — only step the rows whose ids appear in the batch.

---

## TIER 4 — Architecture improvements (after Tier 1–3 are in and stable)

### 13. No final LayerNorm before the vocab projection
- **File:** `models/transformers/schema_rag_net.h:60-71`.
- **Problem:** `context + Q_emb` is fed **raw** into `vocab_proj`. The Transformer blocks are pre-LN,
  so the residual stream is never normalized before the output head.
- **Fix:** Add a `LayerNorm(dim)` on the combined representation before `vocab_proj`. Standard and
  cheap; improves stability and final accuracy.

### 14. Tie input embedding and output projection weights
- **Files:** `models/transformers/schema_rag_net.h`, `src/layers/embedding.cpp`, `src/layers/dense.cpp`.
- **Problem:** `query_encoder.token_emb` (`vocab×dim`) and `vocab_proj` (`dim×vocab`) are independent.
- **Why it matters:** Weight tying is standard for LMs — it halves output-head params, regularizes,
  and usually improves accuracy, especially with a small vocab (2000) and limited data.
- **Fix:** Share the embedding matrix as the transpose of the output projection (tie weights; let
  gradients accumulate into one tensor).

### 15. Cross-attention is applied once after the whole stack; schema signal is shallow
- **File:** `models/transformers/schema_rag_net.h:56-77`.
- **Problem:** The schema cross-attention ("pointer") runs **once**, after the 4 decoder blocks, then
  straight to the head. The decoder never conditions on the schema layer-by-layer.
- **Why it matters:** Deep cross-attention (schema injected in each decoder block, encoder-decoder
  style) gives the model many chances to align to the schema. The current design leans almost entirely
  on the causal decoder LM, with the schema as a thin add-on.
- **Fix (bigger):** Move toward a proper encoder-decoder: in each decoder Transformer block, add a
  cross-attention sub-layer (query = decoder state, key/value = schema encoder output) with its own
  residual+LN. This is the highest-ceiling architectural change for schema linking, but do it only
  after #1–#3, and re-tune.

### 16. Add capacity once the signal is clean (you are underfitting)
- **File:** `examples/train_schema_rag.cpp:59-61`.
- **Problem:** train_loss ≈ val_loss ⇒ underfitting. `dim=256, depth=4` is modest.
- **Why it matters:** With masking + real schema + copy head fixed, more capacity should convert into
  accuracy instead of into padding memorization.
- **Fix:** After Tier 1, try `dim 256→384/512`, `depth 4→6`. Re-tune LR (see #18). Watch for the
  train/val gap opening (that's when more data starts to help — see #19).

---

## TIER 5 — Hyperparameters & dataset

### 17. Batch size 256 is large for ~12k examples (~47 steps/epoch)
- **File:** `examples/train_schema_rag.cpp:71`.
- **Fix:** Drop to 64 (≈4× more updates/epoch → faster convergence in step terms). Re-check stability
  with the new clipping in place.

### 18. Learning-rate / warmup
- **File:** `models/transformers/schema_rag_net.h` (LR schedule), `examples/train_schema_rag.cpp:71`.
- **Problem:** peak LR 2e-4 with warmup = epochs/10. After masking changes the effective gradient
  scale, the old LR may be wrong.
- **Fix:** With global-norm clipping in place, 2e-4 is usually fine; if you still see instability,
  lower peak to 1e-4 or lengthen warmup. Re-tune after #1 and #6 (loss normalization changes magnitudes).

### 19. Dataset: quantity is **not** the current bottleneck — formulation is
- **Files:** `scripts/prepare_breakwalls.py`, `scripts/generate_tsql_dataset.py`,
  `data/synthetic_dataset.json`.
- **Problem & why:** train_loss ≈ val_loss proves underfitting, so adding examples now is wasted
  effort. The wins are in #1 (mask), #2 (full schema), #10 (seq_len). Address those first.
- **Then** improve data, in this order:
  1. **Add an EOS token** after each SQL target. Current `combined = inp + [sep] + out` has no
     end marker (`prepare_breakwalls.py:95`); the model never learns to stop, which hurts inference
     and wastes target positions on PAD.
  2. **Upgrade the frozen lexical keys.** `K_frozen` is an MD5 char-trigram hash bag
     (`prepare_breakwalls.py:114-125`) — random buckets, no semantics. Replace with real embeddings
     of the schema strings (even averaged BPE/token embeddings), or drop the frozen-key path if it
     isn't earning its 2048 dims.
  3. **Only after** masking + schema + capacity are in and you see train_loss < val_loss (a real
     generalization gap), scale up the synthetic dataset.

---

## Suggested order of execution
1. #1 loss masking + #6 integer targets (do together — they share the CE-kernel rewrite).
2. #2 full schema tokens.
3. #5 gradient clipping (start with the lighter per-tensor variant if short on time).
4. #7 + #8 + #9 perf (kernel accuracy, remove CPU round-trip, drop redundant syncs).
5. #10 seq_len cap, #17 batch size, #18 LR re-tune.
6. #3 copy head, #13 final LN, #14 weight tying, #4 seed fix.
7. #15 deep cross-attention, #16 capacity, #19 data scale-up — last.

Expected outcome: items #1, #2, #6, #10 alone should break the val_loss ≈ 1.19 plateau and lift
top-k substantially; #3/#15 are what push schema-token accuracy toward the top-5 90% target; #5
makes long high-LR runs survivable.
