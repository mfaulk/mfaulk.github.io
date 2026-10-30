# Deep Learning: From Transformers to Language Models

> **Superseded by `transformers-to-language-models.md`.** Kept only as a record of the
> original plan. The draft reorganizes the post around three reference papers (GPT-1,
> GPT-2, OLMo 2) and corrects several factual claims made below — see the "Outline claims
> that turned out to be wrong" section at the end of the draft before reusing anything
> here. Notably wrong below: the SFT mixture size, the GQA and weight-tying assumptions,
> and the usability of the base model card's benchmark table.

> **Status: outline.** This file is a structural draft for review, not the finished post.
> Prose is compressed to intent, code blocks are skeletons, and every number is a `TODO`.
> Nothing here has been run yet. Delete this block before publishing.

---

## Framing

The [previous post](https://mfaulk.github.io/2026/01/13/transformers.html) built an
encoder-decoder transformer from scratch and trained it to reverse a sequence of random
symbols. That model works, and it is recognizably the architecture from *Attention Is All
You Need* — but it is not a thing you can ask a question.

The question this post answers:

> How do we get from the Transformer described in *Attention Is All You Need* to a model
> that can answer questions, write code, solve mathematical problems, and eventually use
> tools?

The answer has three parts, and only one of them is architecture:

1. **Architecture.** Drop the encoder, drop cross-attention, make the task next-token
   prediction over one sequence. This is a small change, and we can make it by deleting
   code from the previous post.
2. **Scale and data.** Train that model on trillions of tokens of text. This is where
   nearly all of the capability comes from, and it is the part we will *not* do here.
   We will download someone else's.
3. **Post-training.** Take the pretrained model and change how it responds to a request.
   This is the part we *will* do, hands-on, on a GPU you can rent for a few dollars.

**Scope.** The hands-on training experiment in this post is supervised instruction tuning
(SFT) only. Pretraining, preference optimization, reinforcement learning, and tool use are
sketched at the end and deferred to later posts.

**A framing I want to be careful about.** It is common to read that base models "can't
answer questions" or "just autocomplete." That is wrong, and the experiments in §5 are
designed to show it is wrong. A pretrained base model is already an extremely capable
next-token predictor, and with the right prompt it will answer questions and write working
code. What instruction tuning changes is *reliability of form*: whether the model treats an
input as a request to be fulfilled rather than as a document to be continued, whether it
stops when it is done, and whether it does this without prompt engineering. That is a real
and valuable change, and it is a smaller change than it is usually described as.

**Proposed figure 1 — `post-training-pipeline.svg`.**
A horizontal pipeline: `raw text → [pretraining] → base model → [SFT] → instruct model →
[DPO / RLHF] → aligned model → [RLVR] → reasoning model`. The SFT stage boxed/highlighted
as "this post"; pretraining greyed as "previous arrow, next post." Reuse the palette from
`tools/figures/README.md`. This figure recurs at the end of §9 with the later stages
filled in.

---

## 1. Where We Left Off

*Goal: a reader who has not reread the previous post can follow this one. A reader who has
should be able to skim this section in a minute.*

- **Recap the artifact, not the derivation.** The previous post ended with a
  `Transformer(src_vocab, tgt_vocab, d_model=128, num_heads=4, d_ff=512, num_layers=2)`,
  930,828 parameters, trained for 6000 steps to reverse sequences of up to 8 symbols drawn
  from a vocabulary of 12. Exact-match 256/256 on held-out sequences.
- **Recap the toy task and why it was chosen.** Reversal cannot be solved by attending to
  the matching position: generating target position $t$ requires locating source position
  $L - 1 - t$, which depends on the length of the particular input. Getting it right is
  evidence that cross-attention is genuinely aligning.
- **Recap the pieces we are about to keep or discard**, as a table. This sets up §2 by
  making the deletion concrete:

  | Component | Fate in a decoder-only LM |
  | --- | --- |
  | Multi-head self-attention | Kept, always causally masked |
  | Cross-attention sublayer | **Deleted** |
  | Encoder stack | **Deleted** |
  | Position-wise FFN | Kept |
  | Residuals + pre-norm LayerNorm | Kept (LayerNorm → RMSNorm in practice, §2.5) |
  | Sinusoidal positional encoding | Kept in spirit, replaced by RoPE (§2.5) |
  | Separate src/tgt embeddings | Collapsed to one table |
  | Padding mask | Kept |
  | Causal mask | Kept, and now *always* on |
  | Teacher forcing | Kept, and now the entire training objective |
  | Greedy decode loop | Kept, and gets a KV cache (§2.6) |

- **Point forward.** Everything about attention itself — the $QK^\top/\sqrt{d_k}$ scaling,
  why the softmax saturates without it, multi-head projections, why masking sets scores to
  $-\infty$ — is in the previous post and is not repeated here. Link each of those back by
  anchor.

- **One honest caveat to state up front:** the model in the previous post was a
  *translation* architecture applied to a toy task, and the toy task is not what makes the
  jump to language models hard. The jump is data and objective, not depth of the stack.

---

## 2. From Encoder-Decoder to Decoder-Only

*The architectural transition. This is the most code-heavy section of the post.*

### 2.1 Why encoder-decoder in the first place?

- The 2017 architecture was designed for machine translation: a *source* sequence in one
  language, a *target* sequence in another, with a hard boundary between them. The encoder
  is bidirectional because the whole source is available at once. The decoder is causal
  because the target is produced left to right.
- Observe what that split buys and what it costs. It buys bidirectional source encoding.
  It costs: two stacks of parameters, a fixed notion of "input" versus "output," and a
  model that cannot condition on its own output as if it were input.
- **The reframe.** For a language model there is no source/target boundary. There is one
  stream of text, and the task is to predict the next token of it. A prompt is not a
  different *kind* of thing from a completion — it is just the part of the stream we
  happened to supply.
- Note the three-way split in the literature, since readers will encounter all three:
  encoder-only (BERT, masked LM, good for representations), decoder-only (GPT, causal LM,
  good for generation), encoder-decoder (T5, BART, good for seq2seq). We are following the
  decoder-only branch because it is what the frontier models are.

**Proposed figure 2 — `encoder-decoder-vs-decoder-only.svg`.**
Side by side. Left: the previous post's two stacks with the long cross-attention arrow.
Right: a single stack, with the deleted cross-attention sublayer drawn as a dashed ghost
box and the encoder greyed out entirely. Same block vocabulary and colors as the previous
post's figures so the correspondence is immediate.

### 2.2 The objective

- A language model defines a probability distribution over token sequences. Factor it by
  the chain rule:

  $$
  p_\theta(x_1, \dots, x_T) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t})
  $$

- Each factor is exactly what one position of a causally-masked decoder produces: a
  distribution over the vocabulary conditioned on everything to its left. Training
  maximizes the log-likelihood of the corpus, equivalently minimizes the average
  cross-entropy

  $$
  \mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
  $$

- **Connect back to teacher forcing.** The previous post already did this: `tgt_in` and
  `tgt_out` were the same sequence offset by one. A causal LM is that idea with the
  encoder removed — `inputs = tokens[:-1]`, `labels = tokens[1:]`, one tensor, shifted.
  The causal mask is what makes it legal to train all $T$ positions in a single parallel
  pass.
- Mention perplexity as $\exp(\mathcal{L})$ in passing, since §5 and §8 will report it.
- **Note the self-supervision.** No labels are needed. Any text is training data for
  itself. This is the whole reason pretraining scales, and it is the one sentence of
  "why pretraining works" that belongs in this post.

**Proposed figure 3 — `shifted-labels.svg`.**
A token strip, `The cat sat on the mat`, drawn twice and offset by one cell, with arrows
from each input position to the label it predicts. Causal mask triangle shown underneath
so the two line up visually. Weights-sum-to-one convention does not apply here; the
figure is combinatorial, not a distribution.

### 2.3 The code delta

Show the decoder-only block as a literal diff against the previous post's `DecoderBlock`
— three sublayers become two:

```python
class CausalBlock(nn.Module):
    """A decoder block with the cross-attention sublayer removed."""

    def __init__(self, d_model, num_heads, d_ff, dropout_p=0.0, activation="gelu", bias=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout_p=dropout_p, bias=bias)
        # self.cross_attn = ...   <- deleted
        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = ...        <- deleted (was: before cross-attn)
        self.norm3 = nn.LayerNorm(d_model)   # renamed norm2
        self.ffn = nn.Sequential(...)        # unchanged
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.self_attn(x_q=self.norm1(x), mask=mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x
```

- **Point out the punchline:** `CausalBlock` is now identical to the previous post's
  `EncoderBlock`. The only difference between an encoder block and a decoder-only block is
  the mask that gets passed in. The words "encoder" and "decoder" name a usage, not a
  structure. This is worth a paragraph — it is the single clarification that does the most
  to unconfuse the three-architecture taxonomy.

Then the full model, again as a delta:

```python
class CausalLM(nn.Module):
    """Decoder-only autoregressive language model."""

    def __init__(self, vocab_size, d_model=128, num_heads=4, d_ff=512,
                 num_layers=2, dropout_p=0.1, max_len=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)  # one table, not two
        self.pos = PositionalEncoding(d_model, max_len=max_len, dropout_p=dropout_p)
        self.blocks = nn.ModuleList([CausalBlock(...) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight      # weight tying — see below
        # same std = d_model ** -0.5 init as the previous post, for the same reason

    def forward(self, idx, mask=None):
        T = idx.size(1)
        if mask is None:
            mask = causal_mask(T, device=idx.device)
        x = self.pos(self.embed(idx) * math.sqrt(self.d_model))
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.lm_head(self.norm(x))            # (B, T, vocab)
```

Discussion points to write out:

- **Weight tying.** `lm_head.weight = embed.weight` ties the input embedding and the output
  projection. Saves $V \cdot d$ parameters — which at a realistic vocabulary is a large
  fraction of a small model — and is well-motivated: both matrices are maps between token
  identity and model space. Cite Press & Wolf (2017). Note the interaction with the
  $\sqrt{d_{\text{model}}}$ scaling the previous post was careful about.
- **The loss, in code**, showing the shift explicitly:

  ```python
  logits = model(tokens[:, :-1])
  loss = F.cross_entropy(
      logits.reshape(-1, vocab_size),
      tokens[:, 1:].reshape(-1),
      ignore_index=PAD,
  )
  ```

- **Why `causal_mask` is now unconditional**, and the padding-mask interaction from the
  previous post (`causal & padding`, and the all-masked-row `NaN` caveat) still applies.

### 2.4 A sanity-check experiment (small, ours, runs on CPU)

Keep one scratch-built experiment so the post is not purely "download a model."

- **Task proposal:** train `CausalLM` on a character-level corpus small enough to ship —
  the standard tiny-Shakespeare file, or a synthetic grammar. ~1M characters, minutes on
  CPU.
- **Alternative, and I think the better one:** reuse the reversal task from the previous
  post, reformatted as a single stream: `3 11 7 | 7 11 3 <eos>`. This makes the
  encoder-decoder → decoder-only transition *directly comparable*, because it is the same
  task solved by both architectures, and it previews the prompt/response structure that
  §6 formalizes with a chat template. The `|` separator is a proto-chat-template.
  → **Open question for review: pick one. I lean toward the reversal-as-one-stream version
  for continuity, possibly with tiny-Shakespeare as a short aside for the "it generates
  text" demo.**
- **TODO(experiment):** train it, report parameter count, loss curve, exact-match accuracy
  against the previous post's 256/256. Report whether the decoder-only model matches the
  encoder-decoder on the same task and at comparable parameter count. Do not guess the
  answer here — run it.
- **TODO(figure):** loss curve, or a side-by-side bar of the two architectures' accuracy.

### 2.5 What a modern decoder-only model changes

*A short reference section. The point is that a reader opening the OLMo 2 source in §4
recognizes everything they see.* Each item: one paragraph, what it replaces, why.

- **RoPE** replaces additive sinusoids. Rotate $q$ and $k$ inside attention by an angle
  proportional to position, so the dot product depends on *relative* separation. The
  previous post flagged this in one sentence; here it gets the full treatment, with the
  2-D rotation written out and a note on why it extrapolates better. Su et al. (2021).
- **RMSNorm** replaces LayerNorm: drop the mean subtraction and the $\beta$ shift, keep
  the rescaling. $\text{RMSNorm}(x) = \gamma \odot x / \sqrt{\frac{1}{d}\sum x_i^2 +
  \epsilon}$. Cheaper, empirically no worse. Zhang & Sennrich (2019).
- **SwiGLU** replaces the two-layer ReLU/GELU FFN with a gated variant, and $d_{ff}$ drops
  from $4d$ to about $\tfrac{8}{3}d$ to hold the parameter count. Shazeer (2020).
- **Grouped-query attention (GQA)** shares key/value heads across query heads, shrinking
  the KV cache. Ainslie et al. (2023). *Verify whether OLMo 2 1B uses GQA or full MHA
  before asserting it — the config says 16 heads; check `num_key_value_heads`.*
- **OLMo 2's specific deviations** — QK-norm, the reordered norm placement ("OLMo 2 uses
  post-norm applied to the sublayer output inside the residual branch," *verify against the
  paper, this is the detail most likely to be stated wrong*), z-loss. Cite the OLMo 2
  paper directly rather than paraphrasing from memory.
- **Tokenization.** Not architecture, but it belongs here: BPE, why not characters
  (sequence length) and why not words (open vocabulary), and the practical consequences a
  reader will trip over — tokens are not words, whitespace is part of the token, and
  `" the"` ≠ `"the"`. Keep to half a page and link out for the rest. Sennrich et al.
  (2016). **TODO(code):** a six-line snippet tokenizing one sentence with OLMo 2's
  tokenizer and printing the ids and the decoded pieces, so the reader sees it.

### 2.6 Generation, and the KV cache

- The previous post deferred the KV cache explicitly. Collect on that debt here, because
  every experiment below depends on generation being fast enough to be interactive.
- The argument: each decode step re-runs the stack over the whole prefix, recomputing $K$
  and $V$ for positions that have not changed. Cache them per layer, compute only the new
  row. $O(n^2)$ total work becomes $O(n)$ steps of $O(n)$ attention. No mathematics
  changes.
- **TODO(code):** minimal cache in `CausalLM.generate`, ~20 lines.
- **TODO(measurement):** tokens/sec with and without the cache on the toy model. Small
  table.
- **Sampling.** Greedy is what the previous post used and it is wrong for open-ended text —
  it produces loops. Introduce temperature, top-$k$, and top-$p$, with the one-line
  definitions and a worked example of how each reshapes a distribution. This matters for
  §5: *how you sample is a confound in every base-vs-tuned comparison*, so fix the sampling
  parameters once, state them, and reuse them everywhere.

---

## 3. What Is a Base Model?

*Short, conceptual, and deliberately incomplete — the deep version is its own post.*

- **Definition.** A base (or "pretrained," or "foundation") model is the result of step 2:
  a decoder-only transformer trained on a large text corpus with the next-token objective
  and nothing else. No instructions, no demonstrations of helpfulness, no human preference
  signal. Its entire training signal is "what token comes next in this document."
- **What that produces.** Enough of the structure of the corpus to be useful: syntax, facts,
  code, arithmetic, translation, and the ability to continue a pattern it has been shown.
  The in-context learning result (Brown et al., 2020) is the headline: a base model given
  a few worked examples in its prompt will continue the pattern, with no gradient update.
- **What it does not produce.** A notion that the text in its context is a *request*. A
  base model's prior is "this is a document." Faced with `What is the capital of France?`
  the most likely continuation of that document, in the corpus, might be another quiz
  question. That is not a failure of knowledge, it is a correct prediction about documents.
- **Scope note, stated plainly:** *why* next-token prediction on web text yields arithmetic
  and code and reasoning is a genuinely deep question, and it is not this post's. Flag it,
  gesture at scaling laws (Kaplan et al. 2020; Hoffmann et al. 2022) and emergence, and
  move on with an explicit forward reference.
- **Open vs. open-weight vs. closed.** Brief, because it justifies choosing OLMo: most
  "open" models release weights only. OLMo releases weights, data, code, and intermediate
  checkpoints, which is what makes it a usable teaching object — we can point at the exact
  data and the exact SFT recipe, and compare our fine-tune against the official one.

---

## 4. OLMo 2 1B

*Introduce the specific model everything downstream uses.*

- **Why this model.** Fully open (data + code + weights + checkpoints, Apache 2.0); small
  enough to fine-tune on one consumer GPU; and — the decisive property — AI2 publishes the
  base, SFT, DPO, and RLVR checkpoints *of the same model*, so §8 can compare our fine-tune
  against the official one rather than against a different model family.
- **The facts**, as a table. Verified against the model card:

  | | |
  | --- | --- |
  | Base | `allenai/OLMo-2-0425-1B` |
  | SFT | `allenai/OLMo-2-0425-1B-SFT` |
  | DPO | `allenai/OLMo-2-0425-1B-DPO` |
  | Instruct (RLVR) | `allenai/OLMo-2-0425-1B-Instruct` |
  | Parameters | ~1B |
  | Layers / hidden / heads | 16 / 2048 / 16 |
  | Context length | 4096 |
  | Pretraining data | OLMo-mix-1124 (3.9T tokens), then Dolmino-Mix-1124 mid-training |
  | Total pretraining tokens | ~4T |
  | SFT data | `allenai/tulu-3-sft-olmo-2-mixture-0225` (~939k examples) |
  | Paper | [arXiv:2501.00656](https://arxiv.org/abs/2501.00656) |
  | License | Apache 2.0 |

  **TODO(verify):** re-check `num_key_value_heads`, vocab size, and the exact tokenizer
  name from `config.json` before publishing, rather than from the model card prose.

- **Map the config onto the code from §2.3.** Load `config.json` and walk the fields:
  `hidden_size` → `d_model`, `num_hidden_layers` → `num_layers`, `intermediate_size` →
  `d_ff`, `rope_theta`, `rms_norm_eps`. The point is that the reader has already built
  every one of these; the published model is the same object with bigger numbers and the
  §2.5 substitutions.
- **TODO(code):** load the model, print the module tree, print the parameter count, and
  compare the printed tree line-by-line against `CausalLM`. This is the moment the post
  earns its title — the thing on HuggingFace is visibly the thing we built.
- **Practical setup**, in a collapsible or a short block: `transformers`, `torch`, `trl`,
  `peft`, `datasets` with pinned versions; VRAM needed for inference (bf16, ~2.5 GB) and
  for LoRA SFT (**TODO: measure**); a note that everything in the post runs on one 16 GB
  GPU or a free Colab T4 at reduced batch size. **TODO(decide):** ship a companion repo or
  notebook? I lean yes — a `tools/olmo-sft/` directory in this repo, following the
  `tools/figures/` precedent, with a README.

---

## 5. What the Base Model Can and Cannot Do

*The empirical heart of the "before" half. Every item here is a TODO — no results are
invented. Fix sampling parameters once (temperature, top-p, seed, max_new_tokens) and
state them, so the comparison in §8 is controlled.*

### 5.1 It is a very good next-token predictor

- **Experiment A — text continuation.** Feed a few sentences of prose; show the
  continuation. Expect fluent, on-topic, grammatical text that does not stop on its own.
  **TODO(run).** Show the raw output including the run-on, un-truncated.
- **Experiment B — code completion.** Give a Python function signature and docstring; show
  the body. This is deliberately chosen to be a task where the base model should do *well*,
  because code completion is exactly a continuation task. **TODO(run).**
  Expected observation to verify: the model also happily continues past the function into a
  second, unrequested function — correct document-continuation behavior, unhelpful
  assistant behavior.
- **Experiment C — few-shot prompting.** Three worked examples of some mapping, then a
  fourth input. Show that it completes the pattern. **TODO(run).** This is the experiment
  that makes the §3 claim concrete: the capability is already there, and the prompt is what
  unlocks it.
- **Experiment D — quantitative baseline.** Perplexity on a held-out slice of something the
  model did not train on. **TODO(run).** Gives §8 a number that is not vibes.

### 5.2 Where it is unreliable

- **Experiment E — a bare instruction.** `Write a haiku about recursion.` with no scaffold.
  **TODO(run).** Report what actually happens, honestly, including if it works fine. A
  1B base model's behavior here is genuinely variable across seeds, and *that variability
  is the finding* — run it with N seeds and report the distribution, not one cherry-picked
  failure.
- **Experiment F — the same instruction, scaffolded.** Wrap it as a document the corpus
  contains: `Q: ... \nA:`, or a few-shot instruction prompt. **TODO(run).** Expect it to
  work much better. The contrast between E and F is the single most important result in
  the post: *the capability did not change, the elicitation did.*
- **Experiment G — stopping.** Measure how often generation terminates on its own vs. runs
  to `max_new_tokens`. **TODO(run).** Concrete, countable, and it is the failure mode SFT
  most obviously fixes.
- **Experiment H — format compliance.** Ask for JSON, or exactly three bullets, or a
  one-word answer. Score mechanically over N prompts. **TODO(run).** This is the IFEval
  axis in miniature and it is what §8 will re-measure after tuning.

### 5.3 Reading the results honestly

- Write the conclusion as: base models are not *incapable*, they are *unaligned to the
  request frame*. The deltas to expect from SFT are in stopping, format compliance, and
  robustness to un-engineered prompts — not in whether the model knows what a haiku is.
- Also state the cost: instruction tuning is not free. The OLMo 2 model card's own
  benchmark table shows the post-trained checkpoints are not uniformly better than the base
  model on every benchmark. **TODO(verify + reproduce the table from the model card, with
  correct row labels — my fetched copy of it was garbled, do not transcribe from notes).**
  This is a useful honesty beat and a good lead-in to §9.

---

## 6. Supervised Instruction Tuning

*The conceptual core of the "after" half. Four subsections, each one thing.*

### 6.1 The idea

- Same model, same architecture, same loss function. The *only* changes are (a) what text
  we train on and (b) which tokens the loss is computed over.
- Framing: SFT does not teach the model facts or skills. It teaches a *format* — the
  conditional distribution of "assistant response given user request" — and in doing so
  makes the request frame the model's default prior instead of a thing you have to prompt
  your way into. State this clearly, because it is the mental model that makes the rest
  of post-training make sense.

### 6.2 The data

- Shape: `(prompt, response)` pairs, optionally multi-turn. Show two or three real examples
  from `tulu-3-sft-olmo-2-mixture-0225`, rendered as the raw JSON `messages` list.
- Where it comes from: human-written, model-distilled, and templated-from-existing-datasets.
  Tülu 3's mixture is a good concrete case to describe — its composition is public, it
  deliberately includes math, code, precise-instruction-following, and safety subsets, and
  the mixture ratios were themselves tuned. Cite the Tülu 3 paper.
- Scale: ~939k examples for the official checkpoint. **We will use a few thousand.** Say
  this early and loudly, because it sets expectations for §8: we are demonstrating the
  *direction* of the change on a budget, not reproducing the checkpoint.
- A short note on data quality > data quantity in SFT (LIMA, Zhou et al. 2023), because it
  is both true and the reason a small experiment is informative at all.

### 6.3 Chat templates

- **The problem.** The model consumes a flat token sequence. A conversation is structured.
  Something has to serialize one into the other, unambiguously, including marking where a
  turn ends.
- **The mechanism.** Special tokens plus a fixed layout, stored as a Jinja template in the
  tokenizer config and applied by `tokenizer.apply_chat_template`.
- OLMo 2 / Tülu format (**TODO: verify exact whitespace and `<|endoftext|>` placement by
  calling `apply_chat_template` and printing the raw string with `repr()` — do not
  transcribe this from the model card prose, it is the kind of detail that is wrong in
  every blog post**):

  ```
  <|user|>
  {user message}
  <|assistant|>
  {assistant response}<|endoftext|>
  ```

- **Why this is the hinge of the whole section.** `<|assistant|>` is a token that, after
  SFT, is reliably followed by a helpful response — because in every training example it
  was. And `<|endoftext|>` at the end of each response is what teaches the model to *stop*,
  which is Experiment G's failure fixed by construction.
- **The practical warning.** Template mismatch between training and inference silently
  destroys quality. Train with one template, serve with another, and you get a model that
  looks broken for no visible reason. Show the exact code that guarantees they match.
- **Generation prompt.** `add_generation_prompt=True` appends the opening `<|assistant|>`
  at inference so the model's next token is the start of a response. Show it.

**Proposed figure 4 — `chat-template-loss-mask.svg`.**
One token strip of a rendered training example. Role tokens in one color, prompt tokens
greyed with `label = -100`, response tokens colored with `label = token`. An annotation
bracket under the response span reading "loss computed here." This figure carries §6.3 and
§6.4 at once and is the figure I most want in the post.

### 6.4 The loss, and response-token masking

- The loss is the same cross-entropy from §2.2. The difference is the label tensor.
- **The argument for masking.** If the loss includes prompt tokens, the model spends
  capacity learning to *generate user messages* — a distribution we have no interest in
  modeling and, with templated or synthetic prompts, one that is badly non-representative.
  We want $p_\theta(\text{response} \mid \text{prompt})$, so we score only response tokens.
- Mechanically: set `labels[i] = -100` on prompt and role tokens; `F.cross_entropy`'s
  default `ignore_index=-100` drops them. Note the direct parallel to `ignore_index=PAD`
  in the previous post's training loop — same mechanism, different motivation.
- **TODO(code):** the collator. Build `input_ids`, `attention_mask`, and `labels`; assert
  that the number of unmasked labels equals the number of response tokens; print one
  example decoded with masked positions shown as `·`. Include the assertion in the post —
  off-by-one errors here are common and silent.
- **Is masking actually necessary?** It is standard and well-motivated, but the measured
  effect size depends on the data. **TODO(optional experiment): train one variant without
  masking and compare.** Only include this if it fits the post's length budget; mark as a
  stretch goal.
- **Packing vs. padding.** One paragraph on concatenating short examples to fill the
  context window, the cross-contamination hazard, and why we will keep it simple and pad.

---

## 7. Fine-Tuning OLMo 2 with LoRA

*The hands-on experiment. This section should be reproducible end to end.*

### 7.1 Why not full fine-tuning

- Full SFT of a 1B model needs weights + gradients + Adam's two moments: roughly 16 bytes
  per parameter in mixed precision, ≈16 GB before activations. Walk the arithmetic
  explicitly — it is a genuinely useful thing for a reader to be able to do, and it
  motivates everything that follows.
- Note that full fine-tuning *is* feasible for 1B on a rented A100, and that we are
  choosing LoRA for reproducibility on commodity hardware, not out of necessity. Being
  straight about this is better than implying LoRA is required.

### 7.2 LoRA

- **The method.** Freeze $W_0 \in \mathbb{R}^{d \times k}$. Learn a low-rank update:

  $$
  W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A, \qquad
  B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d, k)
  $$

- $A$ initialized Gaussian, $B$ initialized to zero, so $\Delta W = 0$ at step 0 and
  training starts exactly at the pretrained model. Say why that matters.
- $\alpha/r$ is a scaling that decouples the effective learning rate from the rank, so
  changing $r$ does not force retuning the learning rate.
- Parameter arithmetic: for a $2048 \times 2048$ projection at $r = 16$, $2 \cdot 2048
  \cdot 16 = 65{,}536$ trainable parameters against $4{,}194{,}304$ frozen — 1.6%.
  **TODO:** compute the real total over all targeted modules and report trainable/total.
- **The hypothesis it rests on**, stated as a hypothesis: the weight update needed to adapt
  a pretrained model to a downstream task has low intrinsic rank. Hu et al. (2021).
- At inference, $BA$ can be merged into $W_0$, so there is no added latency. Mention
  briefly; it is the practical reason LoRA won.
- Which modules to target: attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
  and optionally the MLP (`gate_proj`, `up_proj`, `down_proj`). **TODO(decide + justify):**
  attention-only is the classic recipe; including the MLP is now common. Pick one, state
  why, and note the parameter cost of each.

**Proposed figure 5 — `lora-adapter.svg`.**
Input vector → a frozen grey $W_0$ box and, in parallel, a thin $A$ (down to $r$) then $B$
(back up to $d$) in the accent color, summed at the output. Dimensions labeled on every
edge; the visual point is the narrow waist at $r$.

### 7.3 The training run

**TODO(code):** full script, ~60 lines, using `peft` + `trl`'s `SFTTrainer` or a plain
PyTorch loop. **Decide which (open question for review):** `SFTTrainer` is shorter and
is what a reader would actually use; a hand-rolled loop is more in keeping with the
previous post's from-scratch ethos and makes the masking visible. *I lean toward the plain
loop, with `SFTTrainer` shown afterward as the two-line production equivalent.*

```python
# Sketch only.
base = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B", torch_dtype=torch.bfloat16)
tok  = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")

model = get_peft_model(base, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=[...],           # TODO: decide, see 7.2
    task_type="CAUSAL_LM",
))
model.print_trainable_parameters()   # TODO: paste real output

# dataset: N examples sampled from allenai/tulu-3-sft-olmo-2-mixture-0225
# collator: apply_chat_template + response-only label masking (6.4)
# loop: AdamW, cosine schedule with warmup, grad clipping, bf16
```

Hyperparameters to state and justify, all **TODO(decide/measure)**: dataset subset size and
how it was sampled; `max_seq_length`; LoRA $r$ and $\alpha$; learning rate (1e-4–2e-4 is
the usual LoRA range, well above a full-FT rate — explain why); batch size and gradient
accumulation; epochs (1–3; note overfitting risk on a small subset); warmup.

**TODO(run + report):** wall-clock time, peak VRAM, hardware used, final train loss,
held-out loss on a validation split of the mixture. **TODO(figure):** training and
validation loss curves.

**Failure modes worth documenting if they occur** — these are the parts of a tutorial that
are actually scarce, so write them down as they happen rather than cleaning them up:
OOM and what was reduced; loss not moving (usually LR too low, or adapters not actually
attached — show `print_trainable_parameters()` as the check); template mismatch producing
garbage at inference; forgetting `add_generation_prompt=True`.

---

## 8. Evaluation

*Three-way comparison: base, ours, official SFT. Every number TODO.*

### 8.1 What we are comparing and how

| Model | Role |
| --- | --- |
| `allenai/OLMo-2-0425-1B` | base — the "before" |
| ours: base + LoRA on N examples | the "after," on a hobbyist budget |
| `allenai/OLMo-2-0425-1B-SFT` | the official SFT, ~939k examples — the ceiling for this stage |

- **State the expected ordering up front and commit to reporting it even if it comes out
  differently:** ours should land between base and official SFT on format and stopping, and
  may be *worse* than base on knowledge benchmarks. If it isn't, say so and investigate.
- Controls: identical prompts, identical sampling parameters, identical seeds, identical
  chat template handling per model (the base model has no template — **TODO: decide and
  document how the base model is prompted for each eval, since this choice drives the
  result**).

### 8.2 Quantitative

- **Held-out loss / perplexity** on a validation split of the SFT mixture. Easy, cheap,
  and directly measures "did it learn the response distribution."
- **Instruction-following, mechanically scored.** A small IFEval-style set of verifiable
  constraints ("answer in exactly one word", "output valid JSON", "use no commas"). Score
  with a checker function, not a judge. **TODO(build the set + run).**
- **Stopping rate.** Fraction of generations that emit EOS before `max_new_tokens`.
  Re-run Experiment G from §5.2 on all three models. This is the cleanest single number
  in the post.
- **A capability benchmark that is not about format** — GSM8K or MMLU, small subset — to
  test the alignment-tax claim from §5.3. **TODO(run).** Report honestly if our fine-tune
  degrades it.
- Present as one table, three rows, with the §5 base-model numbers reused so the reader
  can see the delta.

### 8.3 Qualitative

- The same prompts from §5, all three models, side by side, outputs quoted verbatim and
  untruncated. **TODO(run).** This is the part readers will actually read.
- Include at least one case where our model is clearly worse than the official SFT, and
  say what the likely cause is (data volume, LoRA rank, single epoch).

### 8.4 What this does and does not show

- Honest limits: tiny eval sets, no statistical significance, one seed per cell unless
  budget allows more, LoRA ≠ full SFT, our data subset is a sample of a mixture that was
  itself carefully balanced.
- The claim the post is entitled to make: supervised instruction tuning reliably changes
  *how the model responds to a request*, measurably, with a few thousand examples and an
  hour of GPU time. It does not make the model smarter.

---

## 9. What Comes Next

*Brief — one or two paragraphs each, each one a forward reference to a future post. No
code, no experiments.*

- **Why SFT is not enough.** SFT imitates demonstrations. It has no way to express "this
  response is better than that one," no signal about what to avoid, and it is bounded by
  the quality of the demonstrations — the model can only learn to be as good as its data.
- **Preference optimization.** Collect pairs $(y_w, y_l)$ — preferred and dispreferred
  responses to the same prompt — and optimize the model to prefer $y_w$. Sketch RLHF's
  reward-model + PPO pipeline (Ouyang et al. 2022) and DPO's reward-free reformulation
  (Rafailov et al. 2023), and note that `OLMo-2-0425-1B-DPO` is exactly this applied to
  the §8 model, so the reader can run it themselves today.
- **Reinforcement learning with verifiable rewards (RLVR).** For math, code, and
  instruction-following, correctness is *checkable* — run the tests, check the answer. That
  replaces a learned reward model with ground truth and removes reward hacking as a failure
  mode. This is what produced `OLMo-2-0425-1B-Instruct`, and the idea behind the
  reasoning-model wave (Lambert et al., Tülu 3; DeepSeek-R1).
- **Reasoning.** Long chains of thought, learned rather than prompted; test-time compute as
  a scaling axis.
- **Tool use and agents.** Once a model reliably follows a format, that format can be a
  function call. The jump from "emit `<|assistant|>` then helpful text" to "emit a tool
  call, read the result, continue" is smaller than it looks — and it is the same chat
  template mechanism from §6.3 with another special token.
- **Figure 1 reprised**, with the later stages filled in and the next post's scope boxed.
- **The deferred question**, restated so the series has an explicit thread: *how* does
  next-token prediction on a corpus produce knowledge and reasoning in the first place?
  That is the pretraining post.

---

## References and Further Reading

*Format to match the previous post: numbered list, linked titles, parenthetical gloss.
Verify every link resolves before publishing.*

**Architecture**
1. [Attention is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
1. [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (Radford et al., 2018 — GPT-1, the decoder-only turn)
1. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford et al., 2019 — GPT-2)
1. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Brown et al., 2020 — GPT-3, in-context learning)
1. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021 — RoPE)
1. [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) (Zhang & Sennrich, 2019)
1. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (Shazeer, 2020 — SwiGLU)
1. [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023)
1. [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) (Press & Wolf, 2017 — weight tying)
1. [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (Sennrich et al., 2016 — BPE)

**Scale and pretraining** *(deferred to a later post; listed for the curious)*
1. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020)
1. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Hoffmann et al., 2022 — Chinchilla)

**OLMo**
1. [OLMo 2: Furious](https://arxiv.org/abs/2501.00656) (OLMo team, 2025) — *TODO: confirm the paper's exact title string*
1. [`allenai/OLMo-2-0425-1B`](https://huggingface.co/allenai/OLMo-2-0425-1B) model card
1. [`allenai/tulu-3-sft-olmo-2-mixture-0225`](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture-0225) dataset card
1. [OLMo release notes](https://allenai.org/olmo/release-notes)

**Instruction tuning and post-training**
1. [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) (Wei et al., 2021 — FLAN)
1. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) (Ouyang et al., 2022 — InstructGPT)
1. [Self-Instruct](https://arxiv.org/abs/2212.10560) (Wang et al., 2022)
1. [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) (Zhou et al., 2023)
1. [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124) (Lambert et al., 2024)
1. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023)
1. [Instruction-Following Evaluation for Large Language Models](https://arxiv.org/abs/2311.07911) (Zhou et al., 2023 — IFEval)

**Efficient fine-tuning**
1. [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (Hu et al., 2021)
1. [QLoRA](https://arxiv.org/abs/2305.14314) (Dettmers et al., 2023)
1. [PEFT](https://huggingface.co/docs/peft) and [TRL](https://huggingface.co/docs/trl) documentation

**Background**
1. [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) (Jay Alammar)
1. [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) (Karpathy)

---

## Open Questions for Review

1. **§2.4 toy task** — reversal-as-one-stream (continuity with the previous post, previews
   the prompt/response split) or tiny-Shakespeare (more obviously "a language model")? I
   lean reversal, with Shakespeare as a short aside.
2. **§7.3 training code** — hand-rolled PyTorch loop or `trl`'s `SFTTrainer`? I lean the
   loop first, `SFTTrainer` after as the production shortcut.
3. **Companion code** — add `tools/olmo-sft/` to this repo (precedent: `tools/figures/`),
   or a standalone Colab notebook, or both?
4. **Length.** The previous post is ~1200 lines. This outline is reaching for comparable
   depth across more ground. Candidates to cut if it runs long: §2.5 (compress to a table),
   §2.6 KV cache (could be its own short post), §6.4's optional no-masking ablation.
5. **Figure budget** — five new SVGs is a lot of generator scripts. Which are load-bearing?
   My ranking: figure 4 (loss mask) > figure 2 (arch delta) > figure 5 (LoRA) >
   figure 1 (pipeline) > figure 3 (shifted labels).

## Outstanding Verification Before Writing

- OLMo 2 1B `config.json`: `num_key_value_heads`, `vocab_size`, tokenizer name, norm
  placement, whether QK-norm is used.
- The exact rendered chat template, via `repr(tokenizer.apply_chat_template(...))`.
- The OLMo 2 model-card benchmark table, with correct row labels (my fetched copy had the
  base/Instruct rows garbled — read the card directly).
- The OLMo 2 paper's exact title and author list.
- Current `transformers` / `peft` / `trl` API surface for the code snippets; pin versions.
