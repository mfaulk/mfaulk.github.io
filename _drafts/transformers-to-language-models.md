# Deep Learning: From Transformers to Language Models

> **Status: draft.** The conceptual half of this post is written. The experimental half is
> structured but unrun — every result is marked `TODO`, and nothing is invented. See the
> working notes at the end. Delete this block and that one before publishing.

The [previous post](https://mfaulk.github.io/2026/01/13/transformers.html) built an encoder-decoder transformer from scratch and trained it to reverse sequences of symbols. That model works, and it is recognizably the architecture from *Attention Is All You Need* — but it is not a thing you can ask a question. This post is about the distance between those two objects:

> How do we get from the Transformer described in *Attention Is All You Need* to a model that answers questions, writes code, and follows instructions?

The answer has three parts, and only the first is architecture.

1. **Architecture.** Delete the encoder, delete cross-attention, and make the task next-token prediction over a single stream of text. This is a surprisingly small change, and we will make it by removing code from the previous post.
2. **Scale and data.** Train that model on trillions of tokens. This is where nearly all of the capability comes from, and it is the part we will not do here — we will download someone else's.
3. **Post-training.** Take the pretrained model and change how it responds to a request. This is the part we will do, hands-on.

Like the previous post, this one stays close to its sources. Three papers carry the argument:

- [**Improving Language Understanding by Generative Pre-Training**](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (Radford, Narasimhan, Salimans, and Sutskever, 2018) — now universally called GPT-1 — makes the architectural turn. It throws away the encoder, keeps masked self-attention, trains on next-token prediction, and then fine-tunes the result on individual tasks. It is the origin of the pretrain-then-adapt paradigm that the rest of the post is about.
- [**Language Models are Unsupervised Multitask Learners**](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford, Wu, Child, Luan, Amodei, and Sutskever, 2019) — GPT-2 — argues that the fine-tuning step is optional. A large enough next-token predictor will perform tasks it was never trained on, if you phrase the task as text. This is the paper that explains what a *base model* is and what it can do before anyone instruction-tunes it.
- [**2 OLMo 2 Furious**](https://arxiv.org/abs/2501.00656) (OLMo Team, 2025) describes a model family built the way models are built now: RoPE, RMSNorm, QK-norm, a reordered residual stream, and a published post-training pipeline. OLMo 2 is fully open — data, code, weights, and intermediate checkpoints — which makes it the model we can actually take apart.

**Scope.** The hands-on training experiment in this post is supervised instruction tuning (SFT) only. Pretraining, preference optimization, and reinforcement learning are sketched at the end and left to later posts.

**One framing to be careful about.** It is common to read that base models "can't answer questions" or "just autocomplete." That is misleading, and GPT-2 is the paper that says why. A pretrained base model is an extremely capable next-token predictor, and with the right prompt it will answer questions and write working code. What instruction tuning changes is the *reliability of form*: whether the model treats its input as a request to be fulfilled rather than a document to be continued, whether it stops when it is done, and whether it does these things without prompt engineering. That is a real and valuable change, and it is a smaller change than it is usually described as.

---

## Where We Left Off

The previous post ended with an encoder-decoder transformer trained to reverse a sequence:

```python
model = Transformer(src_vocab_size=12, tgt_vocab_size=12, d_model=128,
                    num_heads=4, d_ff=512, num_layers=2)
# parameters: 930828
```

Six thousand steps of training brought the loss to 0.0059 and exact-match accuracy to 256/256 on held-out sequences. The toy task was chosen because it cannot be solved by attending to the matching position: to produce target position $t$, the decoder has to locate source position $L - 1 - t$, which depends on the length of the particular input. Getting it right is evidence that cross-attention is genuinely aligning rather than copying.

Everything about attention itself — the $QK^\top/\sqrt{d_k}$ scaling and why softmax saturates without it, the multi-head projections, why masking sets scores to $-\infty$, why pre-norm beats post-norm — is in that post and is not repeated here. What matters for this one is the inventory of parts, and which of them survive the transition:

| Component | Fate in a decoder-only language model |
| --- | --- |
| Multi-head self-attention | Kept, and now always causally masked |
| Cross-attention sublayer | **Deleted** |
| Encoder stack | **Deleted** |
| Position-wise feed-forward network | Kept |
| Residual connections | Kept, untouched |
| Pre-norm `LayerNorm` | Kept at first; becomes `RMSNorm` on the sublayer *output* |
| Sinusoidal positional encoding | Kept in spirit, replaced by RoPE |
| Separate source/target embeddings | Collapsed into one table |
| Padding mask | Kept |
| Causal mask | Kept, and now unconditional |
| Teacher forcing | Kept, and now the entire training objective |
| Greedy decoding loop | Kept, and acquires a KV cache |

Two of these are worth flagging before we start. The causal mask stops being a special case and becomes the only case. And teacher forcing — feeding the decoder the correct prefix and asking it to predict the next token — stops being a training trick for a translation model and becomes the whole of what a language model is.

---

## The Decoder-Only Turn

### Why there were two stacks in the first place

The 2017 architecture was designed for machine translation, and translation has a hard boundary in the middle of it. There is a source sequence in one language and a target sequence in another. The encoder is bidirectional because the entire source is available before any output is produced; every source position can attend to every other. The decoder is causal because the target is produced left to right, one token at a time, and position $t$ must not see position $t+1$.

That split buys bidirectional source encoding. What it costs is less obvious but more consequential:

- Two stacks of parameters, plus the cross-attention sublayer that connects them.
- A fixed, architectural commitment about which text is "input" and which is "output."
- A model that cannot condition on its own output as though it were input, because its own output arrives on the wrong side of the boundary.

For a language model, none of that structure is wanted. There is one stream of text, and the task is to predict the next token of it. A prompt is not a different *kind* of object from a completion — it is simply the part of the stream we happened to supply. Once you believe that, the encoder has nothing to encode, and cross-attention has nothing to cross to.

The literature settled into three branches, and it is worth naming all three because readers will meet them all:

| Family | Attention | Objective | Good at |
| --- | --- | --- | --- |
| Encoder-only (BERT) | Bidirectional | Masked token prediction | Representations, classification |
| Encoder-decoder (original Transformer, T5) | Bidirectional source, causal target | Sequence-to-sequence | Translation, summarization |
| Decoder-only (GPT, OLMo) | Causal | Next-token prediction | Generation, and as it turned out, everything else |

We are following the third branch, because it is what the frontier models are.

### "A multi-layer Transformer decoder"

GPT-1 states the architecture in a single sentence:

> In our experiments, we use a multi-layer Transformer decoder [34] for the language model, which is a variant of the transformer [62]. This model applies a multi-headed self-attention operation over the input context tokens followed by position-wise feedforward layers to produce an output distribution over target tokens.
>
> — *Improving Language Understanding by Generative Pre-Training*, §3.1

Note what is in that description and what is absent. Multi-head self-attention: present. Position-wise feed-forward layers: present. An encoder, and attention over its output: not mentioned, because they are gone. The paper writes the whole forward pass as three lines:

$$
\begin{aligned}
h_0 &= U W_e + W_p \\
h_l &= \texttt{transformer\_block}(h_{l-1}) \quad \forall l \in [1, n] \\
P(u) &= \text{softmax}(h_n W_e^\top)
\end{aligned}
$$

where $U = (u_{-k}, \dots, u_{-1})$ is the context of tokens, $n$ is the number of layers, $W_e$ is the token embedding matrix, and $W_p$ the position embedding matrix. Three details are hiding in those lines:

- $W_p$ is a *learned* position embedding matrix, added to the token embeddings. GPT-1 says so explicitly — "We used learned position embeddings instead of the sinusoidal version proposed in the original work" — and that is one of only a handful of ways it departs from 2017.
- The output projection is $W_e^\top$: the same matrix as the input embedding, transposed. The input embedding and the output head are tied. We will come back to this, because it is a decision that modern models go both ways on.
- `transformer_block` is called with one argument. In the previous post, `DecoderBlock.forward` took the decoder state, the encoder output, a self-attention mask, and a cross-attention mask. Here there is only $h_{l-1}$. The signature is the architecture.

### The objective

A language model defines a probability distribution over token sequences. Factor it with the chain rule:

$$
p_\theta(x_1, \dots, x_T) = \prod_{t=1}^{T} p_\theta(x_t \mid x_{<t})
$$

Each factor on the right is exactly what one position of a causally-masked decoder produces: a distribution over the vocabulary, conditioned on everything to its left. Maximizing the log-likelihood of a corpus is the same as minimizing the average cross-entropy

$$
\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

which GPT-1 writes, in maximization form and over a window of $k$ tokens, as

$$
L_1(\mathcal{U}) = \sum_i \log P(u_i \mid u_{i-k}, \dots, u_{i-1}; \Theta)
$$

Perplexity, which shows up in every language modeling result including GPT-1's "very low token level perplexity of 18.4," is just $\exp(\mathcal{L})$ — the effective number of equally-likely choices the model is deciding between at each token.

Two observations about this objective carry most of the weight in what follows.

**It is the previous post's teacher forcing with the encoder removed.** We already built batches where `tgt_in` and `tgt_out` were the same sequence offset by one position. A causal language model is that idea with nothing else in the picture: `inputs = tokens[:-1]`, `labels = tokens[1:]`, one tensor, shifted. The causal mask is what makes it legal to train all $T$ positions in a single parallel forward pass instead of $T$ separate ones — every position sees only its own prefix, so every position is a valid training example simultaneously.

**It needs no labels.** The supervision is the text itself: the label for position $t$ is the token at position $t+1$, which we already have. This is why pretraining scales. There is no annotation budget, no labeling guideline, no inter-annotator agreement — there is only how much text you can collect and how much compute you can spend on it.

### The code delta

Here is the previous post's `DecoderBlock`, with everything the transition deletes marked:

```python
class CausalBlock(nn.Module):
    """A decoder block with the cross-attention sublayer removed."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout_p: float = 0.0, activation: str = "gelu", bias: bool = True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads,
                                            dropout_p=dropout_p, bias=bias)
        # self.cross_attn = MultiHeadAttention(...)   <- deleted
        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)          <- deleted (preceded cross-attention)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            nn.GELU() if activation.lower() == "gelu" else nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=bias),
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(
        self,
        x: torch.Tensor,                      # (B, T, d_model)
        mask: Optional[torch.Tensor] = None,  # broadcastable to (B, H, T, T)
    ) -> torch.Tensor:
        # ---- Self-attention sublayer (pre-norm, causally masked) ----
        x = x + self.dropout(self.self_attn(x_q=self.norm1(x), mask=mask))

        # ---- Cross-attention sublayer: deleted. There is nothing to attend to. ----

        # ---- Feed-forward sublayer (pre-norm) ----
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x
```

Three sublayers have become two. And now the punchline, which is worth stating as plainly as possible:

**`CausalBlock` is identical to the previous post's `EncoderBlock`.** Compare them line by line — same self-attention sublayer, same feed-forward sublayer, same pre-norm residuals, same shapes. The only difference between an "encoder block" and a "decoder-only block" is the mask handed to it at call time. Pass a padding mask and it is an encoder block. Pass a causal mask and it is a decoder block.

This is the single most useful thing to understand about the three-architecture taxonomy. "Encoder" and "decoder" do not name two kinds of transformer block. They name two *usages* of one block, distinguished entirely by which attention pattern is permitted. BERT and GPT-1 are not built from different parts; they are the same part, masked differently and trained on different objectives.

The full model follows, again as a delta against the previous post's `Transformer`:

```python
class CausalLM(nn.Module):
    """Decoder-only autoregressive language model."""

    def __init__(self, vocab_size: int, d_model: int = 128, num_heads: int = 4,
                 d_ff: int = 512, num_layers: int = 2, dropout_p: float = 0.1,
                 max_len: int = 512, tie_weights: bool = True):
        super().__init__()
        self.d_model = d_model

        # One embedding table, not two: there is no longer a source and a target.
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model, max_len=max_len, dropout_p=dropout_p)

        # One stack, not two.
        self.blocks = nn.ModuleList([
            CausalBlock(d_model, num_heads, d_ff, dropout_p=dropout_p)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)     # final pre-norm stack normalization
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Match the initialization that the sqrt(d_model) scaling in forward() assumes.
        nn.init.normal_(self.embed.weight, mean=0.0, std=d_model ** -0.5)
        with torch.no_grad():
            self.embed.weight[PAD].zero_()

        if tie_weights:
            self.lm_head.weight = self.embed.weight   # GPT-1's softmax(h_n W_e^T)

    def forward(
        self,
        idx: torch.Tensor,                    # (B, T) token ids
        mask: Optional[torch.Tensor] = None,  # broadcastable to (B, H, T, T)
    ) -> torch.Tensor:
        T = idx.size(1)
        if mask is None:
            mask = causal_mask(T, device=idx.device)
        x = self.pos(self.embed(idx) * math.sqrt(self.d_model))
        for block in self.blocks:
            x = block(x, mask=mask)
        return self.lm_head(self.norm(x))     # (B, T, vocab_size)
```

A few things to notice.

**The causal mask is now unconditional.** In the previous post, `causal_mask` was one of four mask arguments and applied only to the decoder's self-attention. Here it is the default and there is no sensible way to turn it off: a language model without it would be trained to predict tokens it can already see, which is not a task. When sequences are padded, the two masks still combine exactly as before, `causal_mask(T) & padding_mask(lengths, T)`, with the same caveat that a fully-masked row produces `NaN` out of the softmax.

**Weight tying is a choice, not a law.** Setting `lm_head.weight = embed.weight` is what GPT-1's $\text{softmax}(h_n W_e^\top)$ specifies, and it is well-motivated — both matrices are maps between token identity and model space — as well as cheap, since it saves $V \cdot d_{\text{model}}$ parameters. The argument is [Press & Wolf (2017)](https://arxiv.org/abs/1608.05859). But it is not universal: OLMo 2 sets `tie_word_embeddings: false` and pays for two separate tables, which at its vocabulary size is a substantial fraction of the model. We will see exactly how substantial later.

The loss, with the shift written out:

```python
logits = model(tokens[:, :-1])               # (B, T-1, vocab_size)
loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),     # (B*(T-1), vocab_size)
    tokens[:, 1:].reshape(-1),               # (B*(T-1),)
    ignore_index=PAD,
)
```

That is the entire training objective of a pretrained language model. Everything else — the trillions of tokens, the thousands of GPUs, the months of wall-clock time — is this loss, computed over more data.

### The other half of GPT-1: adaptation

The architectural change is the half of GPT-1 that everyone remembers. The other half is the half this post is actually about. Having pretrained a language model, GPT-1 then *adapts* it to specific tasks, and the paper is organized around that two-stage structure:

> Our training procedure consists of two stages. The first stage is learning a high-capacity language model on a large corpus of text. This is followed by a fine-tuning stage, where we adapt the model to a discriminative task with labeled data.
>
> — §3

Fine-tuning takes a labeled dataset $\mathcal{C}$ of inputs $x_1, \dots, x_m$ with labels $y$, runs them through the pretrained stack, and feeds the final block's activation $h_l^m$ into an added linear layer $W_y$:

$$
P(y \mid x_1, \dots, x_m) = \text{softmax}(h_l^m W_y), \qquad L_2(\mathcal{C}) = \sum_{(x,y)} \log P(y \mid x_1, \dots, x_m)
$$

and — a detail worth keeping, because it recurs throughout post-training in different clothes — it keeps the language modeling objective around as an auxiliary term:

$$
L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda * L_1(\mathcal{C})
$$

with $\lambda = 0.5$. The paper found this "helped learning by (a) improving generalization of the supervised model, and (b) accelerating convergence." The pattern is general: when you adapt a pretrained model, continuing to train on the original objective keeps it from drifting away from what pretraining gave it. Every later post-training method has some version of this anchor, whether an auxiliary loss, a KL penalty against the reference model, or simply a very small learning rate.

The second adaptation mechanism matters even more for where this post is going. GPT-1 has one architecture but twelve tasks, some of which have structured inputs — sentence pairs, or document-question-answer triples. Rather than build a task-specific head for each, the paper serializes the structure into a token sequence:

> Instead, we use a traversal-style approach, where we convert structured inputs into an ordered sequence that our pre-trained model can process. These input transformations allow us to avoid making extensive changes to the architecture across tasks. [...] All transformations include adding randomly initialized start and end tokens.
>
> — §3.3

So entailment becomes premise, delimiter `$`, hypothesis. Question answering becomes `[z; q; $; a_k]` for each candidate answer $a_k$. The structure of the task is expressed in the token stream, using special tokens that the model learns embeddings for during fine-tuning.

That idea is the direct ancestor of the chat template, and the line from one to the other is short. GPT-1 serializes "here is a premise and a hypothesis" with a `$`. A modern instruction-tuned model serializes "here is a user turn and here is where your reply goes" with `<|user|>` and `<|assistant|>`. Same mechanism, same reason: the model consumes a flat sequence of tokens, so any structure you want it to respect has to be written into that sequence.

### What GPT-1 actually was

Because the paper is specific, and because the numbers are a useful calibration against both the previous post's toy model and the OLMo 2 model we will download:

| | GPT-1 (2018) |
| --- | --- |
| Layers | 12, decoder-only, masked self-attention |
| $d_{\text{model}}$ | 768 |
| Attention heads | 12 |
| FFN inner dimension | 3072 |
| Activation | GELU |
| Positional encoding | Learned |
| Context length | 512 tokens |
| Tokenizer | BPE, 40,000 merges |
| Dropout | 0.1 (residual, embedding, attention) |
| Initialization | $\mathcal{N}(0, 0.02)$ |
| Optimizer | Adam, max LR 2.5e-4, linear warmup over 2000 updates, cosine anneal to 0 |
| Batch | 64 sequences of 512 tokens |
| Training | 100 epochs over BooksCorpus (7,000+ unpublished books) |
| Fine-tuning | 3 epochs, LR 6.25e-5, batch 32, $\lambda = 0.5$ |

The headline result was that this "general task-agnostic model" beat architectures purpose-built for each task, "significantly improving upon the state of the art in 9 out of the 12 tasks studied." For a 2018 reader the surprise was not the architecture, it was that one pretrained model with a linear layer bolted on could outperform a decade of task-specific engineering.

Also worth noting: GPT-1 chose BooksCorpus over the 1B Word Benchmark, which is about the same size, specifically because the Benchmark is "shuffled at a sentence level - destroying long-range structure." The paper wanted "long stretches of contiguous text, which allows the generative model to learn to condition on long-range information." Data decisions of exactly this shape — what to keep, in what order, at what granularity — turn out to dominate everything else once scale arrives.

---

## Training the Thing We Just Built

> **Status: this experiment has not been run yet.** The numbers below are `TODO`.

Before scaling up, it is worth confirming that `CausalLM` actually works, on a task we already know an encoder-decoder transformer can solve. That gives a controlled comparison: same task, same rough parameter budget, one architecture against the other.

### The toy task, reformatted as one stream

The previous post's reversal task had a source sequence and a target sequence, which is exactly the structure a decoder-only model does not have. So we flatten it. Instead of a `(src, tgt)` pair, each training example becomes a single token stream with a separator in the middle:

```
BOS 3 11 7 4 SEP 4 7 11 3 EOS
```

The model is trained on the whole stream with the ordinary next-token loss, and at inference we supply everything through `SEP` and let it generate the rest.

This reformatting is the entire conceptual content of the architectural transition, in a form small enough to hold in your head. Nothing about the task changed. What changed is that the "input" is now a prefix rather than a separate tensor, and the boundary between input and output is a token rather than a dataflow edge. That `SEP` token is a chat template with one field — and when we get to `<|user|>` and `<|assistant|>` later, they will be doing precisely this job.

**TODO(run):** train it, and report parameter count, loss curve, and exact-match accuracy against the previous post's 256/256. Report whether the decoder-only model matches the encoder-decoder at comparable size, and if it does not, say so. Do not guess this — the decoder-only model sees the source through a causal mask rather than a bidirectional encoder, which is a real handicap on a task that requires looking at the whole input before producing the first output token, and it may well need more capacity or more steps to compensate.

**TODO(figure):** loss curves for the two architectures on the same task, or a side-by-side accuracy bar.

### Generation, sampling, and the KV cache

The previous post generated with a greedy loop and explicitly deferred the KV cache. Time to collect on that, because every experiment in the rest of the post depends on generation being fast enough to iterate on.

**The cache.** Each decoding step calls the model on the entire prefix generated so far, so step $t$ recomputes the keys and values for positions $1 \dots t-1$ — which have not changed, because causal masking means no position's representation depends on anything to its right. Caching $K$ and $V$ per layer and computing only the new row turns each step from $O(t)$ work into $O(1)$ projections plus an $O(t)$ attention. No mathematics changes; it is pure memoization of values that were already determined.

```python
# TODO(code): minimal per-layer K/V cache threaded through CausalLM.generate, ~20 lines.
# TODO(measure): tokens/sec with and without, on the toy model. Small table.
```

**Sampling.** Greedy decoding was fine for the reversal task, where exactly one output is correct, and it is wrong for open-ended text, where it produces loops and bland repetition. The three standard knobs:

- **Temperature** $T$ divides the logits before the softmax: $p_i \propto \exp(z_i / T)$. $T < 1$ sharpens the distribution toward the mode, $T > 1$ flattens it, $T \to 0$ recovers greedy decoding.
- **Top-$k$** keeps the $k$ highest-probability tokens, zeroes the rest, and renormalizes.
- **Top-$p$** (nucleus) keeps the smallest set of tokens whose cumulative probability exceeds $p$, then renormalizes. Unlike top-$k$ it adapts: a confident distribution keeps few candidates, a flat one keeps many.

**TODO(figure or table):** one distribution reshaped by each of the three, so the difference is visible rather than asserted.

This matters more than it looks for what follows, because *how you sample is a confound in every base-versus-tuned comparison*. A base model sampled at $T = 1.0$ and an instruction-tuned model sampled greedily are not being compared on instruction tuning. Fix the parameters once, state them, and reuse them everywhere.

---

## The Task Is in the Text

GPT-1 pretrains, then fine-tunes. GPT-2, a year later, asks whether the second step is necessary at all — and the argument it makes is the reason the phrase "base model" means something.

### Tasks as conditional distributions

Start where GPT-2 starts. A single task can be written as estimating $p(\text{output} \mid \text{input})$. But a system that performs many tasks has to condition on which task it is being asked to do:

> Learning to perform a single task can be expressed in a probabilistic framework as estimating a conditional distribution $p(\textit{output}|\textit{input})$. Since a general system should be able to perform many different tasks, even for the same input, it should condition not only on the input but also on the task to be performed. That is, it should model $p(\textit{output}|\textit{input}, \textit{task})$.
>
> — *Language Models are Unsupervised Multitask Learners*, §2

Historically, the `task` conditioning was architectural: task-specific encoders and decoders, or a task embedding, or — as in GPT-1 — a fresh output head $W_y$ and a set of delimiter tokens per task. GPT-2's move is to put the task conditioning *in the text*, where it costs nothing:

> But as exemplified in McCann et al. (2018), language provides a flexible way to specify tasks, inputs, and outputs all as a sequence of symbols. For example, a translation training example can be written as the sequence (`translate to french`, *english text*, *french text*).

Once the task description, the input, and the output are all just tokens in one stream, something slightly startling follows:

> Since the supervised objective is the the same as the unsupervised objective but only evaluated on a subset of the sequence, the global minimum of the unsupervised objective is also the global minimum of the supervised objective.

Read that twice, because it is the load-bearing sentence of the paper and it is easy to skim past. Supervised learning on `(translate to french, english, french)` triples means maximizing the probability of the French tokens given the rest. Unsupervised next-token prediction over the same text means maximizing the probability of *all* the tokens, French ones included. The supervised objective is a sub-sum of the unsupervised one. A model that perfectly minimizes next-token loss over a corpus containing translations has, as a consequence and not as a separate goal, perfectly learned to translate.

The practical question is therefore not whether next-token prediction *can* express a task, but whether we can optimize it well enough in practice — and whether the corpus happens to contain enough demonstrations. GPT-2's bet is that the open web does:

> Our speculation is that a language model with sufficient capacity will begin to learn to infer and perform the tasks demonstrated in natural language sequences in order to better predict them, regardless of their method of procurement. If a language model is able to do this it will be, in effect, performing unsupervised multitask learning.

The paper backs this up with a wonderful table of naturally-occurring translation pairs it found in its training data — people quoting a French phrase and then glossing it in English, in the middle of ordinary web prose. Nobody put those there to teach a model French. They are there because that is how people write, and a next-token predictor has no way to benefit from them except by learning some French.

### WebText, and why the data section is the architecture section

If the task demonstrations have to come from the corpus, the corpus is the design. GPT-2's is built on a proxy for human curation:

> [...] we scraped all outbound links from Reddit, a social media platform, which received at least 3 karma. This can be thought of as a heuristic indicator for whether other users found the link interesting, educational, or just funny.

The result, WebText, is the text of 45 million links, which after de-duplication and heuristic cleaning came to "slightly over 8 million documents for a total of 40 GB of text." Wikipedia was removed deliberately, because it is a common source for evaluation datasets and its presence would muddy the zero-shot claims. Note the contrast with GPT-1's 7,000 books: one generation apart, and the corpus went from a curated genre collection to a filtered slice of the internet.

### Byte-level BPE

GPT-2 also fixes a problem that the previous post's toy vocabulary of 12 symbols let us ignore entirely: what *is* a token?

The two obvious answers both fail. Characters give a tiny vocabulary but very long sequences, and attention is quadratic in sequence length. Words give short sequences but an unbounded vocabulary, and every unseen word becomes `<UNK>` — which means the model cannot assign a probability to arbitrary text at all. [Byte-Pair Encoding](https://arxiv.org/abs/1508.07909) (Sennrich et al., 2016) interpolates: start from single characters, repeatedly merge the most frequent adjacent pair, and stop after a fixed number of merges. Frequent words end up as single tokens; rare words fragment into pieces.

GPT-2's refinement is to run BPE over *bytes* rather than Unicode code points. The motivation is coverage: a code-point-level implementation "would result in a base vocabulary of over 130,000 before any multi-symbol tokens are added," whereas "a byte-level version of BPE only requires a base vocabulary of size 256." Since every string is a byte sequence, the model can assign a probability to any Unicode string whatsoever — no `<UNK>`, no preprocessing, no lossy normalization. That is what lets GPT-2 be evaluated on any benchmark regardless of how it was tokenized, which the paper leans on heavily.

One wrinkle, and it is the source of a practical annoyance every reader will eventually hit. Naive byte-level BPE produces separate tokens for `dog`, `dog.`, `dog!`, and `dog?`, wasting vocabulary slots on punctuation variants. GPT-2 prevents merges across character categories, "with an exception for spaces which significantly improves the compression efficiency." That exception is why, in essentially every modern tokenizer, leading whitespace is *part of the token*: `" the"` and `"the"` are different tokens with different ids. Tokens are not words, and the mismatch will bite you when you count them, truncate on them, or try to force a model to produce an exact string.

The final vocabulary is 50,257 tokens.

### What changed in the model

Architecturally, GPT-2 is GPT-1 with four changes, and the paper lists them in a single compact paragraph:

> Layer normalization was moved to the input of each sub-block, similar to a pre-activation residual network and an additional layer normalization was added after the final self-attention block. A modified initialization which accounts for the accumulation on the residual path with model depth is used. We scale the weights of residual layers at initialization by a factor of $1/\sqrt{N}$ where $N$ is the number of residual layers. The vocabulary is expanded to 50,257. We also increase the context size from 512 to 1024 tokens and a larger batchsize of 512 is used.
>
> — §2.3

The first two of those are things the previous post already built, which is worth pausing on. "Layer normalization moved to the input of each sub-block" is pre-norm — `x + sublayer(norm(x))` rather than `norm(x + sublayer(x))` — which the previous post adopted and justified: it preserves an exact identity path through the residual stream and makes deep stacks trainable without heroic learning-rate schedules. And "an additional layer normalization after the final self-attention block" is the `self.norm` at the end of `CausalLM`, which exists precisely because pre-norm leaves the last residual addition un-normalized. GPT-2 is where both of these entered this lineage; every model in the rest of this post has them.

The $1/\sqrt{N}$ residual scaling addresses a quieter problem. Each block adds its sublayer output into the residual stream, so after $N$ residual layers the variance of what is flowing down that stream has accumulated $N$ contributions. Scaling the residual-branch weights down by $1/\sqrt{N}$ at initialization keeps that sum at roughly unit scale regardless of depth. It is the same genre of reasoning as the $\sqrt{d_k}$ in attention and the $\sqrt{d_{\text{model}}}$ embedding multiply: track what happens to the variance, and insert the constant that cancels it.

GPT-2 then trains four models, which is the first appearance in this lineage of scale as the experimental variable rather than the architecture:

| Parameters | Layers | $d_{\text{model}}$ |
| --- | --- | --- |
| 117M | 12 | 768 |
| 345M | 24 | 1024 |
| 762M | 36 | 1280 |
| 1542M | 48 | 1600 |

The smallest is GPT-1's size. The largest, the one called GPT-2, is "over an order of magnitude more parameters than GPT." And the paper is candid that it has not finished: "All models still underfit WebText and held-out perplexity has as of yet improved given more training time." Read in hindsight, that sentence is the whole scaling-laws research program waiting to happen.

### The result, and the honest version of it

Zero-shot — no fine-tuning, no gradient updates, no task-specific head, nothing but a prompt — GPT-2 improved "the state of the art on 7 out of the 8 datasets" of language modeling it was evaluated on. On the broader task suite the paper is careful not to overclaim: "We achieve promising, competitive, and state of the art results depending on the task."

That phrasing is the correct summary of a base model, and it is the frame this post carries forward into the experiments. A pretrained next-token predictor is not a model that lacks capabilities and needs them installed. It is a model whose capabilities are *unevenly accessible*, and how you phrase the input determines how much of them you get. GPT-2 demonstrated that the prompt is a control surface. Everything we call post-training is an attempt to make that surface less sensitive — to make the model behave well on a plainly-phrased request, instead of only on a cleverly-constructed one.

Two specific gaps are worth naming now, because they are what the experiments later measure:

- **The request frame.** A base model's prior over its context is "this is a document." Given `What is the capital of France?`, a perfectly accurate document model might continue with a second quiz question, because that is what follows the first question in a document that contains one. That is not a knowledge failure. It is a correct prediction about documents.
- **Stopping.** Nothing in next-token prediction over contiguous web text teaches a model that a response has an end. Documents keep going. A base model will too.

---

## A Model Built the Way Models Are Built Now

GPT-1 established the architecture and GPT-2 established what it can do. Seven years of refinement later, a current decoder-only model is still recognizably the same object — but every component has been replaced with a cheaper or more stable variant. This section reads one specific model closely enough that the next one you open holds no surprises.

### Why OLMo 2

Most "open" models release weights and nothing else. [OLMo 2](https://arxiv.org/abs/2501.00656) releases the data, the training code, the evaluation harness, the intermediate checkpoints, and a written account of what went wrong — under Apache 2.0. For a post whose point is that you can see all the way down, that is decisive.

It also has a property that makes the experiments later in this post possible. Ai2 publishes the base model *and* each post-training stage applied to it, as separate checkpoints of the same model:

| Stage | Checkpoint |
| --- | --- |
| Base (pretrained) | [`allenai/OLMo-2-0425-1B`](https://huggingface.co/allenai/OLMo-2-0425-1B) |
| + supervised fine-tuning | [`allenai/OLMo-2-0425-1B-SFT`](https://huggingface.co/allenai/OLMo-2-0425-1B-SFT) |
| + preference optimization | [`allenai/OLMo-2-0425-1B-DPO`](https://huggingface.co/allenai/OLMo-2-0425-1B-DPO) |
| + RL with verifiable rewards | [`allenai/OLMo-2-0425-1B-Instruct`](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct) |

So when we fine-tune the base model ourselves, we are not comparing against a different model family. We are comparing against the same weights, post-trained properly, with a published recipe. That makes the comparison informative in both directions: it shows what our small experiment achieves, and it shows what it does not.

A caveat on sourcing. The OLMo 2 paper describes the 7B, 13B, and 32B models; the 1B arrived in April 2025, after publication. The architecture and recipe below come from the paper, but the specific numbers come from the 1B's own `config.json` and model card, and I have noted where the 1B departs from what the paper describes.

| | OLMo 2 1B |
| --- | --- |
| Total parameters | 1,484,916,736 |
| Layers | 16 |
| $d_{\text{model}}$ (`hidden_size`) | 2048 |
| Attention heads | 16 |
| Key/value heads | 16 (no grouped-query attention) |
| FFN inner dimension (`intermediate_size`) | 8192 |
| Activation | SiLU, in a gated (SwiGLU) feed-forward network |
| Normalization | RMSNorm, $\epsilon = 10^{-6}$, applied to sublayer *outputs* |
| Positional encoding | RoPE, $\theta = 500{,}000$ |
| Context length | 4096 |
| Vocabulary | 100,352 |
| Biases | None |
| Weight tying | No (`tie_word_embeddings: false`) |
| Pretraining | ~4T tokens of [OLMo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124) (1 epoch), then 50B tokens of [Dolmino-mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124) mid-training |
| License | Apache 2.0 |

### Reading the config

The whole configuration file is twenty-five lines, and at this point in the post nearly every line maps onto something we have already built:

```json
{
  "architectures": ["Olmo2ForCausalLM"],
  "attention_bias": false,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "intermediate_size": 8192,
  "max_position_embeddings": 4096,
  "num_attention_heads": 16,
  "num_hidden_layers": 16,
  "num_key_value_heads": 16,
  "rms_norm_eps": 1e-06,
  "rope_theta": 500000,
  "tie_word_embeddings": false,
  "vocab_size": 100352
}
```

| `config.json` | `CausalLM` |
| --- | --- |
| `hidden_size` | `d_model` |
| `num_hidden_layers` | `num_layers` |
| `num_attention_heads` | `num_heads` |
| `intermediate_size` | `d_ff` |
| `max_position_embeddings` | `max_len` |
| `vocab_size` | `vocab_size` |
| `rms_norm_eps`, `rope_theta` | no counterpart — see below |

`num_key_value_heads` equals `num_attention_heads`, so this model does *not* use grouped-query attention; every query head has its own key and value head, exactly as in the previous post's `MultiHeadAttention`. (GQA is near-universal at larger scales, where the KV cache dominates inference memory. At 1B with 16 layers it buys little.) And `tie_word_embeddings: false` means the input embedding and the output head are two separate matrices — a deliberate departure from GPT-1's $\text{softmax}(h_n W_e^\top)$.

That last decision is expensive enough to be worth tracing through, because it also explains the model's name. With $V = 100{,}352$ and $d = 2048$, each embedding table holds $V \cdot d = 205{,}520{,}896$ parameters, and there are two of them. Per block:

- attention: four $d \times d$ projections, $4 \cdot 2048^2 = 16{,}777{,}216$
- the gated feed-forward network: three $d \times d_{ff}$ matrices, $3 \cdot 2048 \cdot 8192 = 50{,}331{,}648$
- four RMSNorm gain vectors (queries, keys, post-attention, post-feed-forward): $4 \cdot 2048 = 8{,}192$

which totals $67{,}117{,}056$ per block, or $1{,}073{,}872{,}896$ across sixteen blocks. Add the two embedding tables and the final norm:

$$
2(205{,}520{,}896) + 1{,}073{,}872{,}896 + 2048 = 1{,}484{,}916{,}736
$$

That is exactly the parameter count the Hugging Face repository reports, which is a satisfying confirmation that nothing is hiding in the model we have not accounted for. It also shows what "1B" names: the transformer blocks hold 1.07B parameters, and the vocabulary tables add another 411M on top — 28% of the model, spent on getting tokens in and out of it.

### What changed since GPT-2

The paper summarizes the lineage from the original OLMo through OLMo-0424 to OLMo 2. Pulling together the pieces that distinguish OLMo 2 from the model we built:

**No biases.** "We exclude all bias terms from our architecture." Every `nn.Linear` in the previous post defaulted to `bias=True`; modern models set it false throughout. The biases cost parameters, and at scale they do not earn them back.

**RoPE instead of additive position embeddings.** The previous post added a fixed sinusoidal signal to the embeddings before the first block, and noted that the angle-addition identities make $\text{PE}_{t+k}$ a linear function of $\text{PE}_t$ — the property the 2017 paper hypothesized would let a model attend by relative position. [Rotary position embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021) stops hypothesizing and builds that property in directly.

Instead of adding anything to the embeddings, RoPE *rotates* the query and key vectors inside attention. Split a $d$-dimensional query into $d/2$ coordinate pairs. For a token at position $m$, rotate pair $i$ by the angle $m\theta_i$:

$$
R_m^{(i)} =
\begin{pmatrix}
\cos m\theta_i & -\sin m\theta_i \\
\sin m\theta_i & \cos m\theta_i
\end{pmatrix},
\qquad
\theta_i = \theta^{-2(i-1)/d}
$$

with the base $\theta$ a hyperparameter — the `rope_theta` in the config. Because rotations compose and $R_m^\top R_n = R_{n-m}$, the attention score between a query at position $m$ and a key at position $n$ becomes

$$
(R_m q)^\top (R_n k) = q^\top R_m^\top R_n k = q^\top R_{n-m}\, k
$$

The score depends on $n - m$ and not on $m$ or $n$ separately. Relative position is not learned and not approximated; it is an algebraic identity of the encoding. The practical consequences are that there is no positional parameter table to run off the end of, and that behavior degrades more gracefully past the training context length.

The choice of base matters. OLMo 2 raises it from the conventional 10,000 to 500,000, matching Llama 3, which "increases the resolution of positional encoding" — a larger base stretches the wavelengths of the low-frequency pairs, so distant positions remain distinguishable rather than wrapping into near-identical angles. It is one of the standard moves for supporting longer contexts.

**RMSNorm instead of LayerNorm.** LayerNorm centers and rescales, then applies a learned gain and shift. [RMSNorm](https://arxiv.org/abs/1910.07467) (Zhang & Sennrich, 2019) drops the centering and the shift and keeps the rescaling:

$$
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}
$$

No mean subtraction, no $\beta$. It is cheaper, and empirically no worse — which is the whole argument. OLMo 2's ablations found "no difference between the two," so they chose the cheaper one. ($\epsilon$ is the config's `rms_norm_eps`, $10^{-6}$.)

**Normalization moved to the outputs.** This is the change most likely to trip up a reader comparing code against the previous post, because it reverses a decision we justified at length. OLMo 2 applies the norm to each sublayer's *output*, inside the residual branch:

| OLMo-0424 (pre-norm) | OLMo 2 (reordered norm) |
| --- | --- |
| $h := x + \text{Attention}(\text{LN}(x))$ | $h := x + \text{RMSNorm}(\text{Attention}(x))$ |
| $h_{\text{out}} := h + \text{MLP}(\text{LN}(h))$ | $h_{\text{out}} := h + \text{RMSNorm}(\text{MLP}(h))$ |

This is not a return to the original post-norm formulation, which normalized *after* the residual addition and broke the identity path. The residual stream here is still untouched — $x$ flows through the addition unmodified — so the gradient path that pre-norm was adopted to protect is intact. What is normalized is only the correction the block contributes. The motivation is stability: together with QK-norm, the paper reports the pair "improve both the growth and the spikiness of the L2 norm of the gradient," while "in isolation, neither of these changes yield good results." The idea comes from Liu et al. (2021) by way of Chameleon.

(A note for anyone reading the paper alongside this: equation (2) in §2.1 prints $h_{\text{out}} := h + \text{RMSNorm}(\text{MLP}(x))$, with $x$ where $h$ belongs. The table in §3.3.2 has it right, and so does the implementation.)

**QK-norm.** An extra RMSNorm applied to the query and key projections before attention scores are computed: "This avoids attention logits being too large, which can lead to training loss divergence." This is the same failure mode the previous post discussed when motivating the $\sqrt{d_k}$ denominator — large dot products drive softmax into saturation, where gradients vanish — except that at scale the constant denominator is not enough on its own, because the *learned* projections can grow. Normalizing $q$ and $k$ bounds the problem at the source. These are the `q_norm` and `k_norm` gain vectors in the parameter count above.

**SwiGLU feed-forward networks.** The previous post's FFN was two matrices with a GELU between them. A [gated variant](https://arxiv.org/abs/2002.05202) (Shazeer, 2020) uses three:

$$
\text{SwiGLU}(x) = \big(\text{SiLU}(x W_{\text{gate}}) \odot x W_{\text{up}}\big) W_{\text{down}},
\qquad \text{SiLU}(z) = z \cdot \sigma(z)
$$

One branch computes a value, the other computes a multiplicative gate, and their elementwise product is projected back down. Since three matrices cost more than two at equal width, the paper compensates by narrowing: it sets the inner dimension to "approximately $\frac{8}{3}d$, but increased to the closest multiple of 128," which is where the 7B model's distinctive `intermediate_size` of 11,008 comes from. The 1B does not follow that rule — its 8192 is a full $4d$, so its feed-forward networks are proportionally wider than its larger siblings'. This is the kind of thing you only learn by reading the config rather than the paper.

**Z-loss.** A regularization term added to the training loss, "$10^{-4} \cdot \log^2 Z$ [...] where $Z$ is the denominator in the softmax over the logits." It penalizes the softmax normalizer for drifting away from 1, which keeps the output logits from growing large. Like QK-norm and the reordered norm, it buys stability rather than accuracy. The paper's §3 is largely a catalogue of changes of this kind, which is a useful corrective to the impression that architecture research at this scale is about capability: most of it is about getting a very long run to finish without diverging.

### Tokenization in practice

OLMo 2 uses a 100,352-token byte-level BPE vocabulary — double GPT-2's, which is typical, driven by code and multilingual text. The interface is the same one GPT-2 introduced, and the behavior it is most important to see for yourself is that token boundaries are not word boundaries:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")

ids = tok("Tokens are not words.")["input_ids"]
print(ids)
print([tok.decode([i]) for i in ids])
```

**TODO(run):** paste real output. Make the leading-space point concrete by also printing `tok("the")["input_ids"]` against `tok(" the")["input_ids"]`.

Two numbers from the config are worth remembering because they will matter when we build training batches: the end-of-text token is id 100257 and the padding token is id 100277. The model's `eos_token` and `bos_token` are the *same* string, `<|endoftext|>` — a document separator doing double duty as both boundary markers.

---

## What the Base Model Can and Cannot Do

> **Status: these experiments have not been run yet.** Every number and every quoted
> generation in this section is a `TODO`. Nothing here is invented or guessed.

There is one more fact about the base checkpoint that is more telling than anything in its config file. Its `tokenizer_config.json` contains:

```python
chat_template = None
```

The SFT checkpoint has one. The base checkpoint does not. There is no notion of a user, an assistant, or a turn anywhere in the base model's interface, because nothing in pretraining put one there. It is a document continuation engine, and the only thing you can hand it is a prefix.

The experiments below are designed to find out what that is and is not good for. Sampling is a confound in every base-versus-tuned comparison, so the parameters are fixed once here and reused for every generation in the rest of the post: **TODO(decide)** temperature, top-$p$, `max_new_tokens`, and seed.

### It is a very good next-token predictor

- **Experiment A — prose continuation.** A few sentences of prose as prefix. Expect fluent, on-topic, grammatical text that does not stop on its own. **TODO(run)** — show the output untruncated, including the run-on.
- **Experiment B — code completion.** A Python signature and docstring. This is chosen as a task the base model should do *well*, because code completion is continuation. **TODO(run)** — and check whether it continues past the requested function into a second, unrequested one. That would be correct document behavior and unhelpful assistant behavior at the same time.
- **Experiment C — few-shot prompting.** Three worked examples of a mapping, then a fourth input, as in GPT-2 §2. **TODO(run).** This is the experiment that makes the capability claim concrete: no gradients, no fine-tuning, pattern continued.
- **Experiment D — perplexity on held-out text.** A number rather than a vibe, and a baseline for the evaluation section. **TODO(run).**

### Where it is unreliable

- **Experiment E — a bare instruction.** `Write a haiku about recursion.` with no scaffold. **TODO(run).** A 1B base model's behavior here varies across seeds, and *that variability is the finding* — run N seeds and report the distribution rather than cherry-picking one failure.
- **Experiment F — the same instruction, scaffolded.** Wrapped as a document the corpus plausibly contains: `Q: ... \nA:`, or a few-shot instruction prompt. **TODO(run).** The contrast between E and F is the most important result in the post, and the one GPT-2 predicts: *the capability did not change, the elicitation did.*
- **Experiment G — stopping.** What fraction of generations emit `<|endoftext|>` before `max_new_tokens`? **TODO(run).** Countable, and the failure mode instruction tuning most obviously fixes.
- **Experiment H — format compliance.** Ask for JSON, or exactly three bullets, or a one-word answer; score mechanically over N prompts. **TODO(run).** This is [IFEval](https://arxiv.org/abs/2311.07911) in miniature, and it is what gets re-measured after tuning.

---

## Supervised Instruction Tuning

### The idea

Same model, same architecture, same loss function. The only changes are *what text we train on* and *which tokens the loss is computed over*.

That is worth saying twice, because the vocabulary around post-training ("alignment," "instruction following," "assistant behavior") suggests something more exotic is happening. It is not. SFT is the cross-entropy objective from the top of this post, run over a corpus of demonstrations instead of a corpus of web documents, with the loss restricted to the parts we want the model to produce.

What it changes is the model's prior over its own context. Pretraining left it believing its input is a document. SFT trains it on a corpus in which the input is *always* a request and the continuation is *always* a response — so the request frame stops being something you prompt your way into and becomes the default. GPT-1 already had both halves of this: the auxiliary $\lambda L_1$ term that keeps a fine-tuned model anchored to its pretraining, and the delimiter tokens that tell a flat sequence where the structure is. SFT is those ideas with the task left general.

It does not teach the model facts or skills. This is the claim the evaluation section has to hold itself to.

### The data

The shape is `(prompt, response)`, optionally multi-turn, stored as a list of role-tagged messages:

```python
{"messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
]}
```

OLMo 2 1B's SFT stage uses [`allenai/tulu-3-sft-olmo-2-mixture-0225`](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture-0225), which contains **866,138 examples**. The mixture is a deliberate construction, documented in the [Tülu 3 paper](https://arxiv.org/abs/2411.15124): it combines human-written, model-distilled, and templated-from-existing-dataset examples, with subsets targeting math, code, precise instruction following, and safety, and the mixture ratios themselves were tuned.

**We will use a few thousand.** Stating that early sets expectations correctly: the goal is to demonstrate the *direction* of the change on a hobbyist budget, not to reproduce the checkpoint. That a few thousand examples is informative at all is itself a finding from the literature — [LIMA](https://arxiv.org/abs/2305.11206) (Zhou et al., 2023) got competitive instruction-following from 1,000 carefully-curated examples, on the argument that SFT is teaching a format the model already has the capability to fill.

**TODO(run):** show two or three real examples from the mixture, as raw `messages` lists, with their `source` field.

### Chat templates

The model consumes a flat sequence of tokens. A conversation has structure. Something has to serialize one into the other, unambiguously, including marking where each turn ends — which is precisely the problem GPT-1 solved for entailment pairs with a `$`.

The modern mechanism is special tokens plus a fixed layout, stored as a Jinja template in the tokenizer configuration. Here is OLMo 2's, from the SFT checkpoint's `tokenizer_config.json`, reformatted for legibility:

{% raw %}
```jinja
{{ bos_token }}
{%- for message in messages %}
  {%- if message['role'] == 'system' %}    {{ '<|system|>\n' + message['content'] + '\n' }}
  {%- elif message['role'] == 'user' %}    {{ '<|user|>\n'   + message['content'] + '\n' }}
  {%- elif message['role'] == 'assistant' %}
    {%- if not loop.last %}                {{ '<|assistant|>\n' + message['content'] + eos_token + '\n' }}
    {%- else %}                            {{ '<|assistant|>\n' + message['content'] + eos_token }}
    {%- endif %}
  {%- endif %}
  {%- if loop.last and add_generation_prompt %}
                                           {{ '<|assistant|>\n' }}
  {%- endif %}
{%- endfor %}
```
{% endraw %}

Applying it to a single-turn training example produces exactly this string:

```
<|endoftext|><|user|>
What is the capital of France?
<|assistant|>
Paris.<|endoftext|>
```

and at inference time, with `add_generation_prompt=True` and no assistant message supplied:

```
<|endoftext|><|user|>
What is the capital of France?
<|assistant|>
```

**TODO(verify):** confirm both strings with `repr(tok.apply_chat_template(...))` rather than by reading the Jinja. Whitespace and `<|endoftext|>` placement are exactly the kind of detail that is wrong in every blog post, including possibly this one.

Three things about this deserve emphasis.

**`<|assistant|>` is the hinge of the entire section.** After SFT, that token is reliably followed by a helpful response — because in every single training example, it was. The model has not acquired helpfulness as a disposition; it has acquired a very strong conditional distribution over what follows one particular token.

**`<|endoftext|>` at the end of each response is how the model learns to stop.** Experiment G's failure mode is not fixed by an architectural change or a decoding heuristic. It is fixed by training on 866,138 examples in which the response ends with an explicit end token, until predicting that token at the end of a response becomes the likeliest continuation. This is the clearest case in the whole post of a behavior that looks like a capability and is really a data format.

**Template mismatch between training and inference silently destroys quality.** Train with one layout and serve with another and you get a model that looks broken for no visible reason — no error, no warning, just degraded output. The only defense is to render both sides with the same `apply_chat_template` call, which is why the code below never builds these strings by hand.

### The loss, and response-token masking

The loss is the cross-entropy we started with. What changes is the label tensor.

If the loss includes prompt tokens, the model spends capacity learning to *generate user messages* — a distribution we have no use for and, when the prompts are templated or synthetic, one that is badly unrepresentative of real user input. What we want is $p_\theta(\text{response} \mid \text{prompt})$, so we score only the response tokens. Mechanically: set `labels` to $-100$ on the prompt and role tokens, and rely on `F.cross_entropy`'s default `ignore_index=-100` to drop them from the average. It is the same mechanism as the previous post's `ignore_index=PAD`, with a different motivation — there we were ignoring padding because it was meaningless, here we are ignoring real tokens because they are not ours to model.

```python
def build_example(tok, messages, max_len):
    """Render one conversation and mask all non-response tokens out of the loss."""
    input_ids, labels = [], []

    for i, msg in enumerate(messages):
        # Render the prefix through the prompt side of this turn, then the full turn,
        # so the response span is exactly the difference between the two.
        prefix = tok.apply_chat_template(messages[:i] + [msg], tokenize=True,
                                         add_generation_prompt=(msg["role"] != "assistant"))
        ...  # TODO(code): finish, then assert the spans line up

    return {"input_ids": input_ids, "labels": labels}
```

**TODO(code):** finish the collator, and include in the post the two checks that catch the off-by-one errors this code is prone to — assert that the count of unmasked labels equals the count of response tokens, and print one example decoded with masked positions replaced by `·` so the mask is visible rather than asserted.

Two smaller decisions, each worth a sentence in the final post:

- **Is masking necessary?** It is standard and well-motivated, but the measured effect depends on the data. **TODO(optional):** train one variant without it and compare. Stretch goal, only if the length budget allows.
- **Packing versus padding.** Short examples can be concatenated to fill the context window, which is far more efficient but risks cross-contamination between unrelated examples unless attention is blocked at the boundaries. We will pad and keep it simple, and say so.

---

## Fine-Tuning with LoRA

### Why not full fine-tuning

Walk the memory arithmetic, because it is a genuinely useful thing to be able to do from a parameter count. For standard mixed-precision AdamW training, each parameter needs:

| | bytes/param |
| --- | --- |
| bf16 weights | 2 |
| bf16 gradients | 2 |
| fp32 master weights | 4 |
| AdamW first moment $m$ | 4 |
| AdamW second moment $v$ | 4 |
| **total** | **16** |

At 1,484,916,736 parameters that is $16 \times 1.485 \times 10^9 \approx 23.8$ GB, before a single activation is stored. So full fine-tuning of this model does not fit on a 16 GB card and is uncomfortable on a 24 GB one.

It is worth being straight about the conclusion: full fine-tuning of a 1B model *is* perfectly feasible on rented hardware. We are choosing LoRA so that the experiment reproduces on a consumer GPU or a free Colab instance, not because the model demands it.

### LoRA

[Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) (Hu et al., 2021) freezes the pretrained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ and learns a low-rank update beside it:

$$
W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A,
\qquad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times k},\; r \ll \min(d, k)
$$

$A$ is initialized Gaussian and $B$ is initialized to zero, so $\Delta W = 0$ at step 0 and training begins at exactly the pretrained model rather than near it. The $\alpha/r$ factor decouples the effective update magnitude from the rank, so changing $r$ does not force retuning the learning rate.

The hypothesis underneath it — and it is a hypothesis, which the paper is clear about — is that the weight update needed to adapt a pretrained model to a downstream task has low intrinsic rank. There is no reason this must be true, and it is more plausible for "learn this response format" than for "learn a new language." For our purposes that is the favorable case.

Only $A$ and $B$ receive gradients, so the optimizer state shrinks in proportion. For this model, with $r = 16$:

| Target modules | Trainable parameters | Share of model |
| --- | --- | --- |
| Attention only (`q,k,v,o_proj`) | 4,194,304 | 0.28% |
| Attention + MLP (`gate,up,down_proj`) | 12,058,624 | 0.81% |

The attention-only figure is $r(d_{\text{out}} + d_{\text{in}}) = 16(2048 + 2048) = 65{,}536$ per projection, four projections per block, sixteen blocks. The MLP adds three matrices of shape $2048 \times 8192$ or its transpose, each costing $16(2048 + 8192) = 163{,}840$. **TODO(decide):** attention-only is the classic recipe; including the MLP is now common practice and is the stronger default for format learning. Pick one, state why, and confirm the number against `model.print_trainable_parameters()`.

At inference $BA$ can be folded into $W_0$, so a merged adapter adds no latency. That, more than the memory saving, is why LoRA won.

### The training run

> **Status: not yet run.** The hardware for this section is not yet set up, so every
> hyperparameter below is a decision to be made and every result is a `TODO`.

**TODO(decide):** hand-rolled PyTorch loop, or `trl`'s `SFTTrainer`? The loop is more in keeping with this series — it makes the masking visible, and it is the same loop as the previous post's with a different data source. `SFTTrainer` is what a reader would actually reach for. The plan is the loop first, with `SFTTrainer` shown afterward as the production equivalent.

```python
# Sketch only — see the companion directory for the version that runs.
base = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B",
                                            torch_dtype=torch.bfloat16)
tok  = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B-SFT")  # the one with the template

model = get_peft_model(base, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=[...],          # TODO: decide
    task_type="CAUSAL_LM",
))
model.print_trainable_parameters()  # TODO: paste real output
```

Note the detail in the second line: the chat template lives on the *SFT* tokenizer, because the base checkpoint has none. Training a base model to follow a template means importing the template from somewhere, and that somewhere needs to be the same place the evaluation imports it from.

Hyperparameters to state and justify, all **TODO**: subset size and sampling method, `max_seq_length`, $r$ and $\alpha$, learning rate (1e-4 to 2e-4 is the usual LoRA range, well above a full fine-tuning rate — explain why), batch size and gradient accumulation, epochs, warmup. **TODO(report):** wall-clock, peak VRAM, hardware, final train loss, held-out loss, and the loss curves.

**Failure modes to document as they happen** rather than cleaning up afterward, since this is the part of a tutorial that is actually scarce: OOM and what was reduced to fix it; loss not moving (usually the learning rate, or adapters not actually attached — `print_trainable_parameters()` is the check); garbage at inference from a template mismatch; forgetting `add_generation_prompt=True`.

---

## Evaluation

> **Status: not yet run.** Every number is a `TODO`.

### What we are comparing

| Model | Role |
| --- | --- |
| `allenai/OLMo-2-0425-1B` | base — the "before" |
| base + our LoRA adapter, N examples | the "after," on a hobbyist budget |
| `allenai/OLMo-2-0425-1B-SFT` | the official SFT, 866,138 examples — the ceiling for this stage |

The expected ordering, stated in advance so that reporting it is not optional: ours should land between base and official SFT on format and stopping, and may be *worse* than base on knowledge benchmarks. If it comes out otherwise, that goes in the post along with an attempt to find out why.

Controls: identical prompts, sampling parameters, and seeds across all three. The awkward case is the base model, which has no chat template — **TODO(decide and document):** how it gets prompted for each evaluation, since that choice substantially drives its score. Reporting the base model under both a bare prompt and a scaffolded one is probably the honest answer.

### Quantitative

- **Held-out loss and perplexity** on a validation split of the mixture. Cheap, and directly measures whether the response distribution was learned.
- **Instruction following, mechanically scored.** A small set of verifiable constraints — "answer in exactly one word," "output valid JSON," "use no commas" — scored by a checker function, not a judge model. **TODO(build + run).**
- **Stopping rate.** The fraction of generations emitting EOS before the cap. Experiment G re-run across all three models, and the cleanest single number in the post.
- **A capability benchmark that is not about format** — a GSM8K or MMLU subset — to test whether our fine-tune degrades what pretraining provided. **TODO(run),** reported honestly either way.

There is published evidence for what to expect from the stages beyond SFT. The Instruct model card reports averages across eleven benchmarks for each post-training stage of this exact model:

| Stage | Avg | GSM8K | IFEval | MATH | MMLU |
| --- | --- | --- | --- | --- | --- |
| OLMo 2 1B SFT | 36.9 | 52.1 | 50.5 | 13.2 | 36.4 |
| OLMo 2 1B DPO | 40.6 | 59.0 | 67.1 | 14.1 | 39.9 |
| OLMo 2 1B Instruct (RLVR) | 42.7 | 68.3 | 70.1 | 20.7 | 40.0 |

Two things to take from that table. SFT is the first step of three and the smallest of the three in effect — which is the honest framing for an experiment that stops after SFT. And the base model is absent from it, because these benchmarks are administered in a chat format the base model has no template for. That absence is itself the point of this post's first half.

### Qualitative

The same prompts from the base-model experiments, run on all three models, quoted verbatim and untruncated. **TODO(run).** This is the part readers actually read. It should include at least one case where our fine-tune is clearly worse than the official SFT, with a guess at the cause — data volume, rank, or a single epoch.

### What this does and does not show

The limits, stated plainly: tiny evaluation sets, no significance testing, one seed per cell unless budget allows more, LoRA is not full SFT, and our few thousand examples are a sample from a mixture that was itself carefully balanced.

The claim the post is entitled to make: supervised instruction tuning measurably changes *how a model responds to a request*, with a few thousand examples and an hour of GPU time. It does not make the model smarter.

---

## What Comes Next

**Why SFT is not enough.** SFT imitates demonstrations. It cannot express "this response is better than that one," gives no signal about what to avoid, and is bounded above by the quality of its data — a model trained to imitate can only learn to be as good as what it imitated. The table above puts a number on the gap.

**Preference optimization.** Collect pairs $(y_w, y_l)$ of preferred and dispreferred responses to the same prompt, and optimize the model to prefer $y_w$. [InstructGPT](https://arxiv.org/abs/2203.02155) (Ouyang et al., 2022) does this by training a reward model on the preferences and then optimizing against it with PPO; [DPO](https://arxiv.org/abs/2305.18290) (Rafailov et al., 2023) shows the reward model can be eliminated and the preferences optimized directly. `OLMo-2-0425-1B-DPO` is exactly this stage applied to the model in this post, which means a reader can run the comparison today.

**Reinforcement learning with verifiable rewards.** For math, code, and precise instruction following, correctness is *checkable*: run the tests, check the answer against ground truth, verify the constraint. That replaces a learned reward model with a verifier and removes reward hacking as a failure mode. This is the stage that produced `OLMo-2-0425-1B-Instruct`, and the idea behind the reasoning-model wave.

**Reasoning.** Long chains of thought, learned rather than prompted, and test-time compute as a scaling axis of its own.

**Tool use and agents.** Once a model reliably follows a format, that format can be a function call. The distance from "emit `<|assistant|>`, then helpful text" to "emit a tool call, read the result, continue" is smaller than it looks: it is the same chat template mechanism with another special token and a loop around the generate call.

**And the question this post deferred.** *Why* does next-token prediction over a corpus produce knowledge, arithmetic, and code in the first place? GPT-2 speculated that a model "with sufficient capacity will begin to learn to infer and perform the tasks demonstrated in natural language sequences in order to better predict them," and the seven years since have been an extended confirmation of that speculation without a complete explanation of it. That is the pretraining post.

---

## References and Further Reading

**The three papers this post follows**

1. [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (Radford, Narasimhan, Salimans, Sutskever, 2018 — GPT-1: the decoder-only turn, and pretrain-then-fine-tune)
1. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (Radford, Wu, Child, Luan, Amodei, Sutskever, 2019 — GPT-2: tasks specified in text, byte-level BPE, zero-shot transfer)
1. [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656) (OLMo Team, 2025 — a fully open modern model, its architecture, and its post-training pipeline)

**Architecture**

1. [Attention is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
1. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (Brown et al., 2020 — GPT-3, in-context learning)
1. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) (Su et al., 2021 — RoPE)
1. [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) (Zhang & Sennrich, 2019 — RMSNorm)
1. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) (Shazeer, 2020 — SwiGLU)
1. [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) (Ainslie et al., 2023)
1. [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) (Press & Wolf, 2017 — weight tying)
1. [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (Sennrich et al., 2016 — BPE)

**Scale and pretraining** *(deferred to a later post; listed for the curious)*

1. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) (Kaplan et al., 2020)
1. [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (Hoffmann et al., 2022 — Chinchilla)

**OLMo 2 artifacts**

1. [`allenai/OLMo-2-0425-1B`](https://huggingface.co/allenai/OLMo-2-0425-1B) — base model card and config
1. [`allenai/OLMo-2-0425-1B-SFT`](https://huggingface.co/allenai/OLMo-2-0425-1B-SFT) / [`-DPO`](https://huggingface.co/allenai/OLMo-2-0425-1B-DPO) / [`-Instruct`](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct)
1. [`allenai/tulu-3-sft-olmo-2-mixture-0225`](https://huggingface.co/datasets/allenai/tulu-3-sft-olmo-2-mixture-0225) — the SFT mixture
1. [OLMo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124) and [Dolmino-mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124) — pretraining and mid-training data

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

# Working Notes

> Not part of the post. Delete before publishing.

## Verification log

Everything factual in the written sections was checked against a primary source, not recalled:

- **GPT-1** — `pdftotext` of the OpenAI PDF. §3.1 objective and the three-equation forward pass, §3.2 $L_2$/$L_3$ and $\lambda = 0.5$, §3.3 input transformations, §4.1 "Model specifications" (all architecture and optimizer numbers, BooksCorpus, perplexity 18.4), abstract (9 of 12 tasks).
- **GPT-2** — `pdftotext` of the OpenAI PDF. §2 $p(\text{output}|\text{input},\text{task})$ and the "supervised objective is a subset" argument, §2.1 WebText (3 karma, 45M links, 8M documents, 40 GB, Wikipedia removed), §2.2 byte-level BPE, §2.3 the four architecture changes verbatim, Table 2 model sizes, §3.1 "state of the art on 7 out of the 8 datasets."
- **OLMo 2** — `pdftotext` of arXiv:2501.00656v3. §2.1 architecture list (no biases, SwiGLU at $\frac{8}{3}d$, RoPE $\theta=5\times10^5$, RMSNorm, reordered norm, QK-norm, z-loss), §3.3.2 the norm-placement table, §3.3.3 z-loss as $10^{-4}\log^2 Z$. Exact title is **"2 OLMo 2 Furious"**.
- **OLMo 2 1B** — live `config.json`, `tokenizer_config.json`, and the HF API parameter count.

## Outline claims that turned out to be wrong

Carried over from the outline and corrected in the prose:

1. **GQA.** The 1B has `num_key_value_heads: 16 == num_attention_heads`, so there is **no** grouped-query attention. The outline flagged this for checking; it is now stated correctly.
2. **Weight tying.** `tie_word_embeddings: false`. The outline's §2.3 discussion implied modern models tie; OLMo 2 does not, and the 411M parameters that costs is now part of the parameter walkthrough.
3. **SFT mixture size.** 866,138 examples, not ~939k.
4. **Parameter count.** "1B" is 1,484,916,736 total — 1.07B in blocks, 411M in the two embedding tables. The hand arithmetic reproduces the published figure exactly.
5. **$d_{ff}$.** The paper's $\frac{8}{3}d$ SwiGLU rule does *not* hold for the 1B, whose `intermediate_size` is $8192 = 4d$.
6. **The model card's benchmark table.** The table on the *base* model card is headed "Instruct Model" and lists instruct-tuned models; its `OLMo 2 1B` row is the Instruct checkpoint, not the base model. It cannot be used for a base-vs-post-trained comparison. The usable table is on the Instruct card, which has SFT/DPO/Instruct rows and no base row.
7. **Norm placement.** OLMo 2 is not post-norm. It normalizes the sublayer output *inside* the residual branch, leaving the residual stream clean. The paper's own eq. (2) in §2.1 has a typo ($x$ for $h$); §3.3.2 is correct.
8. **Chat template.** The base checkpoint's `chat_template` is `None`. Only the post-trained checkpoints have one — which is now a load-bearing fact in two sections rather than a footnote.

## Decisions made

- **Structure.** Three papers as the spine: GPT-1 for the architectural turn and adaptation, GPT-2 for what a base model is, OLMo 2 for a current model. This replaces the outline's thematic ordering and is what the post is organized around now.
- **Toy task** (outline open question 1): reversal reformatted as one stream, with a `SEP` token, for continuity with the previous post and because `SEP` previews the chat template. Tiny-Shakespeare dropped rather than kept as an aside — the "it generates text" demo is better served by the real OLMo base model in the experiments section.
- **Training code** (open question 2): hand-rolled loop first, `SFTTrainer` after as the production equivalent. Still worth revisiting once the loop is written.

## Still open

- **Companion code** (open question 3). Leaning `tools/olmo-sft/` in this repo, following the `tools/figures/` precedent, with a README. Not started.
- **Hardware.** No usable GPU on this machine right now — `torch` is not installed and `nvidia-smi` reports a driver/library version mismatch. Every unrun experiment is blocked on fixing that or renting a box.
- **LoRA target modules.** Attention-only vs. attention+MLP. Numbers for both are in the post; the choice is not made.
- **Length.** The written sections already run long. If the finished post needs cutting, the KV cache subsection is the most separable piece and could be its own short post.

## Figure plan

Ranked by how much they earn their generator script. Conventions in `tools/figures/README.md`; add a row there per figure.

1. **`chat-template-loss-mask.svg`** — one rendered training example as a token strip: role tokens in one color, prompt tokens greyed with `label = -100`, response tokens colored with `label = token`, and a bracket under the response reading "loss computed here." Carries the template and masking subsections at once. The figure I most want.
2. **`encoder-decoder-vs-decoder-only.svg`** — side by side. Left: the previous post's two stacks and the long cross-attention arrow. Right: one stack, cross-attention drawn as a dashed ghost, encoder greyed out. Same block vocabulary and palette as the previous post so the correspondence is immediate.
3. **`rope-rotation.svg`** — a query vector's 2-D coordinate pair rotated by $m\theta_i$ at three positions, with the relative-angle identity $(R_m q)^\top(R_n k) = q^\top R_{n-m} k$ shown as the angle between two rotated vectors. Nothing in the written RoPE subsection is visual yet, and this is the one idea in it that wants a picture.
4. **`lora-adapter.svg`** — input → frozen grey $W_0$ and, in parallel, a thin $A$ down to $r$ then $B$ back up to $d$ in the accent color, summed at the output. Dimensions on every edge; the visual point is the narrow waist at $r$.
5. **`post-training-pipeline.svg`** — `raw text → [pretraining] → base → [SFT] → instruct → [DPO] → aligned → [RLVR] → reasoning`, with SFT boxed as "this post" and pretraining greyed. Would open the post and recur at the end. Lowest priority: the prose already carries it, and the OLMo checkpoint table does the same work.
6. **`shifted-labels.svg`** — token strip drawn twice, offset by one, arrows from each input position to its label, causal mask triangle beneath. Cheap to generate but may be redundant with the loss code block.



