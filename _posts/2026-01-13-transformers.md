# Deep Learning: Transformers

This post is a detailed look at the original Transformer. We'll be working through the intuition, architecture, and a PyTorch implementation. By the end, you should be able to train and run your own transformer, and feel comfortable with each component in this diagram:

<figure style="text-align: center;">
  <img src="/assets/images/transformer-architecture.png" alt="Transformer Architecture" style="max-width: 100%; height: auto; margin: auto;">
  <figcaption>Transformer Architecture. Image credit: Efficient Transformers: A Survey</figcaption>
</figure>
 
# The Transformer

In less than a decade, transformers have become the dominant architecture in modern machine learning, spanning text embedding models for search and retrieval, large language models for generation, and models for images, audio, and protein structure. Introduced by Vaswani et al. in the 2017 paper *Attention Is All You Need*, the transformer architecture replaced recurrence and convolution with a single mechanism: attention. Earlier sequence models like RNNs and LSTMs compressed all prior information into a fixed-length hidden state, forcing the network to distill an entire sequence into a single vector. In contrast, the transformer computes a representation of each position in the sequence by letting it directly "attend to" every other position, drawing in context from the parts of the input most relevant to it.

This section steps through the definition and implementation of each part of the diagram:

- Attention
- Scaled dot product attention
- Multi-Head Attention
- QKV projections
- Layer norm / dropout
- Positional embeddings (Maybe RoPE?)
- Masking

Before getting started, let's define some symbols for the dimensions involved, and import some things we'll use in the code snippets:

```python
# Dimensions

# B    Batch size
# T    Sequence length
# T_q  When computing cross-attention, sequence length corresponding to queries
# T_k  When computing cross-attention, sequence length corresponding to keys and values
# D    Dimension of keys and queries, often written d_k
# d_v  Dimension of values. This post assumes d_v = D
# H    Attention head dimension
```

```python
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

```

### If Attention is All You Need, What *Don't* You Need? Recurrence.

*"We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely"*

Before transformers, most sequence-to-sequence models relied on [recurrent neural networks](https://doi.org/10.1207/s15516709cog1402_1) (RNNs) and their more sophisticated variants like [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) (LSTM) and [Gated Recurrent Units](https://arxiv.org/abs/1406.1078) (GRUs). These architectures process sequences one element at a time, updating a hidden state as each new element arrives. This design gives RNNs a natural sense of order but also serious limitations: long-range dependencies are difficult to capture, gradients tend to vanish over many time steps, and the inherently sequential nature of recurrence makes training slow and difficult to parallelize.

<figure style="text-align: center;">
  <img src="/assets/images/seq-to-seq-encoder-decoder.png" alt="Encoder-Decoder" style="max-width: 100%; height: auto; margin: auto;">
  <figcaption>An encoder-decoder architecture with fixed-length context vector (encoding). <a href="https://lilianweng.github.io/posts/2018-06-24-attention/">Image credit</a>: Lilian Weng, "Attention? Attention!"</figcaption>
</figure>


A specific example: The [sequence-to-sequence model](https://arxiv.org/abs/1409.3215) introduced by Sutskever et al. (2014) extended RNNs into an encoder-decoder framework, where the encoder compressed the entire input sequence into a single fixed-length context vector, and the decoder generated the output sequence from it. This worked well for short sentences, but as the sequences grow longer, that single vector becomes a bottleneck, forcing the network to represent all relevant information in a narrow space. Transformers break from this paradigm. Instead of a fixed-length context vector, transformers process all elements of the sequence simultaneously and model the relationships between each pair of them using a mechanism called *attention*.

## Attention

Attention allows a model to dynamically focus on the most relevant parts of its input when making each decision, weighting information based on how strongly it relates to the current context.

<figure style="text-align: center;">
  <img src="/assets/images/attention-weighted-edges.png" alt="Attention" style="max-width: 100%; height: auto; margin: auto;">
  <figcaption>Attention models the relevance of a query to each element of a sequence. Image credit: unknown (TODO: identify source or redraw)</figcaption>
</figure>


Attention was first introduced in the context of neural machine translation as a way to overcome the limitations of fixed-length context vectors used in early encoder–decoder models. In Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014), the authors proposed allowing the decoder to “softly search” through all positions of the input sequence at each step of translation—deciding dynamically which parts of the input were most relevant to the word being generated. This replaced the rigid, one-size-fits-all summary of the source sentence with a flexible mechanism that could focus on different words at different times. Subsequent refinements, such as Luong et al. (2015)’s Effective Approaches to Attention-Based Neural Machine Translation, further formalized this idea, showing that attention consistently improved translation quality, especially for long or complex sentences.

At its core, *attention is a method for computing a weighted average over a set of elements*, where the weights are determined dynamically based on the relationship between an input query and each element’s representation. Each element in the sequence is represented by a key and a value—the key describing what the element is about, and the value carrying the actual content we might want to retrieve. The query is another vector in the same feature space, representing what we’re looking for. The attention mechanism measures how relevant each key is to the query, typically using a similarity function such as a dot product. Elements whose keys are most similar to the query receive higher weights, meaning the model “attends” to them more strongly.

<figure style="text-align: center;">
  <img src="/assets/images/dot-product-attention.png" alt="Dot product attention" style="max-width: 100%; height: auto; margin: auto;">
  <figcaption>Dot product attention. <a href="https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html">Image credit</a>: Phillip Lippe, UvA Deep Learning Tutorials, "Tutorial 6: Transformers and Multi-Head Attention"</figcaption>
</figure>

## Scaled Dot Product Attention

<figure style="text-align: center;">
  <img src="/assets/images/scaled-dot-product-attention-diagram.png" alt="Scaled dot product attention" style="max-width: 100%; height: auto; margin: auto;">
  <figcaption>Scaled dot product attention</figcaption>
</figure>


Formally, this process can be viewed as mapping a query and a set of key–value pairs to an output vector. The relevance scores between the query and each key are computed via dot products, scaled for numerical stability, and passed through a softmax function to form a probability distribution over the elements. The final output is then a weighted sum of the values — effectively the expected value of the sequence given the query. This is known as *scaled dot-product attention*. 

For a single query, we have

<figure style="text-align: center;">
  <img src="/assets/images/single-query-scaled-dpa.png" alt="Scaled dot product attention" style="max-width: 100%; height: auto; margin: auto;">
  <figcaption>Scaled dot product attention</figcaption>
</figure>

The scaling term $\sqrt{d_k}$ in the denominator serves to normalize the magnitude of the dot products between queries and keys. Without it, when the dimensionality $d_k$ is large, the dot products tend to grow in magnitude, pushing the softmax function into regions where it becomes extremely peaked. As a result, gradients vanish and learning slows. We can see this by looking at the gradients of the softmax function w.r.t. its inputs. Recall that the softmax of a vector $z$ is given by

$$ \text{softmax}(\mathbf{z}) = \frac{e^{z_i}}{ \sum_{j = 1}^N e^{z_j}}$$

and its gradients are 

<figure style="text-align: center;">
  <img src="/assets/images/softmax-gradients.png" alt="Softmax gradients" style="max-width: 100%; height: auto; margin: auto;">
  <figcaption>Gradients of the softmax function. <a href="https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1">Image credit</a>: Thomas Kurbiel, "Derivative of the Softmax Function and the Categorical Cross-Entropy Loss"</figcaption>
</figure>

When softmax saturates, a single $s_i$ value approaches 1, and the others approach $0$, and the gradients all vanish. Dividing by $\sqrt{d_k}$ counteracts this effect by keeping the variance of the dot products roughly constant, ensuring that attention weights remain well-scaled and that the model can learn effectively across different embedding dimensions.

In code:

```python
def scaled_dot_product_attention(
    q: torch.Tensor,   # (d_k,)
    K: torch.Tensor,   # (T, d_k)
    V: torch.Tensor,   # (T, d_v)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention for a single query.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - context (torch.Tensor): The attention output, a weighted sum
              of values with shape (d_v,).
            - weights (torch.Tensor): The attention weights (probabilities)
              over the T keys with shape (T,).
    """
    T, d_k = K.size(0), K.size(1)
    # Scores over T keys: (T,)
    scores = torch.matmul(q, K.t()) / math.sqrt(d_k)

    # Attention weights over T positions: (T,)
    # Subtracting scores.max() improves numerical stability by preventing overflow in exp()
    weights = F.softmax(scores - scores.max(), dim=0)

    # Weighted sum of values: (d_v,)
    context = torch.matmul(weights.unsqueeze(0), V).squeeze(0)

    return context, weights
```


## Self-Attention
So far, our attention mechanism takes a sequence of elements (i.e., keys and values) and a query, and returns a single value. We can extend this mechanism by associating a query with each element, so we can compute how much each element "attends" to every other element:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

The matrix multiplication $QK^T$ performs the dot product for every possible pair of queries and keys, resulting in a matrix of shape $T \times T$.

In the special case where the queries, keys, and values all come from the same sequence, we call this *self-attention*. 
Self-attention allows every element in a sequence to attend to every other element, including itself. This means that each element’s representation is updated based on a weighted combination of all elements in the sequence, where the weights reflect their relevance to that element.

<figure style="text-align: center;">
  <img src="/assets/images/attention-weights-small.png" alt="Attention weights" style="max-width: 100%; height: auto; margin: auto;">
  <figcaption>Self-attention weights for each pair of positions. Image credit: unknown (TODO: identify source or redraw)</figcaption>
</figure>

Let's update the implementation of scaled_dot_product_attention to compute attention for multiple queries, and allow for an optional dimension for batches:

```python
def scaled_dot_product_attention(
    Q: torch.Tensor,  # (T, d_k) or (B, T, d_k)
    K: torch.Tensor,  # (T, d_k) or (B, T, d_k)
    V: torch.Tensor,  # (T, d_v) or (B, T, d_v)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled dot-product attention (multiple queries, optional batching)
      
    Returns:
        context: (T, d_v) or (B, T, d_v)
        weights: (T, T) or (B, T, T)
    """
    # Validate ranks
    if Q.dim() not in (2, 3) or K.dim() != Q.dim() or V.dim() != Q.dim():
        raise ValueError("Q, K, V must all be rank-2 (T,D) or rank-3 (B,T,D) tensors of the same rank.")

    # Validate inner dims
    if Q.size(-1) != K.size(-1):
        raise ValueError(f"Query/key dims must match: {Q.size(-1)} != {K.size(-1)}")
    if K.size(-2) != V.size(-2):
        raise ValueError(f"Time dims of K and V must match: {K.size(-2)} != {V.size(-2)}")

    # Normalize to 3D: (B, T, D) where B=1 for unbatched
    def _to_3d(x):
        return x.unsqueeze(0) if x.dim() == 2 else x

    Q3, K3, V3 = _to_3d(Q), _to_3d(K), _to_3d(V)
    d_k = Q3.size(-1)

    # Scores: (B, T, T)
    scores = torch.matmul(Q3, K3.transpose(-2, -1)) / math.sqrt(d_k)

    # Attention weights
    weights = F.softmax(scores, dim=-1) # Softmax over keys

    # Context: (B, T, d_v)
    context = torch.matmul(weights, V3)

    # Restore to original rank
    if Q.dim() == 2:
        context = context.squeeze(0)   # (T, d_v)
        weights = weights.squeeze(0)   # (T, T)

    return context, weights

```

## Multi-Head Attention

<figure style="text-align: center;">
  <img src="/assets/images/multi-head-attention-diagram.png" alt="Multi-Head Attention Diagram" style="max-width: 100%; height: auto; margin: auto;">
  <figcaption>Multi-head attention</figcaption>
</figure>

Scaled dot-product attention gives us a powerful primitive: given a query, we compute a weighted average over a set of values based on their relevance. However, a single attention operation forces the model to express all relationships in one shared representation space. In practice, different types of relationships—such as syntactic structure, semantic similarity, or positional alignment—can interfere with one another when compressed into a single set of attention weights. Multi-head attention addresses this limitation by running several attention operations in parallel, each with its own learned projections of queries, keys, and values.

Intuitively, each attention “head” can be thought of as looking at the sequence from a different perspective. One head might learn to attend to nearby words, another to long-range dependencies, and another to specific structural patterns like subject–verb agreement. Because each head operates on a lower-dimensional subspace, the model can represent diverse relationships simultaneously without forcing them into a single attention map. The outputs of these heads are then combined, allowing the model to integrate multiple views of the same sequence into a richer representation.

Casually, multi-head attention is a fancy name for running several (smaller) attention instances in parallel.

Formally, multi-head attention applies $h$ independent scaled dot-product attention operations. Given input matrices $Q \in \mathbb{R}^{T \times D}$, $K \in \mathbb{R}^{T \times D}$, and 
$V \in \mathbb{R}^{T \times D}$, we first project them using learned linear transformations:

$$
Q_i = Q W_i^Q \in \mathbb{R}^{T \times H}, \quad
K_i = K W_i^K \in \mathbb{R}^{T \times H}, \quad
V_i = V W_i^V \in \mathbb{R}^{T \times H}
$$

Here, $H$ is the dimension of each head, often chosen as $H = D / h$ so that the total computational cost remains similar to that of single-head attention with dimension $D$.

Each head computes scaled dot-product attention independently:

$$\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)$$

The resulting head outputs are concatenated and linearly transformed once more:

$$
 \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W^O 
$$

where $W^O \in \mathbb{R}^{(h \cdot H) \times D}$. 


Back to the code. Let's extend scaled_dot_product_attention to accept a tensor with leading dimensions for batching and heads:

Note the two sequence lengths in the signature. Self-attention draws queries, keys, and values from the same sequence, so `T_q == T_k`; cross-attention draws the queries from a different sequence, so the two can differ. Writing them separately keeps the decoder's cross-attention working without a second implementation.

```python
def scaled_dot_product_attention(
    Q: torch.Tensor,                     # (B, H, T_q, d_k)
    K: torch.Tensor,                     # (B, H, T_k, d_k)
    V: torch.Tensor,                     # (B, H, T_k, d_v)
    mask: Optional[torch.Tensor] = None, # broadcastable to (B, H, T_q, T_k)
    dropout_p: float = 0.0,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute scaled dot-product attention

    Args:
        Q: Queries of shape (B, H, T_q, d_k)
        K: Keys    of shape (B, H, T_k, d_k)
        V: Values  of shape (B, H, T_k, d_v)
        mask: Optional attention mask, broadcastable to (B, H, T_q, T_k).
              - If boolean: True means "keep/allow", False means "mask out".
              - If floating: additive mask (e.g., 0 for keep, -inf for mask),
                added to the attention scores before softmax.
        dropout_p: Dropout probability applied to attention weights.
        training:  If true, dropout is applied

    Returns:
        context: Weighted sum of values, shape (B, H, T_q, d_v)
        attn:    Attention weights over keys, shape (B, H, T_q, T_k)
    """
    d_k = Q.size(-1)

    # Scores: (B, H, T_q, T_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Masking
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask

    # Numerical stability: subtract per-row max so exp() doesn't overflow (softmax is shift-invariant).
    scores = scores - scores.max(dim=-1, keepdim=True).values

    # Attention weights: (B, H, T_q, T_k)
    attn = F.softmax(scores, dim=-1)

    # Optional dropout on attention weights
    if dropout_p and dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=training)

    # Context: (B, H, T_q, d_v)
    context = torch.matmul(attn, V)

    return context, attn

```

```python
class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module.

    Shapes:
      - Input:  x_q: (B, T_q, d_model)
                x_k: (B, T_k, d_model)  (defaults to x_q if not provided)
                x_v: (B, T_k, d_model)  (defaults to x_k if not provided)
      - Mask:   broadcastable to (B, H, T_q, T_k)
      - Output: y:   (B, T_q, d_model)
                attn:(B, H, T_q, T_k)

    Notes:
      - This is the "standard" Transformer MHA: per-head d_k = d_v = d_model / num_heads.
      - For tutorial clarity, d_model must be divisible by num_heads.
    """

    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.0, bias: bool = True):
        """
          Args:
            d_model: Total embedding of the input and output tensors.
            num_heads: Number of attention heads. d_model must be divisible by num_heads.
            dropout_p: Dropout probability applied to the attention weights.
            bias: If true, include additive bias terms in the linear projections.
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model must be divisible by num_heads; got d_model={d_model}, num_heads={num_heads}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.dropout_p = float(dropout_p)

        # Q/K/V projections are implemented as single "packed" linear layers.
        # Conceptually, each head h has its own projection W_h^Q, W_h^K, W_h^V
        # with shape (d_model, d_head). In practice we concatenate the H per-head
        # matrices along the output dimension to form one matrix:
        #
        #   W^Q = [W_1^Q | W_2^Q | ... | W_H^Q] ∈ R^{d_model × (H·d_head)} = R^{d_model × d_model}
        #
        # so applying nn.Linear(d_model, d_model) produces an output of shape
        # (B, T, H·d_head), which we then reshape into (B, H, T, d_head) to recover
        # the per-head queries/keys/values.
        self.W_q = nn.Linear(d_model, d_model, bias=bias)
        self.W_k = nn.Linear(d_model, d_model, bias=bias)
        self.W_v = nn.Linear(d_model, d_model, bias=bias)

        # Output projection that mixes information across heads.
        # After attention, per-head outputs are concatenated back into
        # a (B, T, d_model) tensor; this linear layer projects that
        # concatenated representation back into the model space.
        self.W_o = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """
        Xavier (Glorot) uniform initialization for the four projections.

        nn.Linear defaults to Kaiming-uniform, which is scaled for ReLU-style
        activations. The attention projections have no activation, so Xavier's
        fan_in/fan_out scaling is the better match: it keeps the variance of
        the activations roughly constant across the projection, which in turn
        keeps the pre-softmax scores in a range where the softmax gradient has
        not yet saturated.
        """
        for proj in (self.W_q, self.W_k, self.W_v, self.W_o):
            nn.init.xavier_uniform_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split the model dimension into multiple attention heads.

        This reshapes and permutes a tensor from (B, T, d_model) into
        (B, H, T, d_head), where d_head = d_model / H.
        """
        B, T, _ = x.shape
        x = x.view(B, T, self.num_heads, self.d_head)
        return x.transpose(1, 2)  # (B, H, T, d_head)

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Combine multiple attention heads back into the model dimension.

        This permutes and reshapes a tensor from (B, H, T, d_head) into
        (B, T, d_model) by concatenating the head dimensions.
        """
        B, H, T, d_head = x.shape
        x = x.transpose(1, 2).contiguous()  # (B, T, H, d_head)
        return x.view(B, T, H * d_head)

    def forward(
        self,
        x_q: torch.Tensor,                        # (B, T_q, d_model)
        x_k: Optional[torch.Tensor] = None,       # (B, T_k, d_model)
        x_v: Optional[torch.Tensor] = None,       # (B, T_k, d_model)
        mask: Optional[torch.Tensor] = None,      # broadcastable to (B, H, T_q, T_k)
        return_attn: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_q: Query input, shape (B, T_q, d_model)
            x_k: Key input,   shape (B, T_k, d_model). Defaults to x_q (self-attention).
            x_v: Value input, shape (B, T_k, d_model). Defaults to x_k.
            mask: Optional mask broadcastable to (B, H, T_q, T_k).
            return_attn: If True, also return attention weights (B, H, T_q, T_k).

        Returns:
            y: (B, T_q, d_model)
            attn (optional): (B, H, T_q, T_k)
        """
        if x_k is None:
            x_k = x_q
        if x_v is None:
            x_v = x_k

        # Project to model dim, then split into heads.
        Q = self._split_heads(self.W_q(x_q))  # (B, H, T_q, d_head)
        K = self._split_heads(self.W_k(x_k))  # (B, H, T_k, d_head)
        V = self._split_heads(self.W_v(x_v))  # (B, H, T_k, d_head)

        # Attention per head.
        context, attn = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout_p=self.dropout_p, training=self.training
        )  # context: (B, H, T_q, d_head)

        # Recombine heads and project out.
        y = self.W_o(self._combine_heads(context))  # (B, T_q, d_model)

        if return_attn:
            return y, attn
        return y
```

## Keys, Values, Queries?
- Keys, Values, Queries aren't given, they're computed via projection matrices
- Where did these matrices come from? They're learned jointly with the rest of the model
- Meta-lesson: the architecture can assume that functions exist, and then learn them into existence

So far we have talked about queries, keys, and values as though they were given — each element arriving with a key describing what it is about and a value carrying its content. Look at what `forward` actually receives: a single tensor `x_q` of shape `(B, T, d_model)`, and in the self-attention case that one tensor supplies the keys and values too. Keys, values, and queries aren't given. They're computed, via the projection matrices `self.W_q`, `self.W_k`, and `self.W_v`.

The three projections map the same input into three different subspaces, so a single element representation can act as a query when asking what it needs from the sequence, a key when advertising what it offers, and a value when delivering its content. The separation also makes the relationship directional: with one projection producing both queries and keys, $QK^T$ would be symmetric, and "the verb attends to its subject" would carry exactly the same weight as "the subject attends to its verb."

Where did these matrices come from? They're learned jointly with the rest of the model. `W_q`, `W_k`, and `W_v` are `nn.Linear` layers like any other. The training objective says nothing about what makes a good key or a good query; the only signal is the loss at the output, and the projections arrive at whatever notion of relevance reduces it.

Meta-lesson: Attention assumes that a useful notion of relevance exists, then learns it into existence.

<figure style="text-align: center;">
  <img src="/assets/images/projections-and-attention.png" alt="Projection matrices, self-attention" style="max-width: 100%; height: auto; margin: auto;">
  <figcaption>Projection matrices and self-attention. <a href="https://newsletter.theaiedge.io/p/understanding-the-self-attention">Image credit</a>: The AiEdge, "Understanding the Self-Attention Mechanism"</figcaption>
</figure>


## Encoder-Decoder

<figure style="text-align: center;">
  <img src="/assets/images/transformer-encoder-and-decoder-architectures.png" alt="Encoder-Decoder Architectures" style="max-width: 100%; height: auto; margin: auto;">
  <figcaption>Encoder-Decoder Architectures. <a href="https://towardsdatascience.com/a-complete-guide-to-bert-with-code-9f87602e4a11/">Image credit</a>: Bradney Smith, "A Complete Guide to BERT with Code", adapted from the architecture diagram in <em>Attention Is All You Need</em></figcaption>
</figure>


```python
class EncoderBlock(nn.Module):
    """
    A standard Transformer encoder block (pre-norm), composed of:

      1) Multi-head self-attention + residual
      2) Position-wise feed-forward network + residual

    This implementation uses "pre-norm" LayerNorm:
      x -> LN -> sublayer -> dropout -> residual add

    Shapes:
      x:    (B, T, d_model)
      mask: optional, broadcastable to (B, H, T, T) for self-attention

    Returns:
      y:    (B, T, d_model)
      attn: (B, H, T, T) if return_attn=True
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_p: float = 0.0,
        activation: str = "gelu",
        bias: bool = True,
    ):
        """
        Initialize an encoder block.

        Args:
            d_model: Model/embedding dimension.
            num_heads: Number of attention heads.
            d_ff: Hidden dimension of the feed-forward network (FFN).
            dropout_p: Dropout probability applied after attention/FFN outputs.
            activation: Activation function in the FFN ("gelu" or "relu").
            bias: Whether to include bias terms in linear layers.
        """
        super().__init__()
        self.dropout_p = float(dropout_p)

        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout_p=dropout_p, bias=bias)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            nn.GELU() if activation.lower() == "gelu" else nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=bias),
        )

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(
        self,
        x: torch.Tensor,                    # (B, T, d_model)
        mask: Optional[torch.Tensor] = None, # broadcastable to (B, H, T, T)
        return_attn: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequence embeddings, shape (B, T, d_model).
            mask: Optional attention mask broadcastable to (B, H, T, T).
            return_attn: If True, also return attention weights.

        Returns:
            y: Updated embeddings, shape (B, T, d_model).
            attn (optional): Attention weights, shape (B, H, T, T).
        """
        # ---- Self-attention sublayer (pre-norm) ----
        x_norm = self.norm1(x)
        if return_attn:
            attn_out, attn = self.self_attn(x_q=x_norm, mask=mask, return_attn=True)
        else:
            attn_out = self.self_attn(x_q=x_norm, mask=mask, return_attn=False)

        x = x + self.dropout(attn_out)

        # ---- Feed-forward sublayer (pre-norm) ----
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)

        return (x, attn) if return_attn else x
```

The original Transformer paper, Attention Is All You Need, used the post-norm formulation, where LayerNorm is applied after each residual addition. This worked well for the relatively shallow models studied at the time (e.g., 6 encoder layers), but later work found that post-norm Transformers become increasingly difficult to train as depth grows, often requiring careful learning-rate warmup and tuning to remain stable. As a result, most modern implementations adopt the pre-norm variant, which applies LayerNorm before each sublayer. Pre-norm preserves an exact identity path through the residual connections, greatly improving gradient flow and making deep Transformers significantly more stable and easier to optimize, which is why it is now the default choice in practice.

```python
from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn


class DecoderBlock(nn.Module):
    """
    A standard Transformer decoder block (pre-norm), composed of:

      1) Masked multi-head self-attention + residual
      2) Cross-attention over encoder outputs + residual
      3) Position-wise feed-forward network + residual

    This implementation uses "pre-norm" LayerNorm:
      x -> LN -> sublayer -> dropout -> residual add

    Shapes:
      x:        (B, T, d_model)          decoder hidden states
      enc_out:  (B, S, d_model)          encoder outputs (source length S)
      self_mask:  optional, broadcastable to (B, H, T, T)   (typically causal and/or padding)
      cross_mask: optional, broadcastable to (B, H, T, S)   (typically padding over source keys)

    Returns:
      y:        (B, T, d_model)
      attn_self:  (B, H, T, T) if return_attn=True
      attn_cross: (B, H, T, S) if return_attn=True
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout_p: float = 0.0,
        activation: str = "gelu",
        bias: bool = True,
    ):
        """
        Initialize a decoder block.

        Args:
            d_model: Model/embedding dimension.
            num_heads: Number of attention heads.
            d_ff: Hidden dimension of the feed-forward network (FFN).
            dropout_p: Dropout probability applied after attention/FFN outputs.
            activation: Activation function in the FFN ("gelu" or "relu").
            bias: Whether to include bias terms in linear layers.
        """
        super().__init__()
        self.dropout_p = float(dropout_p)

        self.self_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout_p=dropout_p, bias=bias)
        self.cross_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout_p=dropout_p, bias=bias)

        self.norm1 = nn.LayerNorm(d_model)  # before self-attn
        self.norm2 = nn.LayerNorm(d_model)  # before cross-attn
        self.norm3 = nn.LayerNorm(d_model)  # before FFN

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            nn.GELU() if activation.lower() == "gelu" else nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=bias),
        )

        self.dropout = nn.Dropout(p=dropout_p)

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, d_model)
        enc_out: torch.Tensor,                    # (B, S, d_model)
        self_mask: Optional[torch.Tensor] = None, # broadcastable to (B, H, T, T)
        cross_mask: Optional[torch.Tensor] = None,# broadcastable to (B, H, T, S)
        return_attn: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Decoder input (hidden states), shape (B, T, d_model).
            enc_out: Encoder outputs to attend over, shape (B, S, d_model).
            self_mask: Optional mask for decoder self-attention, broadcastable to (B, H, T, T).
            cross_mask: Optional mask for cross-attention, broadcastable to (B, H, T, S).
            return_attn: If True, also return (self_attn_weights, cross_attn_weights).

        Returns:
            y: Updated decoder states, shape (B, T, d_model).
            attn_self (optional): (B, H, T, T)
            attn_cross (optional): (B, H, T, S)
        """
        # ---- Masked self-attention (pre-norm) ----
        x_norm = self.norm1(x)
        if return_attn:
            self_out, attn_self = self.self_attn(x_q=x_norm, mask=self_mask, return_attn=True)
        else:
            self_out = self.self_attn(x_q=x_norm, mask=self_mask, return_attn=False)
        x = x + self.dropout(self_out)

        # ---- Cross-attention (pre-norm): queries from decoder, keys/values from encoder ----
        x_norm = self.norm2(x)
        if return_attn:
            cross_out, attn_cross = self.cross_attn(x_q=x_norm, x_k=enc_out, mask=cross_mask, return_attn=True)
        else:
            cross_out = self.cross_attn(x_q=x_norm, x_k=enc_out, mask=cross_mask, return_attn=False)
        x = x + self.dropout(cross_out)

        # ---- Feed-forward (pre-norm) ----
        x_norm = self.norm3(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)

        if return_attn:
            return x, attn_self, attn_cross
        return x
```

## Detailed Recap
Revisit the diagram at the start of the post in more detail. This could show the dimensions of matrices, einsums, etc.

# References and Further Reading

1. [Attention is All You Need](https://arxiv.org/abs/1706.03762)
1. [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
1. [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)
1. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
1. [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
1. [Finding Structure in Time](https://doi.org/10.1207/s15516709cog1402_1) (Elman, 1990 — introduces the simple recurrent network)
1. [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735) (Hochreiter & Schmidhuber, 1997)
1. [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) (Cho et al., 2014 — introduces the GRU)
1. [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555) (Chung et al., 2014 — LSTM vs. GRU comparison)
1. [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) (Sutskever et al., 2014)
1. [Derivative of the Softmax Function and the Categorical Cross-Entropy Loss](https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1)
	
