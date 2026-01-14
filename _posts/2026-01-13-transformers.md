# Deep Learning: Transformers

This post unpacks the original transformer from first principles. We'll working through the intuition, architecture, and a PyTorch implementation. By the end, you should be able to train and run your own transformer, and feel comfortable with each component in this diagram:

<figure style="text-align: center;">
  <img src="/assets/images/transformer-architecture.png" alt="Transformer Architecture" style="max-width: 100%; height: auto; margin: auto;">
  <figcaption>Transformer Architecture. Image credit: Efficient Transformers: A Survey</figcaption>
</figure>
 
# The Transformer

In less than a decade, transformers have become the dominant architecture in modern machine learning. Introduced by Vaswani et al. in the 2017 paper *Attention Is All You Need*, the transformer architecture replaced recurrence and convolution with a single mechanism: attention. Earlier sequence models like RNNs and LSTMs compressed all prior information into a fixed-length hidden state, forcing the network to distill an entire sequence into a single vector. In contrast, the transformer allows each token to directly "attend to" every other token, and produces a representation of the token that incorporates context from the most relevant parts of the entire input.

This section steps through the definition and implementation of each part of the diagram:

- Attention
- Scaled dot product attention, 
- Multi-Head Attention
- QKV projections
- Layer norm / dropout
- Positional embeddings (Maybe RoPE?)
- Masking

Before getting started, let's define some symbols for the dimensions involved, and import some things we'll use in the code snippets:

```python
# Dimensions

# B - Batch size
# T - Sequence length
# D - d_model, embedding dimension
```

```python

import math
import torch
import torch.nn.functional as F
from typing import Tuple

```

### If Attention is All You Need, What *Don't* You Need? Recurrence.
Before transformers, most sequence-to-sequence models relied on recurrent neural networks (RNNs) and their more sophisticated variants like LSTMs and GRUs. These architectures process sequences one element at a time, updating a hidden state as each new token arrives. This design gives RNNs a natural sense of order but also serious limitations: long-range dependencies are difficult to capture, gradients tend to vanish over many time steps, and the inherently sequential nature of recurrence makes training slow and difficult to parallelize.

<figure style="text-align: center;">
  <img src="/assets/images/seq-to-seq-encoder-decoder.png" alt="Encoder-Decoder" style="max-width: 100%; height: auto; margin: auto;">
  <figcaption>An encoder-decoder architecture with fixed-length context vector (encoding). Image credit: ???</figcaption>
</figure>


A specific example: The sequence-to-sequence model introduced by Sutskever et al. (2014) extended RNNs into an encoder-decoder framework, where the encoder compressed the entire input sequence into a single fixed-length context vector, and the decoder generated the output sequence from it. This worked well for short sentences, but as sequences grew longer, that single vector became a bottleneck—forcing the network to represent all relevant information in a narrow space. Transformers broke from this paradigm entirely. Instead of a fixed-length context vector, transformers process all tokens simultaneously and model the relationships between each pair of tokens using a mechanism called *attention*.

## Attention

Attention allows a model to dynamically focus on the most relevant parts of its input when making each decision, weighting information based on how strongly it relates to the current context.

<figure style="text-align: center;">
  <img src="/assets/images/attention-weighted-edges.png" alt="Attention" style="max-width: 100%; height: auto; margin: auto;">
  <figcaption>Attention models the relevance of a query to each element of a sequence. Image credit: ???</figcaption>
</figure>


Attention was first introduced in the context of neural machine translation as a way to overcome the limitations of fixed-length context vectors used in early encoder–decoder models. In Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al., 2014), the authors proposed allowing the decoder to “softly search” through all positions of the input sequence at each step of translation—deciding dynamically which parts of the input were most relevant to the word being generated. This replaced the rigid, one-size-fits-all summary of the source sentence with a flexible mechanism that could focus on different words at different times. Subsequent refinements, such as Luong et al. (2015)’s Effective Approaches to Attention-Based Neural Machine Translation, further formalized this idea, showing that attention consistently improved translation quality, especially for long or complex sentences.

At its core, *attention is a method for computing a weighted average over a set of elements*, where the weights are determined dynamically based on the relationship between an input query and each element’s representation. Each element in the sequence is represented by a key and a value—the key describing what the element is about, and the value carrying the actual content we might want to retrieve. The query is another vector in the same feature space, representing what we’re looking for. The attention mechanism measures how relevant each key is to the query, typically using a similarity function such as a dot product. Elements whose keys are most similar to the query receive higher weights, meaning the model “attends” to them more strongly.

<figure style="text-align: center;">
  <img src="/assets/images/dot-product-attention.png" alt="Dot product attention" style="max-width: 100%; height: auto; margin: auto;">
  <figcaption>Dot product attention. Image credit: ???</figcaption>
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

The scaling term $\sqrt(d_k)$ in the denominator serves to normalize the magnitude of the dot products between queries and keys. Without it, when the dimensionality $d_k$ is large, the dot products tend to grow in magnitude, pushing the softmax function into regions where it becomes extremely peaked. causing gradients to vanish and learning to become unstable. This is evident from the gradients of the softmax function w.r.t its inputs. Recall that the softmax of a vector $z$ is given by

$$ \text{softmax}(\mathbf{z}) = \frac{e^{z_i}}{ \sum_{j = 1}^N e^{z_j}}$$

and its gradients are 

<figure style="text-align: center;">
  <img src="/assets/images/softmax-gradients.png" alt="Softmax gradients" style="max-width: 100%; height: auto; margin: auto;">
  <figcaption>Gradients of the softmax function. Image credit: ???</figcaption>
</figure>

When softmax saturates, a single $s_i$ value approaches 1, and the others approach $0$, and the gradients all vanish. Dividing by $\sqrt(d_k)$ counteracts this effect by keeping the variance of the dot products roughly constant, ensuring that attention weights remain well-scaled and that the model can learn effectively across different embedding dimensions.

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

<figure style="text-align: center;">
  <img src="/assets/images/scaled-dot-product-attention-equation.png" alt="Scaled dot product attention" style="max-width: 100%; height: auto; margin: auto;">
  <figcaption>Scaled dot product attention. Image credit: ???</figcaption>
</figure>

In the special case where the queries, keys, and values all come from the same sequence, we call this *self-attention*. 
Self-attention allows every element in a sequence to attend to every other element, including itself. This means that each token’s representation is updated based on a weighted combination of all tokens in the sequence, where the weights reflect their relevance to that token.

TODO: Self-attention diagram


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

## Keys, Values, Queries?
- Keys, Values, Queries aren't given, they're computed via projection matrices
- Where did these matrics come from? They're learned jointly with the rest of the model
- Meta-lesson: the architecture can assume that functions exist, and then learn them into existence

<figure style="text-align: center;">
  <img src="/assets/images/projections-and-attention.png" alt="Projection matrices, self-attention" style="max-width: 100%; height: auto; margin: auto;">
  <figcaption>Projection matrices and self-attention. [Image credit](https://newsletter.theaiedge.io/p/understanding-the-self-attention)</figcaption>
</figure>

## Multi-Head Attention

<figure style="text-align: center;">
  <img src="/assets/images/multi-head-attention-diagram.png" alt="Multi-Head Attention Diagram" style="max-width: 100%; height: auto; margin: auto;">
  <figcaption>Mlti-head attention</figcaption>
</figure>


## Encoder-Decoder

## Input Embeddings

## Training

## Inference

# References and Further Reading

1. [Attention is All You Need](https://arxiv.org/abs/1706.03762)
1. [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
1. [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)
1. [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
1. [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
1. [Derivative of the Softmax Function and the Categorical Cross-Entropy Loss](https://medium.com/data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1)
	
