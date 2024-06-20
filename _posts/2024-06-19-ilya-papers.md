## Ilya's List of AI papers: Key Takeaways

The [internet has been talking about](https://news.ycombinator.com/item?id=34641359) a list of AI papers that Ilya Sutskever, co-founder of OpenAI, recommended to John Carmack:

*"So I asked Ilya, their chief scientist, for a reading list. This is my path, my way of doing things: give me a 
stack of all the stuff I need to know to actually be relevant in this space. And he gave me a list of like 40 
research papers and said, ‘If you really learn all of these, you’ll know 90% of what matters today.’ And I did. I plowed through all those things and it all started sorting out in my head."*

The exact list seems to be lost to time, but something like it recently [resurfaced on Xwitter](https://x.
com/keshavchan/status/1787861946173186062). Regardless of origin, the list includes some landmark papers in the development of large language models — i.e., the class of artificial neural networks responsible for things like ChatGPT. Together, they tell the story of how we learned to assemble basic neural units into increasingly sophisticated architectures, what those architectures are capable of learning, and the practical techniques that make training them possible.

# "Ilya's List" of AI Papers
I've collected the papers here, along with some key takeaways from each. I've put them in chronological order to 
make it easier to see how they build on each other. Most of them are papers, but a few are books, blog posts, course 
notes, etc. In any case, it's a lot of reading! I've highlighted a few that I think are particularly worth a look.

* [1993 - Keeping Neural Networks Simple By Minimizing the Description Length of the Weights](#keeping-neural-networks-simple-by-minimizing-the-description-length-of-the-weights)
* [2004 - A Tutorial Introduction to the Minimum Description Length Principle](#a-tutorial-introduction-to-the-minimum-description-length-principle)
* [2008 - Machine Super Intelligence](#machine-super-intelligence)
* [2011 - The First Law of Complexodynamics](#the-first-law-of-complexodynamics)
* [__2012 - ImageNet Classification with Deep Convolutional Neural Networks__](#imagenet-classification-with-deep-convolutional-neural-networks)
* __[2014 - Neural Turing Machines](#neural-turing-machines)__
* [2014 - Quantifying the Rise and Fall of Complexity in Closed Systems](#quantifying-the-rise-and-fall-of-complexity-in-closed-systems)
* [2015 - Deep Residual Learning for Image Recognition](#deep-residual-learning-for-image-recognition)
* [__2015 - Neural Machine Translation by Jointly Learning to Align and Translate__](#neural-machine-translation-by-jointly-learning-to-align-and-translate)
* [2015 - Recurrent Neural Network Regularization](#recurrent-neural-network-regularization)
* [__2015 - The Unreasonable Effectiveness of Recurrent Neural Networks__](#the-unreasonable-effectiveness-of-recurrent-neural-networks)
* [__2015 - Understanding LSTM Networks__](#understanding-lstm-networks)
* [2016 - Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](#deep-speech-2-end-to-end-speech-recognition-in-english-and-mandarin)
* [2016 - Identity Mappings in Deep Residual Networks](#identity-mappings-in-deep-residual-networks)
* [2016 - Multi-Scale Context Aggregation by Dilated Convolutions](#multi-scale-context-aggregation-by-dilated-convolutions)
* [2016 - Order Matters: Sequence to sequence for sets](#order-matters-sequence-to-sequence-for-sets)
* [2016 - Variational Lossy Autoencoder](#variational-lossy-autoencoder)
* [2017 - A Simple Neural Network Module for Relational Reasoning](#a-simple-neural-network-module-for-relational-reasoning)
* __[2017 - Attention is All You Need](#attention-is-all-you-need)__
* [2017 - Kolmogorov Complexity and Algorithmic Randomness](#kolmogorov-complexity-and-algorithmic-randomness)
* [2017 - Neural Message Passing for Quantum Chemistry](#neural-message-passing-for-quantum-chemistry)
* [2017 - Pointer Networks](#pointer-networks)
* [2018 - Relational Recurrent Neural Networks](#relational-recurrent-neural-networks)
* [2019 - GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](#gpipe-efficient-training-of-giant-neural-networks-using-pipeline-parallelism)
* [__2020 - Scaling Laws for Neural Language Models__](#scaling-laws-for-neural-language-models)
* [2024 - CS231n Convolutional Neural Networks for Visual Recognition](#cs231n-convolutional-neural-networks-for-visual-recognition)

# Keeping Neural Networks Simple by Minimizing the Description Length of the Weights
*Hinton, Geoffrey E., and Drew Van Camp, 1993* [(PDF)](https://www.cs.toronto.edu/~fritz/absps/colt93.pdf)

The [Minimum Description Length Principle](https://en.wikipedia.org/wiki/Minimum_description_length) asserts that the best model of some data is the one that minimizes the sum of the length of the description of the model and the length of the data encoded using that model.

* Applies the MDL principle to neural networks to control model complexity and prevent overfitting.
* Uses weight sharing to minimize the description length of the weights: The same weight is applied to multiple 
  connections in the network, reducing the number of free parameters, and thus the model complexity.

# A Tutorial Introduction to the Minimum Description Length Principle
*Peter Grünwald, 2004* [(PDF)](https://arxiv.org/pdf/math/0406077)

A clearly-written intro to the [Minimum Description Length Principle](https://en.wikipedia.org/wiki/Minimum_description_length). 

* Learning as data compression: Every regularity in data may be used to compress that data, and learning can be 
  equated with finding those regularities.
* MDL as model selection: The best model is the one that minimizes the sum of the length of the description of the 
  model and the length of the data encoded using that model.

# Machine Super Intelligence
*Legg, Shane, 2008* [(PDF)](https://www.vetta.org/documents/Machine_Super_Intelligence.pdf)

TODO

# The First Law of Complexodynamics

TODO

# ImageNet Classification with Deep Convolutional Neural Networks
*Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton, 2012* [(PDF)](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)

*"We trained a __large, deep convolutional neural network__ to classify the 1.2 million
high-resolution images in the ImageNet LSVRC-2010 contest into the 1000 different classes. On the test data, we 
achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous 
state-of-the-art. The neural network, which has __60 million parameters and 650,000 neurons__, consists of __five 
convolutional layers__, some of which are followed by max-pooling layers, __and three fully-connected layers__ with a 
final 1000-way softmax. To make training faster, we used non-saturating neurons and __a very efficient GPU 
implementation__ of the convolution operation. To reduce overfitting in the fully-connected layers we employed a 
recently-developed __regularization method called “dropout” that proved to be very effective__. We also entered a 
variant of this model in the ILSVRC-2012 competition and achieved a winning top-5 test error rate of 15.3%,
compared to 26.2% achieved by the second-best entry."*

__Takeaways__
* Introduced __AlexNet, a deep convolutional neural network__ that achieved unprecedented performance in image 
  classification. AlexNet won the ImageNet Large Scale Visual Recognition Challenge in 2012 by a substantial 
  margin, bringing deep learning to the forefront of computer vision research.
* __Deep Architecture__ AlexNet's depth -- five convolutional and three fully connected layers -- allowed the network to learn hierarchical feature representations, capturing complex patterns in the data that were not possible with shallower architectures.
* __Training with GPUs__ made it feasible to train deep networks on large datasets like ImageNet.

## Neural Turing Machines
*Graves, Alex, Greg Wayne, and Ivo Danihelka, 2014* [(PDF)](https://arxiv.org/pdf/1410.5401)

*"We extend the capabilities of neural networks by coupling them to __external memory resources__, which they can 
__interact with by attentional processes__. The combined system is analogous to a Turing Machine or Von Neumann 
architecture but is __differentiable end-to-end__, allowing it to be __efficiently trained with gradient descent__. 
Preliminary results demonstrate that Neural Turing Machines can infer simple algorithms such as copying, sorting, and associative recall from input and output examples"*

__Takeaways__

* Neural Turing Machines are __fully differentiable computers__ that learn their own programming. Differentiability 
  is the key to efficiently training NTMs with gradient descent.
* Augmenting a NN with external memory addresses a key limitation of vanilla RNNs: their inability to store 
  information for long periods of time.
* [Earlier work](https://dl.acm.org/doi/abs/10.1145/130385.130432) showed that __RNNs are Turing Complete__, meaning 
  that in principle they are capable of learning algorithms from examples, but didn't show how to 
  actually do it.

# Quantifying the Rise and Fall of Complexity in Closed Systems

TODO

# Deep Residual Learning for Image Recognition

TODO

# Neural Machine Translation by Jointly Learning to Align and Translate
*Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio, 2015* [(PDF)](https://arxiv.org/pdf/1409.0473)

*"Neural machine translation is a recently proposed approach to machine translation. Unlike the traditional 
statistical machine translation, the neural machine translation aims at building a single neural network that can be 
jointly tuned to maximize the translation performance. The models proposed recently for neural machine translation 
often belong to a family of encoder-decoders and consists of an encoder that encodes a source sentence into a 
fixed-length vector from which a decoder generates a translation. In this paper, __we conjecture that the use of a 
fixed-length vector is a bottleneck in improving the performance of this basic encoder-decoder architecture, and 
propose to extend this by allowing a model to automatically (soft-)search for parts of a source sentence that are relevant to predicting a target word__, without having to form these parts as a hard segment explicitly. With this new approach, we achieve a translation performance comparable to the existing state-of-the-art phrase-based system on the task of English-to-French translation. Furthermore, qualitative analysis reveals that the (soft-)alignments found by the model agree well with our intuition."*

__Takeaways__
- __Introduces the Attention Mechanism__, a major breakthrough in machine translation and many other 
  sequence-to-sequence tasks. The attention mechanism allows the model to focus on different parts of the source sentence while generating each word in the target sentence, addressing the limitations of fixed-length context vectors in previous encoder-decoder architectures.

# Recurrent Neural Network Regularization

TODO

# The Unreasonable Effectiveness of Recurrent Neural Networks
*Andrej Karpathy, 2015* [(Blog Post)](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

This blog post from one of the OpenAI co-founders is a great introduction to the power of Recurrent Neural Networks 
(RNNs) for sequence modeling. It shows that RNNs are deceptively simple, and demonstrates the power of RNNs to 
generate reasonable text using only character-by-character predictions. It's fun to see where generative language 
models were just a few years ago, and how far they've come since.

__Takeaways__
*"If training vanilla neural nets is optimization over functions, training recurrent nets is optimization over
programs."*
* Feed-Forward Neural Networks are limited to fixed-sized inputs and fixed-size outputs.
* RNNs operate on sequences of arbitrary length, making them well-suited for speech recognition, 
  language modeling, and machine translation.

# Understanding LSTM Networks
*Christopher Olah, 2015* [(Blog Post)](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

This blog post from one of the co-founders of Anthropic is a great introduction to an important type of RNN called 
[Long Short-Term Memory](https://deeplearning.cs.cmu.edu/F23/document/readings/LSTM.pdf) (LSTM) networks. In a 
step-by-step fashion, it explains how the different "gates" of an LSTM unit work together to store and retrieve 
information.

__Takeaways__
* LSTMs improve on RNNs by adding a piece of __hidden state called a memory cell__ that can store information for 
  long periods of time. This improves their ability to learn from long sequences, and makes them well-suited to natural language processing. In contrast, regular RNNs struggle to connect separate pieces of information, say, words that are far apart in a sentence.
* An LSTM unit contains a __memory cell__, an __input gate__, an __output gate__ and a __forget gate__ that regulate 
  the flow of information in and out of the memory cell.
* Interesting to see an idea (LSTMs) from 1997 returning to the forefront of research activity in 2015. Plenty of 
  things happened in those 18 years, including some pretty [dramatic increases in computational power](https://www.flickr.com/photos/jurvetson/51391518506/).

# Deep Speech 2: End-to-End Speech Recognition in English and Mandarin

TODO

# Identity Mappings in Deep Residual Networks

TODO

# Multi-Scale Context Aggregation by Dilated Convolutions

TODO

# Order Matters: Sequence to sequence for sets

TODO

# Variational Lossy Autoencoder

TODO

# A Simple Neural Network Module for Relational Reasoning

TODO

# Attention is All You Need
*Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser, Polosukhin. 2017*
[(Helpfully Annotated Paper)](https://nlp.seas.harvard.edu/annotated-transformer/) 
[(Original Paper)](https://arxiv.org/pdf/1706.03762)

"The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention
mechanism. __We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms, dispensing with recurrence and convolutions
entirely.__ Experiments on two machine translation tasks show..."

__Takeaways__
* Transformer architecture removes the recurrent (sequential) connections of RNNs. This allows efficient, parallel 
  training with GPUs.
* The self-attention mechanism enables the model to weigh the importance of different words in a sentence, 
  regardless of their position. This allows for capturing long-range dependencies and relationships more effectively than RNNs.

# Kolmogorov Complexity and Algorithmic Randomness

TODO

# Neural Message Passing for Quantum Chemistry

TODO

# Pointer Networks

TODO

# Relational Recurrent Neural Networks

TODO

# GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism

TODO

# Scaling Laws for Neural Language Models
*Kaplan, et al. 2020* [(PDF)](https://arxiv.org/pdf/2001.08361)

*"We study empirical scaling laws for language model performance on the cross-entropy loss.
__The loss scales as a power-law with model size, dataset size, and the amount of compute
used for training__, with some trends spanning more than seven orders of magnitude. Other
architectural details such as network width or depth have minimal effects within a wide
range. Simple equations govern the dependence of overfitting on model/dataset size and the
dependence of training speed on model size. These relationships allow us to determine the
__optimal allocation of a fixed compute budget__. Larger models are significantly more sample-efficient, such that 
__optimally compute-efficient training involves training very large models
on a relatively modest amount of data and stopping significantly before convergence.__"*

__Takeaways__
* __"Language modeling performance improves smoothly and predictably as we appropriately scale up model size, data, 
  and compute."__
* Language model (Transformer) __performance depends most strongly on scale__: model size, dataset size, and compute 
  resources.
* Performance has a smooth [power-law](https://en.wikipedia.org/wiki/Power_law) relationship with each scale factor 
  (implying diminishing marginal returns with increasing scale).
* Performance has "very weak dependence on many architectural and optimization hyper-parameters."
* "Our results strongly suggest that larger models will continue to perform better, and will also be much more
  sample efficient than has been previously appreciated. __Big models may be more important than big data__."

__CAVEAT: [Not everyone agrees with these scaling laws.](https://www.lesswrong.com/posts/midXmMb2Xg37F2Kgn/new-scaling-laws-for-large-language-models)__

# CS231n Convolutional Neural Networks for Visual Recognition
[(2024 Course Notes)](https://cs231n.github.io/)

"This course is a deep dive into the details of deep learning architectures with a focus on learning end-to-end models for these tasks, particularly image classification."

__Takeaways__

The course notes contain excellent treatment of the fundamentals of neural networks, including:
* [The neuron model and NN architectures](https://cs231n.github.io/neural-networks-1/)
* [Backpropagation](https://cs231n.github.io/optimization-2/)
* [Gradient Descent for learning](https://cs231n.github.io/optimization-1/)