# Building a Deep Auto-Encoder with PyTorch

Auto-encoders are an unsuprevised way to learn a low-dimension representation of your data. 

Auto-encoders are a powerful technique for... 
- Dimensionality reduction
- image denoising
- Feature learning, e.g. pre-training for supervised learning
- Variational Auto-Encoders can be used to generate new data that is similar to the training data.

In this post, I'll go over what they are, set up an auto-encoder using a deep convolutional neural network in PyTorch, and train it on some fun image data sets.


## Auto-encoders
Auto-encoders are a type of unsupervised neural network that learns an efficient coding of input data. It's helpful to think about this like image compression: a good compression function compresses a large image to a smaller size, but does it in a way that lets us later recover a good approximation of the original image. An auto-encoder is a neural network that simultaneously learns the compression and decompression functions — or, more traditionally, the encoder and the decoder. 

- **Encoder**: Maps the input data to a lower-dimensional representation (latent space).
- **Decoder**: Reconstructs the input data from this lower-dimensional representation.

The auto-encoder NN has a simple structure designed for compression: it's basically any network that "pinches down" in the middle to form a "bottleneck" between the encoder and the decoder. This description leaves some ambiguity about the number of layers, the type of operations used, and the size of the "bottleneck" layer. We'll get to those model selection issues, but for now, focus on this image:

[TODO: image of auto-encoder network with image as input and output]


## A Deep Auto-Encoder in PyTorch

### How many layers?
- Universal Approximator Theorem. "One major advantage of nontrivial depth is that the universal approximator theorem guarantees that a feed-forward neural network with at least one hidden
layer can represent an approximation of any function (within a broad class) to an
arbitrary degree of accuracy, provided that it has enough hidden units."
- Additional layers should allow the flexibility to enforce additional constraints, such as code sparsity.

### Size of latent code?
Not sure how to choose this... The code should learn interesting features of the data. Is there a way to see what it has learned?

### Activation functions
- ReLU: why? It should help with maintaining healthy gradients. Sigmoidal activation functions should have issues with vanishing gradients.

### Regularization?
- What are we regularizing? Model weights? The codeword?
- L1 vs L2 Regularization: L1 should produce more zeros
    - Image of loss intersecting L1 and L2 norms


```python
# MNIST Autoencoder model. 784 -> 128 -> 64 -> [12] -> 64 -> 128 -> 784
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            # Layer 1: 784 -> 128
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),

            # Layer 2: 128 -> 64
            nn.Linear(128, 64),
            nn.ReLU(True),

            # Layer 3: 64 -> 12
            nn.Linear(64, 12),
            # TODO: ReLU?
        )
        self.decoder = nn.Sequential(
            # Layer 4: 12 -> 64
            nn.Linear(12, 64),
            nn.ReLU(True),

            # Layer 5: 64 -> 128
            nn.Linear(64, 128),
            nn.ReLU(True),

            # Layer 6: 128 -> 784
            nn.Linear(128, 28 * 28),
            nn.Tanh()
        )
        print(self)

    def forward(self, x: Tensor) -> (Tensor, Tensor):
        """
        Forward pass of the autoencoder model.

        :param x: input tensor of shape (batch_size, 28 * 28)
        :return: Recovered tensor, code tensor
        """
        code = self.encoder(x)
        recovered = self.decoder(code)
        return recovered, code


```

## Training and Evaluation

### Data set
- Choice of data set
- Data normalization

### Training with Stochastic Gradient Descent
- Loss function
- How are weights initialized?
    - There are a few strategies for initializing weights. Xavier, He, others?
- Adam Optimizer. PyTorch uses the defaults from the paper
- Plot training error
- Visually investigate gradients
- What other metrics can I see to check that training is proceeding correctly?

```python

    loss_fn: nn.MSELoss = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    # Codeword regularization. Zero is no regularization; increase for more regularization.
    lambda_l2 = 1e-5
    
    # Number of passes over the training data.
    num_epochs = 10
    for epoch in range(num_epochs):
        for img_batch, _labels in train_loader:
            img_batch = img_batch.cuda()  # Move batch to GPU

            # Forward pass
            output, code = autoencoder(img_batch)
            loss = loss_fn(output, img_batch)

            # L2 regularization on the code vector
            l2_reg = lambda_l2 * torch.norm(code, p=2)
            loss += l2_reg

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update parameters

```

- Model Selection and Hyperparameters?
