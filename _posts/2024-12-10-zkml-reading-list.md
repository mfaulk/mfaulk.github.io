# ZKML Reading Notes

A chronological survey of the zkML field, through 2024.

**2017 [SafetyNets: Verifiable Execution of Deep Neural Networks on an Untrusted Cloud](https://proceedings.neurips.cc/paper/2017/hash/6048ff4e8cb07aa60b6777b6f7384d52-Abstract.html)**

* A restricted class of neural nets that can be represented as arithmetic circuits over finite fields.  
* Applies [Thaler's](https://link.springer.com/chapter/10.1007/978-3-642-40084-1_5) IP protocols for “regular” arithmetic circuits and matrix multiplication.  
* Evaluates on a 2-layer CNN (MNIST) and a [4-layer fully-connected network](https://proceedings.neurips.cc/paper/2014/hash/ea8fcd92d59581717e06eb187f10666d-Abstract.html) (TIMIT)

**2018 [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993)**

* Thoughtful approach to ML model transparency  
* Model cards are a template for documenting ML models  
* Model cards contain benchmarks on evaluation datasets

**2020 [On Polynomial Approximations for Privacy-Preserving and Verifiable ReLU Networks](https://arxiv.org/abs/2011.05530)**

* Replaces ReLU activation function with a degree-2 polynomial, f(x) \= x^2 \+ x  
* Evaluates on CNNs, small decrease in accuracy relative to ReLU

**2020 [Zero Knowledge Proofs for Decision Tree Predictions and Accuracy](https://dl.acm.org/doi/abs/10.1145/3372297.3417278)**

* Setup: Prover commits to a decision tree in linear time.  
* Proves validity of a decision path in time proportional to prediction path length.  
* Uses Aurora

**2021 [Mystique: Efficient Conversions for Zero-Knowledge Proofs with Applications to Machine Learning](https://www.usenix.org/system/files/sec21-weng.pdf)**

**2021 [A Greedy Algorithm for Quantizing Neural Networks](https://jmlr.csail.mit.edu/papers/volume22/20-1233/20-1233.pdf)**

* Quantizes the weights of a pre-trained neural network  
* Proposes Greedy Path-Following Quantization (GFPQ) algorithm

**2021 VeriDL: Integrity Verification of Outsourced Deep Learning Services**

**2021 [ZEN: An Optimizing Compiler for Verifiable, Zero-Knowledge Neural Network Inferences](https://eprint.iacr.org/2021/087)**

* Compiles floating-point pyTorch models to R1CS constraints  
* Quantization algorithm with R1CS-friendly optimizations  
* SIMD-style optimization encodes multiple 8-bit integers in a large finite field element

**2021 [zkCNN: Zero Knowledge Proofs for Convolutional Neural Network Predictions and Accuracy](https://eprint.iacr.org/2021/673)**

* [zkCNN repo](https://github.com/TAMUCrypto/zkCNN)  
* Sumcheck protocol for FFT and 2-D convolutions  
* Based on refinement of GKR to arbitrary circuits  
* Efficient gadgets for ReLU activation and max pooling  
* Evaluated on 15M parameter network

**2022 [pvCNN: Privacy-Preserving and Verifiable Convolutional Neural Network Testing](https://arxiv.org/abs/2201.09186)**

* Privacy-preserving CNN testing via FHE and collaborative inference,  
* zk-SNARK for the privacy-preserving CNN testing  
* Quadratic Matrix Program (QMP) arithmetic circuit for 2D convolution

**2022 [Scaling up Trustless DNN Inference with Zero-Knowledge Proofs](https://arxiv.org/abs/2210.08674)**

* zk-SNARK (Halo2) for convolutional NN  
* 8-bit quantized deep neural networks  
* Custom gates for linear layers  
* Lookup arguments for non-linearities  
* Evaluated on ImageNet (avg. resolution: 469 × 387 pixels)

**2022 [ZK-IMG: Attested Images via Zero-Knowledge Proofs to Fight Disinformation](https://arxiv.org/abs/2211.04775)**

* Compiles high-level image transformations to zk-SNARKs.   
* Attests to (multiple) image transformations while hiding the pre-transformed image

**2023 [Honey I SNARKED the GPT](https://blog.ezkl.xyz/post/nanogpt/)**

* Proving nanoGPT in [EZKL](https://ezkl.xyz/) (Halo2-KZG)  
* nanoGPT expressed in [ONNX](https://github.com/onnx/onnx) format  
* Expresses many ONNX operations as a single operation (Einstein summation)

**2023 The Cost of Intelligence: Proving Machine Learning Inference with Zero-Knowledge**

* Modulus Labs  
* Benchmarks several ZKPs for NNs: Groth16, Gemini, Winterfell, Halo2, Plonky2, zkCNN  
* Good survey of ZKML papers

**2023 [zkDL: Efficient Zero-Knowledge Proofs of Deep Learning Training](https://arxiv.org/pdf/2307.16273)**

* **zkReLU**: a specialized proof for the ReLU activation and its backpropagation

**2024 [vCNN: Verifiable convolutional neural network based on zk-snarks](https://eprint.iacr.org/2020/584)**

* Quadratic Polynomial Program (QPP) SNARK for convolutions  
* Quadratic Arithmetic Program (QAP) SNARK for pooling and activation layers  
* Commit-and-Prove SNARK for connecting adjacent layers  
* Significant performance improvements on MNIST and [VGG16](https://medium.com/@mygreatlearning/everything-you-need-to-know-about-vgg16-7315defb5918) image datasets

**2024 [BatchZK: A Fully Pipelined GPU-Accelerated System for Batch Generation of Zero-Knowledge Proofs](https://eprint.iacr.org/2024/1862.pdf)**

* Evaluated on ZKML application (convolutional NN), 9.5 proofs per second  
* Pipelined Modules for Sum-Check, Merkle tree, and [linear-time encoder](https://dl.acm.org/doi/abs/10.1145/780542.780562?casa_token=QWoPIox2EfUAAAAA:icy8XrWk4sFk6DvfjzL9kXGllRcdAy2Tar00T33RqnqYHwU5Z_Z0NvxGrq0YojRVrEuFrH49zlOElQ)


**2024 [Sparsity-Aware Protocol for ZK-friendly ML Models: Shedding Lights on Practical ZKML](https://eprint.iacr.org/2024/1018.pdf)**

* SpaGKR sparsity-aware ZML framework  
* Efficient results with ternary networks  
* Sparsity-Aware Sum-Check  
* Evaluated on…

**2024 [An Efficient and Extensible Zero-knowledge Proof Framework for Neural Networks](https://eprint.iacr.org/2024/703)**

* Improves proofs for non-linear layers by converting complex non-linear relations into range and exponent relations  
* Efficient proofs for range and exponent lookups  
* Evaluation on CNN and Transformer (GPT-2)

**2024 [zkLLM: Zero Knowledge Proofs for Large Language Models](https://arxiv.org/abs/2404.16109)**

* Parallelized lookup argument for non-arithmetic tensor operations  
* zkAttn ZKP for the attention mechanism  
* Evaluated on **13B-parameter LLM**  
* CUDA implementation

**2024 [Scaling intelligence: Verifiable decision forest inference with remainder](https://github.com/Modulus-Labs/Papers/blob/master/remainder-paper.pdf)**

* Modulus Labs  
* Verifiable decision forest inference  
* Remainder: GKR prover with Ligero polynomial commitment scheme

**2024 [ZKML: An Optimizing System for ML Inference in Zero-Knowledge Proofs](https://dl.acm.org/doi/abs/10.1145/3627703.3650088)**

* Compiles from TensorFlow to optimized Halo2 circuits  
* Gadgets for ML operations (dot products, softmax, pointwise non-linearities)  
* Supports CNNs, LSTMs, Transformers, MLPs, diffusion models


**2024** [**ZENO: A Type-based Optimization Framework for Zero Knowledge Neural Network Inference**](https://dl.acm.org/doi/abs/10.1145/3617232.3624852)

* ZENO: ZEro knowledge Neural network Optimizer  
* ZENO language maintains high-level semantics to enable compiler optimizations  
* Knit encoding combines multiple low-bit (e.g.,8-bit) scalars into a high-bit (e.g., 254-bit) scalar  
* Reduces VGG16 proof from 6 minutes to 48 seconds

**2024 [Zero-Knowledge Proofs of Training for Deep Neural Networks](https://eprint.iacr.org/2024/162.pdf)**

* GKR-style (sumcheck-based) proof system for gradient-descent  
* Recursively compose proofs across multiple iterations to attain succinctness

# Quantization

Quantization is the conversion of model weights to a more compact format. Models are commonly trained with 16-bit or 32-bit floating point numbers. Converting these weights to small (e.g. 4-bit) integers or binary values can significantly reduce the size of models and the cost of inference. Converting weights to integers in a finite field allows for more practical implementation in zero knowledge proofs.

[**2016 Ternary Weight Networks**](https://arxiv.org/abs/1605.04711)

* {-1, 0, 1}  
* Almost as small as binary networks, but with better performance

**2016 [Ternary Neural Networks for Resource-Efficient AI Applications](https://arxiv.org/abs/1609.00222)**

[**2018 A Survey on Methods and Theories of Quantized Neural Networks**](https://arxiv.org/abs/1808.04752)

**2018 Quantized Neural Networks: Training Neural Networks with Low Precision Weights and Activations**

**2018 LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks**

[**2018 Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference**](https://openaccess.thecvf.com/content_cvpr_2018/html/Jacob_Quantization_and_Training_CVPR_2018_paper.html)

* Quantizes weights and activations as 8-bit integers  
* Inference with integer-only arithmetic  
* Quantization-aware training framework

**2022 [Post-training Quantization for Neural Networks with Provable Guarantees](https://arxiv.org/abs/2201.11113)**

* Modifies GFPQ to promote sparsity of weights  
* Quantization error analysis

**2023 [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)**

* 1-bit Transformer architecture for LLMs  
* Competitive with 8-bit quantization an FP16

**2024 [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)**

* Every parameter (weight) of the LLM is ternary {-1, 0, 1}.   
* Comparable performance with full-precision (FP16) model

[**2024 Scaling Trust in the Age of Artificial Intelligence**](https://medium.com/polyhedra-network/scaling-trust-in-the-age-of-artificial-intelligence-cab076d6164e)  

# GKR and Sum-Check

[**The Unreasonable Power of the Sum-Check Protocol**](https://people.cs.georgetown.edu/jthaler/blogpost.pdf)

**2012 [Practical verified computation with streaming interactive proofs](https://dl.acm.org/doi/abs/10.1145/2090236.2090245?casa_token=-iZax0xdacoAAAAA:z0IpaMcPHJ78K6fTCzSKzZbtAKa_xEV6yZPZxCvq5EtZEBLExWyctkciM6Nsk9TL0rIB0AVYtahn0w)**

* prover time was improved to quasi-linear $(O(\|C\| log \|C\|))$

**2013 Time-Optimal Interactive Proofs for Circuit Evaluation**

* For "regular" circuits, give a prover that is linear time in circuit size  
* IP for Matrix multiplication

**2017 [vSQL: Verifying Arbitrary SQL Queries over Dynamic Outsourced Databases](https://ieeexplore.ieee.org/abstract/document/7958614?casa_token=38CWc72LuxsAAAAA:Lelbt3M9aIOpNrXli9S-DxVuUBJcwlYbP2qBSfObBO_I4UPS3aPPKIyY7V0R7nWxlgrTbpWfkw)**

* GKR \+ polynomial commitment scheme \= ZK argument

**2019 [Libra: Succinct Zero-Knowledge Proofs with Optimal Prover Computation](https://link.springer.com/chapter/10.1007/978-3-030-26954-8_24)**

* Prover time to strictly **linear** $(O(\|C\|))$ for layered arithmetic circuits without assuming any structures.

**2021 [Doubly Efficient Interactive Proofs for General Arithmetic Circuits with Linear Prover Time](https://eprint.iacr.org/2020/1247.pdf)**

* "Generalizes the GKR protocol to work on arbitrary arithmetic circuits efficiently for the first time. For a general circuit of size $\|C\|$ and depth d, the prover time is $O(\|C\|)$, the same as the original GKR protocol on a layered circuit with the same size."

**Hyperplonk, Brakedown, Orion**

* Combine Sum-Check with multilinear polynomial commitments to build SNARKs

**2023 [Modular Sumcheck Proofs with Applications to Machine Learning and Image Processing](https://dl.acm.org/doi/abs/10.1145/3576915.3623160)**

* “A new information-theoretic primitive called Verifiable Evaluation Scheme on Fingerprinted Data (VE) that captures the properties of diverse sumcheck-based interactive proofs, including the well-established GKR protocol”

**2024 [Sparsity-Aware Protocol for ZK-friendly ML Models: Shedding Lights on Practical ZKML](https://eprint.iacr.org/2024/1018.pdf)**

* SpaGKR: GKR protocol exploiting sparse parameters  
* SpaSum: Sparsity-aware Sum-Check protocol

