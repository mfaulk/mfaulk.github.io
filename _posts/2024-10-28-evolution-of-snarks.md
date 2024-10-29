## The Evolution of SNARKs: Interactive Proofs to Groth16

This post is a high-level recap of the foundational ideas that led to the first practical zk-SNARKs. 
There's a lot to cover, and so this post focuses on the developments leading up to the introduction of the Groth16 SNARK. Groth16 feels like a good milestone: it was one of the first SNARKs used in practice, and its small proofs and fast verification are still, largely, unmatched today.

zk-SNARKs are "zero-knowledge Succinct Non-Interactive Arguments of Knowledge". There's a lot to unpack there, so we'll take it piece by piece.

# New ways of proving: Interactive Proofs and Zero Knowledge

Interactive Proofs introduced by [GMR'85] and [Babai 85]
- [GMR 85/89] The knowledge complexity of interactive proof-systems
- [Babai 85] Trading Group Theory for Randomness

In the late 1980s, two pivotal papers by Shafi Goldwasser, Silvio Micali, and Charles Rackoff (1989) and by László Babai (1988) advanced the concept of *interactive proofs* (IPs), which extended the NP proof system to allow interaction and randomness. Goldwasser, Micali, and Rackoff introduced an interactive proof system allowing the verifier to make private random coin tosses. Babai’s approach, known as Arthur-Merlin games, instead formalized a model where coin tosses are public. These groundbreaking contributions were recognized jointly with the [1993 Gödel Prize](https://sigact.org/prizes/g%C3%B6del/1993.html).

*"Both works suggested models for proving and checking proofs that were different from the standard model of writing down a proof, and checking it step by step. The new models viewed theorem proving as a discussion between a prover and a verifier. The verifier may ask the prover questions, and the questions may be randomized. If the verifier’s impression of the validity of the proof is correct with probability, say, 0.99, they said, this is good enough."* — [The Tale of the PCP Theorem](https://www.cs.utexas.edu/~danama/XRDS.pdf)

*"Independently of GMR, and published in the same STOC ’85, was a paper of Babai, Trading Group Theory for Randomness. This paper was later published in journal version as Arthur-Merlin Games: A Randomized Proof System, and a Hierarchy of Complexity Classes, with Moran as a coauthor; in fact, this paper shared the first ever Godel Prize with GMR"* — [A history of the PCP Theorem](https://courses.cs.washington.edu/courses/cse533/05au/pcp-history.pdf)


# IPs: Randomness and Interaction
Relaxing the "classical" notion of proofs (as represented by NP) to allow for randomness and rounds of interaction between the prover and the verifier yields surprising power.

"Interactive proofs (IPs) [GMR'85] allow proof-verification to be randomized and interactive, which seemingly confers them much more power than their deterministic (and non-interactive) counterparts" — [A PCP Theorem for Interactive Proofs and Applications](https://eprint.iacr.org/2021/915)

It turns out that IP = PSPACE [Shamir '90], which formalizes that interactive proofs are far more powerful than classical static (i.e, NP) proofs.

Both randomness and interaction are required.

# Non-Interaction and the Common Reference String

*"Blum, Feldman and Micali [BFM88] extended the notion to non-interactive zero-knowledge (NIZK) proofs in the common reference string model. NIZK proofs are useful in the
construction of non-interactive cryptographic schemes, e.g., digital signatures and CCAsecure public key encryption."* -- [On the Size of Pairing-based Non-interactive Arguments](https://eprint.iacr.org/2016/260.pdf)

*"First NIZK Schemes Blum et al. first study the non-interactive zero-knowledge proof system and present the common reference string model that is generally applied at present [BFM88, DMP90]. This first construction of [BFM88] is a bounded NIZK proof system, meaning that for different statements in NP language, the proof system has to use different CRSs and the length of the statement is controlled by the length of CRS. Later, Blum et al. [DMP90] presented a more general (multi-theorem) NIZK proof system for 3SAT by improving the previous one, which allows to prove many statements with the same CRS. Both [BFM88] and [DMP90] based their NIZK systems on certain number-theoretic assumptions (specifically, the hardness of deciding quadratic residues modulo a composite number). Feige, Lapidot, and Shamir [FLS90] showed later how to construct computational NIZK proofs based on any trapdoor permutation."* -- [zk-SNARKs: A Gentle Introduction](https://www.di.ens.fr/~nitulesc/files/Survey-SNARKs.pdf#page=7.11)


# Arguments are Succinct, Proofs (probably) aren't

Proofs vs Arguments... Intuitively, computationally bounding the prover means that cryptographic techniques become effective.

Argument systems introduced by [Brassard, Chaum, Crepeau 1988]

Arguments can be succinct; proofs (probably) can't be succinct. Statistically-sound proof systems are unlikely to allow for significant improvements in communication, i.e., the prover needs to communicate roughly as much information as the size of the witness. Argument systems can be succinct, i.e., sublinear (or constant) communication complexity.

*"We know that if statistical soundness is required then any non-trivial savings would cause unlikely complexity-theoretic collapses (see, e.g., [BHZ87, GH98, GVW02, Wee05]). However, if we settle for proof systems with only computational soundness (also known as interactive arguments [BCC88]) then significant savings can be made"* —  [BCCT'11](https://eprint.iacr.org/2011/443.pdf)

*"Quite remarkably, it is possible to break the efficiency barriers by relaxing the soundness requirement to hold only with respect to computationally bounded provers. Such interactive proof systems, introduced by Brassard et al. 1988, are referred to as arguments."* — [IKO'07]

*"Interactive proof systems, as defined by Goldwasser, Micali, and Rackoff..., relax classical proofs by allowing randomness and interaction, but still require soundness to hold with respect to computationally unbounded provers. Unfortunately, proof systems of this type do not seem to be helpful in the current context either. In particular, the existence of interactive proof systems for NP in which the prover is laconic, in the sense that the total length of the messages it sends to the verifier is much smaller than the length of the witness, would imply a nontrivial nondeterministic simulation for coNP ... A similar limitation holds for proof systems in which the verifier’s time complexity is smaller than the length of the witness."*
-- [Efficient Arguments without Short PCPs](https://web.cs.ucla.edu/~rafail/PUBLIC/79.pdf#page=2.14)

The first zero-knowledge argument for an NP-complete problem (Graph 3-Coloring) was given by Goldreich, Micali, and Wigderson [GMW91].

# PCPs: Probabilistically Checkable Proofs
The development of IPs, in turn, led to probabilistically checkable proofs (PCPs). 
IPs also led to hardness of approximation results, see The Tale of the PCP Theorem. I think this material ties into a number of complexity theory results (IP = PSPACE, and MIP = NEXP), hardness of approximations. 

- [Babai 91] Checking computations in polylogarithmic time.
- [AS'92] Probabilistic checking of proofs: a new characterization of NP.
- [ALM+'92] Proof verification and the hardness of approximation problems. (Original statement of PCP theorem?)


*"The PCP model was introduced by Fortnow, Rompel, and Sipser [FRS88], who referred to it as the “oracle” model (we used this terminology in Lemma 8.2). The term Probabilistically Checkable Proofs was coined by Arora and Safra [AS98]."*
— [Proofs, Arguments, and Zero Knowledge](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf#page=135)

*"The celebrated PCP Theorem states that any language in NP can be decided via a verifier that reads O(1) bits from a polynomially long proof."*
-- [A PCP Theorem for Interactive Proofs and Applications](https://eprint.iacr.org/2021/915)

*"A PCP is a redundant encoding of a classical proof that can be probabilistically verified by querying a small number of bits. The celebrated PCP Theorm [2, 1] asserts that every classical proof can be turned into a polynomial-size PCP which can be verified using a constant number of queries (up to a small constant soundness error)"*
— [IKO'07]

*"The study of PCPs was initiated by Babai et al. [3] and by Feige et al. [16] with a very different motivation. While in [3] PCPs were used as a means of speeding up verification of proofs and computation, their study in [16], as well as in most subsequent works in the area, was mainly driven by the exciting connection between PCPs and hardness of approximation"*
— [IKO'07]

# The Kilian-Micali Paradigm: PCP + Merkle Tree => SNARG 

- [Kilian '92] A note on efficient zero-knowledge proofs and arguments
- [Micali '94] CS proofs, maybe cite the 2000 version
- Succinct non-interactive argument systems
- Kilian-Micali transformation compiles a PCP into a (non-preprecessing) SNARG
- [Kilian 95] Improved efficient arguments: Preliminary version
- [BG 02] Universal arguments and their applications

These argument systems combine a short (polynomial size) PCP with cryptographic hashing. Merkle tree is a succinct commitment to the PCP string that can be efficiently opened for queries.

*"The origins of SNARKs date back to the work of Kilian and Micali based on the PCP theorem."*
https://eprint.iacr.org/2022/1355.pdf

*"Kilian [Kilian '92] gave the first succinct four-round interactive argument system for NP based on collision-resistant hash functions and probabilistically checkable proofs (PCPs)"*

*"Under cryptographic assumptions, Kilian [23] constructed an interactive argument system for NP with a polylogarithmic communication complexity. Micali [26] suggested a non-interactive implementation of argument systems, termed CS proofs, whose soundness was proved in the Random Oracle Mode"*
— https://web.cs.ucla.edu/~rafail/PUBLIC/79.pdf#page=2.14

*"The amount of communication is an important performance parameter for zeroknowledge proofs. Kilian [Kil92] gave the first sublinear communication zero-knowledge argument that sends fewer bits than the size of the statement to be proved. Micali [Mic00] proposed sublinear size arguments by letting the prover in a communication efficient argument compute the verifier’s challenges using a cryptographic function, and as remarked in Kilian [Kil95] this leads to sublinear size NIZK proofs when the interactive argument is public coin and zero-knowledge."*
-- [On the Size of Pairing-based Non-interactive Arguments](https://eprint.iacr.org/2016/260.pdf)


*"SNARKs achieve non-interactivity through the Fiat-Shamir heuristic [20], which substitutes the verifier’s random challenges with outputs from a collision-resistant hash function. This modification not only simplifies the verification process but also makes the proofs publicly verifiable, enhancing both the practicality and security of SNARKs in broad applications."*
https://arxiv.org/pdf/2408.00243v1#page=16

*"The key idea that was used in [Kilian '92] for getting around these difficulties is to first require the prover to send a succinct commitment to the PCP string, using a suitable cryptographic technique, and then allow the verifier to open a small, randomly chosen, subset of the committed bits. In order to efficiently support the required functionality, a special type of cryptographic commitment should be used. Specifically, the commitment scheme should be: (1) computationally binding (but not necessarily hiding); (2) succinct, in the sense that it involves a small amount of communication between the prover and the verifier; and (3) support efficient local decommitment, in the sense that following the commitment phase any bit of the committed string can be opened (or “decommitted”) very efficiently, in particular without opening the entire string. An implementation of such a commitment scheme can be based on any collision-resistant hash function by using Merkle’s tree hashing technique [25]. Roughly speaking, the commitment process proceeds by first viewing the bits of the proof as leaves of a complete binary tree, and then using the hash function to compute the label of each node from the two labels of its children. The succinct commitment string is taken as the label of the root. To open a selected bit of the proof, the labels of all nodes on a path from the corresponding leaf to the root, along with their siblings, are revealed. Thus, the number of atomic cryptographic operations required to open a single bit of the proof is logarithmic in the proof length."* — https://web.cs.ucla.edu/~rafail/PUBLIC/79.pdf

*"Indeed, using collision-resistant hash functions (CRHs), Kilian [Kil92] shows a four message interactive argument for NP: the prover first uses a Merkle hash tree to bind itself to a poly-size
PCP (Probabilistically Checkable Proof) string for the statement, and then answers the PCP verifier’s queries while demonstrating consistency with the Merkle tree. This way, membership of an instance y in an NP language L can be verified in time that is bounded by p(k, |y|, log t), where t is the time to evaluate the NP verification relation for L on input y, p is a fixed polynomial independent of L, and k is a security parameter that determines the soundness error. Following tradition, we call such argument systems succinct."* — https://eprint.iacr.org/2011/443.pdf

"Barak and Goldreich [BG02] showed that Kilian’s argument system is not only sound, but in fact
knowledge-sound. This assertion assumes that the underlying PCP that the argument system is based on also satisfies an analogous knowledge-soundness property, meaning that given a convincing PCP proof π, one can efficiently compute a witness"
https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf#page=138

# [IKO'07] Efficient Arguments without Short PCPs, Ishai, Kushilevitz, and Ostrovsky

[IKO '07] introduces linear PCPs, which are PCPs with the additional structure that honest proofs are linear functions. This structure allows the prover to commit to a proof without fully materializing it. That's important because their linear PCP for a circuit of size s has length F^O(s^2). Unfortunately, the verifier is still required to make queries of size O(s^2).

*"Prior to about 2007, SNARKs were primarily designed via the Kilian–Micali paradigm, of taking a “short” probabilistically checkable proof (PCP) and “cryptographically compiling” it into a succinct argument via Merkle hashing and the Fiat-Shamir transformation. Unfortunately, short PCPs are complicated and impractical, even today. This paper (IKO) showed how to use homomorphic encryption to obtain succinct (non-transparent) interactive arguments from “long but structured” PCPs. These can be much simpler than short PCPs, and the resulting SNARKs have much shorter proofs and faster verification. These arguments were first recognized as having the potential for practical efficiency, and refined and implemented, in Pepper. Unfortunately, IKO’s arguments have a quadratic-time prover and quadratic-size structured reference string, so they do not scale to large computations."*
— https://a16zcrypto.com/posts/article/zero-knowledge-canon/#section--6

# GKR 08
Maybe talk about GKR protocol, first IP protocol with a polynomial time prover and quasilinear time verifier.

# Groth-Sahai Proofs
"Groth, Ostrovsky and Sahai [GOS12,GOS06,Gro06,GS12] introduced pairing-based NIZK proofs, yielding the first linear size proofs based on standard assumptions." https://eprint.iacr.org/2016/260.pdf#page=2

# [Groth '10] Short pairing-based non-interactive zero-knowledge arguments
- First constant-size NIZK argument
- Uses pairings
- Succinct NIZK without the PCP theorem
- Verifier runtime is polynomial (not succinct)

Groth10 combines pairing-based NIZK proofs with ideas from interactive argument systems to give the first constant-size NIZK arguments

"Two other non-interactive arguments for NP, based on more concise knowledge assumptions, are due to Mie [Mie08] and Groth [Gro10]. However, neither of these protocols is succinct. Specifically, in both protocols the verifier’s runtime is polynomially related to the time needed to directly verify the NP witness." — https://eprint.iacr.org/2011/443.pdf

"Groth, Ostrovsky and Sahai [GOS12,GOS06,Gro06,GS12] introduced pairing-based
NIZK proofs, yielding the first linear size proofs based on standard assumptions. Groth [Gro10]
combined these techniques with ideas from interactive zero-knowledge arguments [Gro09]
to give the first constant size NIZK arguments. Lipmaa [Lip12] used an alternative construction based on progression-free sets to reduce the size of the common reference
string... Groth’s constant size NIZK argument is based on constructing a set of polynomial equations and using pairings to efficiently verify these equations."
https://eprint.iacr.org/2016/260.pdf


# [Lipmaa '12] Progression-free sets and sublinear pairing-based non-interactive zeroknowledge arguments.
Improves on Groth10 by reducing size of the common reference string from .... to ...

"Groth, Ostrovsky and Sahai [GOS12,GOS06,Gro06,GS12] introduced pairing-based
NIZK proofs, yielding the first linear size proofs based on standard assumptions. Groth [Gro10]
combined these techniques with ideas from interactive zero-knowledge arguments [Gro09]
to give the first constant size NIZK arguments. Lipmaa [Lip12] used an alternative construction based on progression-free sets to reduce the size of the common reference string."


# [BCI+13] Succinct non-interactive arguments via linear interactive proofs

This looks like an important step towards Groth16

# [GGPR 13]
- succinct NIZK constructions that use QAPs
- Improves on IKO07 by reducing size of linear PCP
- a pairing-based NIZK argument with a common reference string size proportional to the size of the statement and witness
- Quadratic Span Programs (QSPs) for proving boolean circuit satisfiability
- Ouadratic Arithmetic Programs (QAPs) for proving arithmetic circuit satisfiability.


"Groth’s constant size NIZK argument is based on constructing a set of polynomial equations and using pairings to efficiently verify these equations. Gennaro, Gentry, Parno and Raykova [GGPR13] found an insightful construction of polynomial equations based on Lagrange interpolation polynomials yielding a pairing-based NIZK argument with a common reference string size proportional to the size of the statement and witness. They gave two types of polynomial equations: quadratic span programs for proving boolean circuit satisfiability and quadratic arithmetic programs for proving arithmetic circuit satisfiability."
— https://eprint.iacr.org/2016/260.pdf

"This breakthrough paper (GGPR) reduced the prover costs of IKO’s approach from quadratic in the size of the circuit to quasilinear. Building on earlier work of Groth and Lipmaa, it also gave SNARKs via pairing-based cryptography, rather than interactive arguments as in IKO. It described its SNARKs in the context of what is now referred to as rank-1 constraint satisfaction (R1CS) problems, a generalization of arithmetic circuit-satisfiability.
This paper also served as the theoretical foundation of the first SNARKs to see commercial deployment (e.g., in ZCash) and directly led to SNARKs that remain popular today (e.g., Groth16). Initial implementations of GGPR’s arguments came in Zaatar and Pinocchio, and later variants include SNARKs for C and BCTV. BCIOP gave a general framework that elucidates these linear-PCPs-to-SNARK transformations (see also Appendix A of Zaatar) and describes various instantiations thereof."
— https://a16zcrypto.com/posts/article/zero-knowledge-canon/#section--6



# [Groth16] On the Size of Pairing-based Non-interactive Arguments

- Shortest proof size (3 elements)
- Fast verifier (bilinear pairing)
- Converts R1CS into QAP via interpolation (prover does FFT)
- R1CS isn't explicitly mentioned, but QAP is the same as R1CS except that it uses polynomials instead of dot products.
- Naively, this is done via Lagrange interpolation. In practice, interpolation is done via the NTT (FFT over finite fields) to reduce complexity from O(n^2) to O(n lg n)
- Converts QAP to a NIZK
- Constructs a NILP argument for QAPs
- Converts it to a pairing-based NIZK
- Non-universal: Requires Circuit-specific trusted setup
- Slow prover
- Similar techniques to KZG commitments?

Good overviews
- The Mathematical Mechanics Behind the Groth16 Zero-knowledge Proving
- Remco Bloemen gives a good overview of Groth16

In a nutshell: Groth16 starts by creating an R1CS instance of an arithmetic circuit. The R1CS instance is transformed into a Quadratic Arithmetic Program (QAP), which encodes the arithmetic cornstraints as polynomial vectors. The QAP consists of three polynomials (from the A, B, C matrices) and a solution polynomial (derived from the solution vector z). These polynomials are computed via Lagrange interpolation (probably interpolation via FFT). The QAP is verified by showing existence of a low-degree quotient polynomial. The next step is converting the QAP into a NIZK argument

"This paper, colloquially referred to as Groth16, presented a refinement of GGPR’s SNARKs that achieves state-of-the-art concrete verification costs even today (proofs are 3 group elements, and verification is dominated by three pairing operations, at least assuming the public input is short). Security is proved in the generic group model."
https://a16zcrypto.com/posts/article/zero-knowledge-canon/#section--6

"Succinct NIZK. We construct a NIZK argument for arithmetic circuit satisfiability
where a proof consists of only 3 group elements. In addition to being small, the proof
is also easy to verify. The verifier just needs to compute a number of exponentiations
proportional to the statement size and check a single pairing product equation, which
only has 3 pairings. Our construction can be instantiated with any type of pairings
including Type III pairings, which are the most efficient pairings."

"All pairing-based SNARKs in the literature follow a common paradigm
where the prover computes a number of group elements using generic group operations
and the verifier checks the proof using a number of pairing product equations. Bitansky
et al. [BCI+13] formalize this paradigm through the definition of linear interactive proofs
(LIPs). A linear interactive proof works over a finite field and the prover’s and verifier’s
messages consist of vectors of field elements. It furthermore requires that the prover
computes her messages using only linear operations. Once we have an approriate 2-move
LIP, it can be compiled into a SNARK by executing the equations “in the exponent”
using pairing-based cryptography. One source of our efficiency gain is that we design
a LIP system for arithmetic circuits where the prover only sends 3 field elements. In
comparison, the quadratic arithmetic programs by [GGPR13,PHGR13] correspond to
LIPs where the prover sends 4 field elements."
https://eprint.iacr.org/2016/260.pdf#page=5



"In Groth16, the proof generation process involves creating a quadratic arithmetic program (QAP) that represents the computation being proved. The QAP is then transformed using fast Fourier transform (FFT) algorithms to create a succinct representation of the computation that can be used in the proof.

The QAP represents the computation being proved as a system of quadratic equations, where the variables are the inputs and intermediate values of the computation. The QAP is then transformed using FFT algorithms to create a polynomial commitment that represents the computation in a succinct form. This polynomial commitment is then used in the proof to verify the correctness of the computation."
https://medium.com/degate/behind-degates-innovation-the-integral-role-of-groth16-dd498f5b5fa7


# References

[GMR 85/89] The knowledge complexity of interactive proof-systems

[Babai 85] Trading Group Theory for Randomness

[Brassard, Chaum, Crepeau 88] Minimum Disclosure Proofs of Knowledge

[BFM 88] Non-interactive zero-knowledge and its applications
- NIZKs, Setup produces circuit-specific common reference string (CRS)

[Ben-Or 90] Everything Provable is Provable in Zero-Knowledge

[DMP 90] 

[GMW 91] Proofs that yield nothing but their validity or all languages in NP have zero-knowledge proof systems
- The first zero-knowledge argument for an NP-complete problem (Graph 3-Coloring).

[LFKN 92] Algebraic methods for interactive proof systems (Sum-Check protocol)

[Babai 91] Checking computations in polylogarithmic time

[AS'92] Probabilistic checking of proofs: a new characterization of NP 

[ALM+'92] Proof verification and the hardness of approximation problems

[Kilian '92] A note on efficient zero-knowledge proofs and arguments

[Micali '94] CS proofs, maybe cite the 2000 version

[Kilian 95] Improved efficient arguments: Preliminary version

[BG 02] Universal arguments and their applications

[IKO'07] Efficient Arguments without Short PCPs, Ishai, Kushilevitz, and Ostrovsky

[GKR 08] 

[Groth '10] Short pairing-based non-interactive zero-knowledge arguments

[BCCT 12] From extractable collision resistance to succinct non-interactive arguments of knowledge, and back again
- Maybe introduces the term SNARK

[Lipmaa '12] Progression-free sets and sublinear pairing-based non-interactive zeroknowledge arguments.

[GGPR 13] 

[PHGR13] Pinocchio: Nearly practical verifiable computation

[Groth16] On the Size of Pairing-based Non-interactive Arguments

# Further Reading
zk-SNARKs: A Gentle Introduction, Anca Nitulescu
- This is an excellent read. Section 2 describes the historical evolution of proofs and arguments, and parallels much of this blog post — I wish I had read it sooner!

Proofs, arguments, and zero-knowledge, Justin Thaler, 2023. 
- Another excellent read, and source for much of this post. Also check out his course on Probabilistic Proof Systems.

https://a16zcrypto.com/posts/article/zero-knowledge-canon/, Especially Section 6