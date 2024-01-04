## Benchmarking Grobner Basis Algorithms

In order to better understand the performance of different algorithms for computing Gröbner bases, I'm working through some of the benchmark problems.

# Cyclic n-roots
The cyclic n-roots problem is frequently used to benchmark the performance of algorithms for solving systems of polynomial equations, but I found it surprisingly hard to find what this problem is and where it comes from. In short, it defines a system of $n+1$ polynomials in $n$ variables:

$$\begin{eqnarray}
x_0 + ... + x_{n-1} = 0 \\
x_0 x_1 + x_1 x_2 + ... + x_{n-2} x_{n-1} + x_{n-1} x_0 \\
i = 3,4,..., n-1 : \sum_{j=0}^{n-1} \prod_{k=j}^{j+i-1} x_{k \mod n} = 0 \\
x_0  x_1  ...  x_{n-1}  1 = 0
\end{eqnarray}$$

Each polynomial in the system is cyclic, i.e., they are invariant under cyclic permutation of the arguments. This problem appears to have been introduced by Göran Björck. If you're curious, [Cyclic p-roots of prime length p and related
complex Hadamard matrices](https://web.math.ku.dk/~haagerup/publications/CyclicRootsNov07.pdf) describes the origin of the problem. The system has nice symmetry, but it seems to be used as a benchmark primarily beause it is hard to solve, even for small values of $n$.

Helpfully, [Sage](https://doc.sagemath.org/html/en/reference/rings/sage/rings/ideal.html#sage.rings.ideal.Cyclic) lets us work with the ideals for these systems:

```sage
P.<x,y,z> = PolynomialRing(QQ, 3, order='lex')
I = sage.rings.ideal.Cyclic(P); I 
I.groebner_basis()
```

# Katsura
The Katsura ideals also pop up frequently as benchmarks in the literature. This is apparently a problem in physics, first introduced by Shigetoshi Katsura, and defined by a system of $n-1$ quadratics and one linear equation. Again, [Sage](https://doc.sagemath.org/html/en/reference/rings/sage/rings/ideal.html#sage.rings.ideal.Katsura) has tooling for these ideals:

```sage
P.<x,y,z> = PolynomialRing(QQ, 3)
I = sage.rings.ideal.Katsura(P, 3); I
I.groebner_basis()
```

Again, this seems to be used as a benchmark because it is an existing application of polynomial system solving, and not because the problem is particularly significant to computational algebra. Katsura-3 is used as a "toy example" in the [FGB Tutorial](https://www-polsys.lip6.fr/~jcf/FGb/Maple/tutorial-fgb.pdf). So far, [this](https://homepages.math.uic.edu/~jan/Demo/katsura5.html) is the best reference I've found. 

