## Solving Polynomial Systems with Gröbner Bases: Part 0

Today, I want to try out [Groebner.jl](https://github.com/sumiya11/Groebner.jl). It's one of the newer Julia 
packages for computing Gröbner bases, and I'd like to use it to solve some systems of polynomial equations over finite 
fields.  

*Why?* I'm currently working with "ZK-Friendly" hash functions like [Poseidon](https://www.poseidon-hash.info/) and 
[Rescue](https://www.esat.kuleuven.be/cosic/sites/rescue). Unlike most hash functions that are designed to be 
efficiently computable on modern CPUs, these hash functions are designed to have concise algebraic descriptions so 
that they can be efficiently computed in zero-knowledge proofs. As a result, the security of these 
hash functions hinges on the difficulty of solving particular systems of polynomial equations over large finite 
fields -- so-called "Gröbner basis attacks". The assumption is that this is hard, and in fact the worst-case 
complexity for computing Gröbner bases is doubly-exponential in the number of variables. Of course, worst-case 
analysis can be misleading -- solving a system of *linear* polynomials is easy -- and so I'm looking for some 
intuition about when a particular system of polynomial equations is solvable.
 
# Polynomial Systems
A polynomial system over a finite field is a collection of polynomial equations where the coefficients and variables
belong to a finite field. Given polynomials $f_1, f_2, ..., f_m$ over a finite field $F_p$, I want to find all 
common solutions to the system of equations:

$$\begin{eqnarray}
f_1(x_1, x_2, ..., x_n) = 0 \\
f_2(x_1, x_2, ..., x_n) = 0 \\
... \\
f_m(x_1, x_2, ..., x_n) = 0
\end{eqnarray}$$

# What is a Gröbner Basis?
If all our polynomials are linear, then we are in the friendly realm of linear algebra. In that case, we'd use 
Gaussian elimination to solve the system $Ax = 0$. Gaussian elimination would put the system into row-echelon form, 
and then it's easy to solve one variable at a time. This is exactly what we get when we compute a Gröbner basis for 
a linear system:

```julia
import Pkg; Pkg.add("Groebner")
using AbstractAlgebra
using Groebner

_R, (a,b,c) = PolynomialRing(GF(127), ["a", "b", "c"], ordering=:lex)

linearSystem = [
    26*b + 52*c + 62, 
    54*b + 119*c + 55,
    41*a + 91*c + 13,
]

groebner(linearSystem)
```

This computes the basis $[c + 75, b + 38, a + 29]$, i.e, the reduced row-echelon form of the system:

$$\begin{equation*}
\begin{bmatrix}
1 & 0 & 0 & 29 \\
0 & 1 & 0 & 38 \\
0 & 0 & 1 & 75
\end{bmatrix}
\end{equation*}$$

Gröbner bases are a generalization of Gaussian elimination to polynomial systems: they transform a system of 
polynomials into an equivalent system that is easier to solve. Unfortunately, the worst-case complexity of computing 
a Gröbner basis is **doubly-exponential** in the number of variables and computing Gröbner bases is 
EXPSPACE-complete in general. The complexity bounds are interesting, and I'll try to talk about those in a future 
post. But ignoring that scary complexity for a moment, let's compute a Gröbner basis over a small finite field:  

```julia
_R, (x, y, z) = PolynomialRing(GF(13), ["x", "y", "z"], ordering=:lex)

system = [
  x + y + z,
  x*y + x*z + y*z,
  x*y*z - 1
]

groebner(system)   
```

This gives us the Gröbner basis:

$$\begin{eqnarray}
z^3 -1 \\
y^2 + yz + z^2 \\
x + y + z
\end{eqnarray}$$

Notice that we can start at the top and solve each equation for a single variable. Working in $F_{13}$, the roots of 
the first equation are 9, 3, and 1. Substituting these into the second equation gives us:

- $z=9$: $y^2 + 9y + 3 = 0$, with roots 3 and 1,
- $z = 3$: $y^2 + 3y + 9 = 0$, with roots 9 and 1,
- $z = 1$: $y^2 + y + 1 = 0$,  with roots 9 and 3

Plugging these combinations of $z$ and $y$ into the third equation gives us the common zeros:
- $z = 9, y = 3$: $x = 1$
- $z = 9, y = 1$: $x = 3$
- $z = 3, y = 9$: $x = 1$
- $z = 3, y = 1$: $x = 9$
- $z = 1, y = 9$: $x = 3$
- $z = 1, y = 3$: $x = 9$

Cool. So what about systems with no common zeros? Let's try:

```julia
noSolutions = [
    x + 1,
    x + 2,
    x + 3
]

grobSolve(noSolutions)
```
The basis is simply $[1]$, which clearly has no zeros.

# Up next
In future posts, I'd like to talk about the definition of a Gröbner basis, algorithms for computing them, the 
complexity of those algorithms, and some applications to cryptography. Stay tuned!