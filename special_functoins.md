# Special Functions

## Gamma Related Functions

### Gamma Function (`sp.gamma`)

The gamma function is defined as follows for $\mathfrak{R}(z)>0$, and is defined by analytic continuation elsewhere.

$$
\Gamma(z)=\int_0^\infty t^{z-1}e^{-t}dt
$$

The gamma function has poles at non-negative integers. In Python, it is
calculated using `sp.gamma` and supports calculations for $z\in\mathbb{C}$ (returns NaN for non-negative integers)

### Beta Function (`sp.beta`)

The beta function is defined as follows for $x,\,y\in\mathbb{R}^+$, and is defined by analytic continuation elsewhere.

$$
\textrm{Beta}(x,y)=\int_0^1t^{x-1}(1-t)^{y-1}dt=\frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)}
$$

The beta function has poles when either $x$ or $y$ is a non-negative integer. In Python, it is calculated using `sp.beta` and supports calculations for $x,\,y\in\mathbb{R}$ (does not support complex $x,\,y$, and returns $\pm\infty$ when either $x$ or $y$ is a non-negative integer)

### Polygamma Function (`sp.polygamma`, `sp.psi`, `sp.digamma`)

The polygamma function is defined as the logarithmic derivative of the gamma function.

$$
\psi^{(n)}(z)=\frac{d^{n+1}}{dz^{n+1}}\log\Gamma(z)=\frac{d^n}{dz^n}\frac{\Gamma'(z)}{\Gamma(z)}
$$

When $n=0$, the polygamma function is also called the psi function or digamma function. The polygamma function has poles at non-negative integers regardless of $n$. In Python, it is calculated using `sp.polygamma` and supports calculations for $x\in\mathbb{R}$ (returns NaN for non-negative integers). However, `sp.psi` and `sp.digamma` support calculations for $z\in\mathbb{C}$ (both are exactly the same object and return NaN for non-negative integers).

## Error Functions

### Error Function (`sp.erf`)

The error function is defined as follows for all $z\in\mathbb{C}$.

$$
\textrm{erf}(z)=\frac{2}{\sqrt\pi}\int_0^z e^{-t^2}dt
$$

For $x\in\mathbb{R}$, the error function has asymptotes at $\pm1$ and is
analytic for all $z\in\mathbb{C}$. In Python, it is calculated using `sp.erf` and supports calculations for $z\in\mathbb{C}$.

### Fresnel Integral (`sp.fresnel`)

The Fresnel integral functions are defined as follows for all $z\in\mathbb{C}$.

$$
\begin{aligned}
S(z)&=\int_0^z\sin\bigg(\frac{\pi t^2}{2}\bigg)\,dt\\
C(z)&=\int_0^z\cos\bigg(\frac{\pi t^2}{2}\bigg)\,dt
\end{aligned}
$$

For $x\in\mathbb{R}$, the Fresnel integral functions have asymptotes at $\pm1/2$ and are analytic for all $z\in\mathbb{C}$. In Python, they are calculated using `sp.fresnel`, which returns a tuple containing both $S(z)$ and $C(z)$ values for a given $z\in\mathbb{C}$.

The parametrized curve using Fresnel integral functions as $x,\,y$ coordinates is known as an Euler spiral, Cornu spiral, or clothoid.

## Orthogonal polynomials

### Associated Laguerree Polynomials (`sp.assoc_laguerre`)

The associated (generalized) Laguerre polynomial은 $n\in\mathbb{N},\,\alpha,\,x\in\mathbb{R}$에 대해 다음 미분방정식의 다항식 해로 정의되며, $k=0$인 경우를 특별히 Laguerre polynomial이라 한다.

$$
xy''+(\alpha+1-x)y'+ny=0
$$

associated Laguerre polynomials은 다음과 같은 닫힌 형식으로 표현할 수 있다.

$$
L^{(\alpha)}_n(x)=\sum_{i=0}^n(-1)^i\binom{n+k}{n-i}\frac{x^i}{i!}
$$

