# Regularization Methods
The advanced inverse methods in ForceFinder generally use one of three regularization types:

1. [Truncated singular value decomposition (TSVD)](sec:tsvd)
2. [Tikhonov regularization](sec:tikhonov)
3. [Elastic net regularization](sec:elastic_net), which is a generalization of the least absolute shrinkage and selection operator (LASSO)

It is important to note that the focus of many of the advanced inverse methods in ForceFinder is selecting an "optimal" hyperparameter that is used to tune the influence of the regularization, rather than the regularization itself. For example, there are multiple advanced inverse methods (for each SPR type) that use Tikhonov regularization, but with a different method to select regularization parameter (e.g., cross validation, L-curve, etc.). This section focuses on describing the fundamental formulations for the different regularization methods and will not address the techniques for selecting a hyperparameter. See the [hyperparameter tuning methods](hyperparameter_tuning_methods) section for more details on how the hyperparameters are selected.

(sec:tsvd)=
## Truncated Singular Value Decomposition
The TSVD is likely the most straightforward of the regularization techniques, since it simply uses the SVD to compute the pseudo-inverse of the FRF matrix with a restricted number of singular values:

$$
\begin{bmatrix}H\end{bmatrix}^+=\begin{bmatrix}V\end{bmatrix}\frac{1}{diag\begin{pmatrix}\begin{Bmatrix}S(1:N)\end{Bmatrix}\end{pmatrix}}\begin{bmatrix}U\end{bmatrix}^*
$$

The practitioner simply selects the number of singular values to include in the inverse (by adjusting $N$). The general concept behind the TSVD is that the low amplitude singular values (commonly called insignificant singular values) are related to information that is insignificant and potentially counterproductive to the ISE problem. Consequentially, this insignificant information should rejected from the inverse so the estimated sources are not contaminated by noise or other errors. In general, the amplitudes of the estimated sources will shrink as $N$ is reduced, since small singular values become large in the inverse problem.

```{tip}
The TSVD is a straightforward regularization technique to use when the practitioner is manually tuning the hyperparameter (the number of singular values to include in the inverse), since it is relatively easy to determine which singular values are significant to the inverse. However, this method can behave erratically when there are not that many singular values in the FRF matrix, since removing one singular value may remove significant information.
```

(sec:tikhonov)=
## Tikhonov Regularization
Tikhonov regularization is an extremely common technique that has been widely studied. It can be written with three different mathematical frameworks: constrained optimization (primal form), penalized optimization (Lagrangian form), and a closed form solution. Each mathematical framework provides a different perspective on how Tikhonov regularization influences the inverse problem, so they are all shown here.

### Constrained Optimization 
The constrained optimization expression of Tikhonov regularization clearly shows that the method is attempting to estimate sources that minimize the sum-squared amplitude response error with an explicit constraint on the sum-squared amplitudes of the sources. Consequently, an smaller $\tau$ will result in in smaller source amplitudes. Further, an smaller $\tau$ tends to shrink the source amplitudes towards each other since the larger sources will be influenced by the constraint before the smaller sources.

$$
\begin{gather*}
\underset{f}{\min}\begin{pmatrix}\lVert \begin{Bmatrix}X\end{Bmatrix} - \begin{bmatrix}H\end{bmatrix}\begin{Bmatrix}F\end{Bmatrix} \rVert_2^2\end{pmatrix} \\
subject~to: \lVert \begin{Bmatrix}F\end{Bmatrix} \rVert_2^2 \le \tau
\end{gather*}
$$

### Penalized Optimization
The penalized optimization expression of Tikhonov regularization is the Lagrangian of the constrained optimization expression, as shown in the expression below, which minimizes the sum-squared amplitude response error with a penalty on the sum-squared source amplitudes. This expression makes it obvious that Tikhonov regularization balances the response prediction error and source amplitudes. The balance between the two parameters in the minimization problem is adjusted by tuning $\lambda$ (commonly called the regularization parameter). A large $\lambda$ value will shift the balance towards solutions with small source amplitudes, but potentially large errors in the reconstructed response. 

$$
\underset{f}{\min}\begin{pmatrix}\lVert \begin{Bmatrix}X\end{Bmatrix} - \begin{bmatrix}H\end{bmatrix}\begin{Bmatrix}F\end{Bmatrix} \rVert_2^2 + \frac{\lambda}{2} \lVert \begin{Bmatrix}F\end{Bmatrix} \rVert_2^2\end{pmatrix}
$$

### Closed Form Solution
A closed form solution for the Tikhonov regularized FRF matrix inverse can be developed from the penalized optimization expression, as shown below with the singular value components of the FRF matrix. This equation shows how $\lambda$ has a nonlinear effect on the condition number of the problem since it will have a more significant impact on the smaller singular values than the larger singular values. Consequently, an appropriately chosen $\lambda$ will mitigate the ill-effects of small singular values while leaving the large singular values unaffected. 

$$
\begin{bmatrix}H\end{bmatrix}^+ = \begin{bmatrix}V\end{bmatrix}\frac{\begin{bmatrix}S\end{bmatrix}}{\begin{bmatrix}S\end{bmatrix}^2 + \lambda}\begin{bmatrix}U\end{bmatrix}^*
$$

```{note}
Some expressions for Tikhonov regularization do not divide the $\lambda$ value by two in the penalized optimization expression. Excluding this division results in the $\lambda$ value being multiplied by two in the closed form solution. This division is inconsequential to the inverse method (since it has a simple scaler impact on the chosen regularization parameter), but can result in a tidier closed form solution. However, it is important to know which version of the solution is being used to obtain consistent solutions between different software packages.
```

```{tip}
It can be challenging to manually tune the regularization parameter in Tikhonov regularization, since the influence of the parameter depends on the amplitude of the singular values in the FRF matrix. As such, it can be important to use a frequency dependent regularization parameter to account for the frequency dependent changes in the amplitudes of the singular values.
```

(sec:elastic_net)=
## Elastic Net Regularization 
Elastic net regularization is a generalized regularization method that combines the LASSO and Tikhonov penalties. To understand the elastic net, it is useful to first review the LASSO, where the constrained optimization expression is:

$$
\begin{gather*}
\underset{f}{\min}\begin{pmatrix}\lVert \begin{Bmatrix}X\end{Bmatrix} - \begin{bmatrix}H\end{bmatrix}\begin{Bmatrix}F\end{Bmatrix} \rVert_2^2\end{pmatrix} \\
subject~to: \lVert \begin{Bmatrix}F\end{Bmatrix} \rVert_1 \le \tau
\end{gather*}
$$

The penalized optimization expression for the LASSO is the Lagrangian of the constrained optimization expression, which is:

$$
\underset{f}{\min}\begin{pmatrix}\lVert \begin{Bmatrix}X\end{Bmatrix} - \begin{bmatrix}H\end{bmatrix}\begin{Bmatrix}F\end{Bmatrix} \rVert_2^2 + \lambda \lVert \begin{Bmatrix}F\end{Bmatrix} \rVert_1\end{pmatrix}
$$

These two expressions make it clear that the LASSO is very similar to Tikhonov regularization except the constraint/penalty is based on the sum of the absolute values of the sources. This subtle change in the formulation has a significant impact on the outcome of the ISE problem, since the different constraint promotes a sparse solution where insignificant sources are set to zero. The sparsity effect can be particularly useful in cases where the practitioner does not know which source DOFs should be included in the ISE problem.

The penalized optimization expressions for Tikhonov regularization and the LASSO can be combined to result in the elastic net, where the so-called "naive" elastic net is:

$$
\underset{f}{\min}\begin{pmatrix}\lVert \begin{Bmatrix}X\end{Bmatrix} - \begin{bmatrix}H\end{bmatrix}\begin{Bmatrix}F\end{Bmatrix} \rVert_2^2 + \lambda_1 \lVert \begin{Bmatrix}F\end{Bmatrix} \rVert_1 + \lambda_2 \lVert \begin{Bmatrix}F\end{Bmatrix} \rVert_2^2\end{pmatrix}
$$

The naive elastic net requires the practitioner to tune two regularization parameters, one for the LASSO penalty and one for the Tikhonov penalty. This expression can be simplified to a single regularization parameter by introducing a mixing parameter, $\alpha$, which controls the balance between the LASSO and Tikhonov penalties:

$$
\underset{f}{\min}\begin{pmatrix}\lVert \begin{Bmatrix}X\end{Bmatrix} - \begin{bmatrix}H\end{bmatrix}\begin{Bmatrix}F\end{Bmatrix} \rVert_2^2 + \lambda \begin{pmatrix} \alpha  \lVert \begin{Bmatrix}F\end{Bmatrix} \rVert_1 + (1-\alpha) \lVert \begin{Bmatrix}F\end{Bmatrix} \rVert_2^2 \end{pmatrix} \end{pmatrix}
$$

An $\alpha$ of one in this expression results in a pure LASSO penalty and an alpha of zero results in a pure Tikhonov penalty. The penalized optimization form of the elastic net is the Lagrangian of the following constrained optimization problem:

$$
\begin{gather*}
\underset{f}{\min}\begin{pmatrix}\lVert \begin{Bmatrix}X\end{Bmatrix} - \begin{bmatrix}H\end{bmatrix}\begin{Bmatrix}F\end{Bmatrix} \rVert_2^2\end{pmatrix} \\
subject~to: \alpha  \lVert \begin{Bmatrix}F\end{Bmatrix} \rVert_1 + (1-\alpha) \lVert \begin{Bmatrix}F\end{Bmatrix} \rVert_2^2 \le \tau
\end{gather*}
$$

The different expressions for the elastic net regularization can be interpreted in the same manner as Tikhonov regularization, meaning that the regularization places a constraint on the amplitudes of the sources, where the regularization parameter determines the significance of the regularization in the ISE problem. Unfortunately, the use of the 1-norm in the penalty means that there isn't a closed form solution to further analyze the regularization method and develop an intuition for manually tuning the regularization parameter.

```{note}
The ISE problem with elastic net regularization is solved as a constrained optimization problem, which is much more computationally complex than the closed form solutions for Tikhonov regularization and the TSVD. This increased computational complexity means that the solution to the ISE problem can take much longer to compute compared to the other regularization methods.
``` 
```{note}
Elastic net regularization is implemented in ForceFinder using the penalized optimization expression where the practitioner tunes the $\lambda$ and $\alpha$ parameters.
```