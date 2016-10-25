# AdaptiveIS.jl

Approximate the expected value of a function of iid standard uniform random variables via adaptive importance sampling, where the importance sampling parameter is updated using robust stochastic approximation. Features include:
* Plotting of the sample paths of the empirical mean and importance sampling parameter
* Acceleration methods if the function is zero-valued with a high probabiilty
* Dimension reduction if the dimension of the domain of the function is large and the function is symmetric with respect to its inputs

Adaptive importance sampling is a variance reduction technique in Monte Carlo simulations. Although it is more computationally intensive than a crude Monte Carlo simulation, the convergence to the expected value is faster over the same number of iterations.

As an example, consider estimating the probability that the mean of three iid uniform random variables is at least 0.85:

```julia
using AdaptiveIS
srand(5)
f(x) = mean(x)>=0.85 ? 1. : 0.
sim1 = ais(f,3,n=10^4)
```

Valid acceleration methods are "directsub", "sa", and "saa" (recommended):

```julia
sim2 = ais(f,3,n=10^4,accel="saa")
```

Dimension reduction is as easy as:

```julia
sim3 = ais(f,3,n=10^4,dimreduc=true)
```

Note that the function is symmetric with respect to its inputs, so this is valid. Compare the above to a crude Monte Carlo simulation:

```julia
sim4 = zeros(10^4)
[sim4[i] = f(rand(3)) for i=1:10^4]
```

A plot of the sample paths of the empirical means and the true mean:

```julia
plot(sim1)
plot!(sim2)
plot!(sim3)
plot!(cumsum(sim4)./(1:10^4))
plot!(0.0151874*ones(10^4),ylims=c(0.01,0.02))
```

<img src=https://github.com/EdgyEggplant/AdaptiveIS.jl/raw/master/images/means.png width=600 height=400>
