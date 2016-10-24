#AdaptiveIS.jl
Approximate the expected value of a function whose domain is the unit hypercube via adaptive importance sampling, where the importance sampling parameter is updated using robust stochastic approximation. Features include:
* Plotting of the sample paths of the empirical mean and importance sampling parameter
* Acceleration methods if the function is zero-valued with a high probabiilty
* Dimension reduction if the domain of the function is high dimensional and the function is symmetric with respect to its inputs
