using AdaptiveIS

f(x)=mean(x)>=0.85 ? 1. : 0.
srand(5)
sim=ais(f,3)

@test_approx_eq sim.�� 0.015014809526721174

@test_approx_eq sim.�� [0.226725, 0.234437, 0.230236]