using AdaptiveIS, Base.Test

f(x)=mean(x)>=0.85 ? 1. : 0.

srand(5)
sim=ais(f,3)
@test_approx_eq_eps sim.μ[end] 0.0151874 1e-3

srand(5)
sim=ais(f,3,t0=[1;0;1])
@test_approx_eq_eps sim.μ[end] 0.0151874 1e-3

srand(5)
sim=ais(f,3,t0=[1;1;0],accel="saa")
@test_approx_eq_eps sim.μ[end] 0.0151874 1e-3

srand(5)
sim=ais(f,3,dimreduc=true)
@test_approx_eq_eps sim.μ[end] 0.0151874 1e-3

srand(5)
sim=ais(f,3,t0=1,dimreduc=true)
@test_approx_eq_eps sim.μ[end] 0.0151874 1e-3

srand(5)
sim=ais(f,3,t0=1,accel="saa",dimreduc=true)
@test_approx_eq_eps sim.μ[end] 0.0151874 1e-3
