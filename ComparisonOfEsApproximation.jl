using LinearAlgebra
using Plots
using SparseArrays
using Arpack


L = 50 #units of micron
dx = 0.1 #units of micron
N = Int(L/dx + 1)
#Checking validity of continuum approximation
c_p = 1.0
c_s = 1.0
ω = 1 + c_p/(2*c_s) - sqrt(c_p/c_s + c_p^2/(4*c_s^2))
A_lattice = 1/c_p
A_limit = 2/(abs(log(ω))*sqrt(c_p*(4*c_s + c_p)))
capMatrix = spdiagm(0 => fill((c_p + 2*c_s), N), -1 => fill(-c_s, N-1), 1 => fill(-c_s, N-1))
Cinv = inv(Matrix(capMatrix))
rowSums = vec(sum(Cinv, dims=2))          

plt = scatter(1:N, rowSums;
              label  = "exact C^-1",
              xlabel = "row index",
              ylabel = "C^(-1)",
              title  = "C_p = $(c_p),  C_s = $(c_s)",
              markersize = 2, markerstrokewidth = 0)
hline!(plt, [A_lattice]; label = "1/c_p", ls = :dash, lw = 2)
hline!(plt, [A_limit];   label = "2/(|ln ω| sqrt(c_p(4c_s+c_p)))", ls = :dot, lw = 2)
display(plt)

# zoom on the first rows, where the end of the array shows up
nEnd = min(N, 40)
pltEnd = scatter(1:nEnd, rowSums[1:nEnd];
                 label  = "exact C^-1",
                 xlabel = "row index",
                 ylabel = "C^(-1)",
                 title  = "C_p = $(c_p),  C_s = $(c_s) (boundary rows)",
                 markersize = 2, markerstrokewidth = 0)
hline!(pltEnd, [A_lattice]; label = "1/c_p", ls = :dash, lw = 2)
hline!(pltEnd, [A_limit];   label = "2/(|ln ω| sqrt(c_p(4c_s+c_p)))", ls = :dot, lw = 2)
display(pltEnd)