#Package imports

using Plots
using CSV, DataFrames
using LinearAlgebra
using SparseArrays
using Arpack

#Variables
x0 = -50
dx = 0.05
g = 1.5
κ = 0.2

#Loading \phi_ss

df = CSV.read("/Users/zoe/Documents/AnalogSimProject/Atom Steady States/SS_L200_dx0p1_g1p5_k0p2.csv", DataFrame, header=false)
phi_ss_original = df[:, 1]
phi_ss = phi_ss_original[5001:5:15001] #making L and dx smaller than imported
N = length(phi_ss)
x = range(x0, step=dx, length=N) 
#plot(x, phi_ss, xlabel = "x", ylabel = "ϕ_ss", label  = "ϕ_ss(x)", title  = "Imported ϕ_ss")

#Building Hamiltonian

maindiagonal = fill(-2.0, N)
offdiagonal = fill(1.0, N-1)
laplacianMatrix = spdiagm(0 => maindiagonal, -1 => offdiagonal, 1 => offdiagonal)/dx^2

massTermMatrix = g^2*I
cosTermMatrix = 2*π*κ*Diagonal(cos.(2*sqrt(π)*phi_ss))

hamiltonian = -laplacianMatrix + massTermMatrix + cosTermMatrix

#Solve eigensystem

vals, vecs = eigs(hamiltonian, nev=50, which=:LM, sigma=2.6)

display(plot(x, vecs[:,1], xlabel = "x", ylabel = "ϕ_b(x)"))

display(plot(vals, title = "Generic Schwinger atom eigenvalues", ylabel = "ω^2", marker=:circle))



#Build Capacitance matrix 

#Variables
e = 1.0;
c_t = 1.0;
c_l = 100.0;
E_t = κ*dx/2;
E_l = (e^2/(4*π^2))*(1/c_t);
L_t = (2*e/(g^2*dx))*sqrt(2/E_l)*sqrt(c_l/(c_t^2*(4*c_l + c_t)));
K = 4

phi_ss_disc = sqrt(π*K)*phi_ss

#e=1.0
#E_l=1.0
#E_t=2*π*κ
#L_t=1/(g^2)
#c_t = 1.0
#c_l = 100.0

capMatrix = spdiagm(0 => fill((c_t + 2*c_l), N), -1 => fill(-c_l, N-1), 1 => fill(-c_l, N-1))

display(heatmap(capMatrix, yflip = true, title = "Capacitance matrix"))

display(heatmap(inv(Matrix(capMatrix)), yflip = true, title = "Inverse capacitance matrix"))

#Hamiltonian with off-diag elements from capacitance included

#TODO: Recalculate \phi_ss here for depending on E_l, E_t, L_t
#TODO: check that when c_l = 0, and for appropriate choices of the other variables, this gives the same results as the continuum case w/o cap matrix

laplacianMatrixDisc = spdiagm(0 => fill(-2.0, N), -1 => fill(1.0, N-1), 1 => fill(1.0, N-1))

hamiltonianCap = -4*e^2*inv(Matrix(capMatrix))*(E_l*Matrix(laplacianMatrixDisc) - E_t*Diagonal(cos.(phi_ss_disc)) - (1/L_t)*I)

valsCap, vecsCap = eigen(hamiltonianCap)

display(plot(-valsCap[1950:end], title = "Eigenvalues in discrete JJA system", ylabel = "ω^2", marker=:circle))