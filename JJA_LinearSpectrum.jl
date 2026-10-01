using LinearAlgebra
using Plots
using SparseArrays
using Arpack

#Variables
#Grid variables
L = 50 #units of micron
dx = 0.1 #units of micron
N = Int(L/dx + 1)

#constants
hbar = 0.1054571817
e = 0.1602176634

#Physical system variables - import these from JJA_ParametersTesting.jl
c_p = 12; #capacitance of parallel JJs in units of fF
c_s = 0.03; #capacitance of series JJs in units of fF
E_s = 0.0001; #energy of series JJs in units of zJ
E_p = 0.04; #energy of parallel JJs in units of zJ
L_h = 1; #inductance of parallel inductor in units of nH


ω = 1 + c_p/(2*c_s) - sqrt(c_p/c_s + c_p^2/(4*c_s^2)); #intermediate variabel, unitless

#Finds the steady state soliton
function solve_soliton(E_p::Real, E_s::Real, L_h::Real, L::Real, N::Integer; Θ::Function = x -> (x ≥ 0 ? 1.0 : 0.0), tol::Real = 1e-9, maxiter::Integer = 50, verbose::Bool = true)
    
    xrange = range(-L/2, L/2; length=N)
    dx = step(xrange)
    k = E_p/E_s    # coeff of sine term
    g0 = (hbar^2/(4*e^2))*(1.0/(E_s*L_h))   # coeff of linear term
    φ_left, φ_right = 0.0, -2*π     #DBC
    
    # initial guess is the SG soliton
    φ = @. -4*atan(exp(sqrt(k)*xrange))   #applies initial guess to definied grid
    src = Θ.(xrange)   #applies Heaviside function to defined grid
    F  = zeros(N)   #function we want equal to zero (difference between guess and solution)
    
    #Calculates how much the guess is different from the real solution
    function residual!(F, φ)
        F[1] = φ[1] - φ_left
        F[N] = φ[N] - φ_right
        @inbounds for i in 2:(N - 1) #skips check that array element is a valid one to speed up loop
            laplacian  = (φ[i + 1] - 2*φ[i] + φ[i - 1])
            F[i] = laplacian - k*sin(φ[i]) - g0*φ[i] - 2*π*g0*src[i]
        end
        return F
    end

    dl = zeros(N - 1)   # sub diagonal of Jacobian
    d  = zeros(N)   # diagonal
    du = zeros(N - 1)   # super diagonal
    
    #Calculates slope of tangent line to guess
    function jacobian!(dl, d, du, φ)
        d[1] = 1.0
        d[N] = 1.0
        @inbounds for i in 2:N-1
            d[i] = -2.0 - k * cos(φ[i]) - g0
            dl[i-1] = 1.0 
            du[i] = 1.0 
        end
        return Tridiagonal(dl, d, du)
    end
    
    #Generates a new guess from x intercept of tangent line to original guess
    for iter in 1:maxiter
        residual!(F, φ)
        resnorm = norm(F)
        verbose && println("iter $iter:  ‖F‖ = $resnorm") #&& can be used for short circuit evaluation - in a && b, the expression b is only evaluated if a evaluates to true
        if resnorm < tol
            verbose && println("Converged in $iter iterations.")
            return xrange, φ
        end
        J = jacobian!(dl, d, du, φ)
        Δφ = J\(-F) #left division, computes -F = Δϕ J
        φ .+= Δφ
    end
    
    @warn "Newton iteration did not reach tol = $tol within $maxiter steps " *
          "(‖F‖ = $(norm(residual!(F, φ))))."
    return xrange, φ
end

#Generating steady state soliton based on variable values declared above
x, ϕ_ss = solve_soliton(E_p, E_s, L_h, L, N)
ϕ_free = @. -4 * atan(exp(sqrt(E_p / E_s) * x))
#display(plot(x, [ϕ, ϕ_free], label=["ϕ_ss" "SG soliton"]))

#Generating hamiltonian
capMatrix = spdiagm(0 => fill((c_p + 2*c_s), N), -1 => fill(-c_s, N-1), 1 => fill(-c_s, N-1))
#display(heatmap(capMatrix, yflip = true, title = "Capacitance matrix"))
display(heatmap(inv(Matrix(capMatrix)), yflip = true, title = "Inverse capacitance matrix"))
laplacian = spdiagm(0 => fill(-2.0, N), -1 => fill(1.0, N-1), 1 => fill(1.0, N-1))
cosTerm = Diagonal(cos.(ϕ_ss))
massTerm = I

mat1 = Matrix(-(E_s/hbar)*laplacian + (E_p/hbar)*cosTerm + (hbar/(4*e^2*L_h))*massTerm)
mat2 = (hbar/(4*e^2))*Matrix(capMatrix)

vals, vecs = eigen(mat1, mat2)
display(plot(sqrt.(vals[1:50]), marker=:circle, xlabel = "eigenvalue index", ylabel = "ω", title = "Low energy spectrum of JJA"))

ns = [1,2,3,4,5,6,7,8,9,10]
offset = 7.0
plt = plot(xlims = (-100,100), xlabel= "x", ylabel = "ϕ(x)", title = "JJA eigenfunctions", yticks=false)
for (i,n) in enumerate(ns)
    plot!(plt,x,vecs[:,n].+ (i-1)*offset,label="n=$n")
end
display(plt)

#Checking validity of continuum approximation
c_p = 1.0
c_s = 1.0
ω = 1 + c_p/(2*c_s) - sqrt(c_p/c_s + c_p^2/(4*c_s^2))
A_lattice = 1/c_p
A_limit = 2/(abs(log(ω))*sqrt(c_p*(4*c_s + c_p)))
capMatrix = spdiagm(0 => fill((c_p + 2*c_s), N), -1 => fill(-c_s, N-1), 1 => fill(-c_s, N-1))
Cinv = inv(Matrix(capMatrix))
rowSums = vec(sum(Cinv, dims=2))          # rowSums[i] = sum over j of Cinv[i, j]

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




#Allowing for disorder in values of junction energies

E_s_sd = 0.02*E_s
E_s_disordered = fill(E_s, N-1) .+ E_s_sd*randn(N-1)

E_p_sd = 0.02*E_p
E_p_disordered = Diagonal(fill(E_p, N) .+ E_p_sd*randn(N))

cosTermDisorder = Diagonal(E_p_disordered).*cosTerm

dl = copy(E_s_disordered)  # sub diagonal of Jacobian
d  = zeros(N)   # diagonal
du = copy(E_s_disordered)   # super diagonal
d[1] = -2*E_s_disordered[1]
d[N] = -2*E_s_disordered[N-1]

@inbounds for i in 2:N-1
    d[i] = -2.0*(E_s_disordered[i-1] + E_s_disordered[i])/2
end
laplacianDisorder = Tridiagonal(dl,d,du)


mat1 = Matrix(-(1/hbar)*laplacianDisorder + (1/hbar)*cosTermDisorder + (hbar/(4*e^2*L_h))*massTerm)
mat2 = (hbar/(4*e^2))*Matrix(capMatrix)

valsDisorder, vecsDisorder = eigen(mat1, mat2)

display(plot(sqrt.(valsDisorder[1:50]), marker=:circle, xlabel = "eigenvalue index", ylabel = "ω", title = "Low energy spectrum of JJA with Disorder"))

ns = [1,2,3,4,5,6,7,8,9,10]
offset = 7.0
plt = plot(xlims = (-100,100), xlabel= "x", ylabel = "ϕ(x)", title = "JJA eigenfunctions with Disorder", yticks=false)
for (i,n) in enumerate(ns)
    plot!(plt,x,vecsDisorder[:,n].+ (i-1)*offset,label="n=$n")
end
display(plt)