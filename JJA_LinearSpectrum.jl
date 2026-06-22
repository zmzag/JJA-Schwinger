using LinearAlgebra
using Plots
using SparseArrays
using Arpack

#Variables
#Grid variables
L = 200
dx = 0.1
N = Int(L/dx + 1)

#Physical Variables
g = 1.5 #the value of g in the continuum model that you want to model in this lattice system
κ = 0.2 #the value of κ in the continuum model that you want to model in this lattice system
K = 4.0 #Luttinger parameter - CANT CHANGE FROM 4.0 IF YOU WISH TO SIMULATE SCHWINGER MODEL

e = 1.0; #physical lattice model electron charge
c_t = 0.001; #capacitance of parallel JJs
c_l = 0.0001; #capacitance of series JJs

#Assuming c_l >> c_t, the lattice system paramters must be determined as follows to accurately depict the continuum Schwinger model
E_t = κ*dx/2;
L_t = π*K/(dx*g^2);
ω = 1 + c_t/(2*c_l) - (c_t/(2*c_l))*sqrt(1 + 4*c_l/c_t);
E_l = 8*e^2/(K^2*π^2*abs(log(ω))*sqrt(c_t*(4c_l + c_t)));

println(E_l)

#Finds the steady state soliton
function solve_soliton(E_t::Real, E_l::Real, L_t::Real, L::Real, N::Integer; Θ::Function = x -> (x ≥ 0 ? 1.0 : 0.0), tol::Real = 1e-9, maxiter::Integer = 30, verbose::Bool = true)
    
    xrange = range(-L/2, L/2; length=N)
    dx = step(xrange)
    k = E_t/E_l    # coeff of sine term
    g0 = 1.0/(E_l*L_t)   # coeff of linear term
    φ_left, φ_right = 0.0, -2*π     #DBC
    
    # initial guess is the SG soliton
    φ = @. -4*atan(exp(sqrt(k)*xrange))   #applies initial guess to definied grid
    src = Θ.(xrange)   #applies Heaviside function to defined grid
    F  = zeros(N)   #function we want equal to zero (difference between guess and solution)
    
    #Calculates how much the guess is different from the real solution
    function residual!(F, φ)
        F[1] = φ[1] - φ_left
        F[N] = φ[N] - φ_right
        @inbounds for i in 2:N-1 #skips check that array element is a valid one to speed up loop
            laplacian  = (φ[i+1] - 2φ[i] + φ[i-1])/dx^2
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
            d[i] = -2.0 / dx^2 - k * cos(φ[i]) - g0
            dl[i-1] = 1.0 / dx^2
            du[i] = 1.0 / dx^2
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
        Δφ = J \ (-F) #left division, computes -F = Δϕ J
        φ .+= Δφ
    end
    
    @warn "Newton iteration did not reach tol = $tol within $maxiter steps " *
          "(‖F‖ = $(norm(residual!(F, φ))))."
    return xrange, φ
end

#Generating steady state soliton based on variable values declared above
x, ϕ_ss = solve_soliton(E_t, E_l, L_t, L, N)
ϕ_free = @. -4 * atan(exp(sqrt(E_t / E_l) * x))
#display(plot(x, [ϕ, ϕ_free], label=["ϕ_ss" "SG soliton"]))

#Generating hamiltonian
capMatrix = spdiagm(0 => fill((c_t + 2*c_l), N), -1 => fill(-c_l, N-1), 1 => fill(-c_l, N-1))
#display(heatmap(capMatrix, yflip = true, title = "Capacitance matrix"))
#display(heatmap(inv(Matrix(capMatrix)), yflip = true, title = "Inverse capacitance matrix"))
laplacian = spdiagm(0 => fill(-2.0, N), -1 => fill(1.0, N-1), 1 => fill(1.0, N-1))
cosTerm = Diagonal(cos.(ϕ_ss))
massTerm = I

mat1 = Matrix(-E_l*laplacian + E_t*cosTerm + (1/L_t)*massTerm)
mat2 = (1/(4*e^2))*Matrix(capMatrix)

vals, vecs = eigen(mat1, mat2)
display(plot(sqrt.(vals[1:30]), marker=:circle, xlabel = "eigenvalue index", ylabel = "ω", title = "Low energy spectrum of JJA"))
display(plot(x, vecs[:,8], xlims = (-100,100), label = ["n=1" "n=2" "n=3" "n=4" "n=5"], xlabel = "x", ylabel = "ϕ(x)_n", title = "JJA eigenfunctions"))


#Allowing for disorder in values of junction energies
#=
E_l_sd = 0.02*E_l
E_l_disordered = Diagonal(fill(E_l, N) .+ E_l_sd*randn(N))

E_t_sd = 0.02*E_t
E_t_disordered = Diagonal(fill(E_t, N) .+ E_t_sd*randn(N))

cosTermDisorder = Diagonal(E_t_disordered).*cosTerm
laplacianDisorder = Diagonal(E_l_disordered)*laplacian

hamiltonianDisorder = -4.0*e^2*inv(Matrix(capMatrix))*(laplacianDisorder - cosTermDisorder - (1/L_t)*massTerm)

valsDisorder, vecsDisorder = eigen(hamiltonianDisorder)
display(plot(sqrt.(valsDisorder[1:300]), marker=:circle, xlabel = "eigenvalue index", ylabel = "ω", title = "Low energy spectrum for disordered JJA"))
display(plot(x, vecsDisorder[:,13], xlims = (-50,50), ylims = (-.0001,.0001), label = ["n=1" "n=2" "n=3" "n=4" "n=5"], xlabel = "x", ylabel = "ϕ(x)_n", title = "Disordered JJA eigenfunctions"))

=#