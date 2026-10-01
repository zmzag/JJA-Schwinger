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
K = 4.0
g = 1.5
κ = 0.2

#Physical system variables - import these from JJA_ParametersTesting.jl
c_p = 1.0; #capacitance of parallel JJs in units of fF
c_s = 1.0; #capacitance of series JJs in units of fF

E_s = 4*e^2/(π^2*K^2*c_p) #energy of series JJs in units of zJ
E_p = hbar^2*κ/(2*π*E_s*K) #energy of parallel JJs in units of zJ
L_h = π^2*K^2*E_s/(4*e^2*g^2) #inductance of parallel inductor in units of nH


ω = 1 + c_p/(2*c_s) - sqrt(c_p/c_s + c_p^2/(4*c_s^2)); #intermediate variabel, unitless

#Finds the steady state soliton
function solve_soliton(E_p::Real, E_s::Real, L_h::Real, L::Real, N::Integer; Θ::Function = x -> (x ≥ 0 ? 1.0 : 0.0), tol::Real = 1e-9, maxiter::Integer = 50, verbose::Bool = true)
    
    xrange = range(-L/2, L/2; length=N)
    dx = step(xrange)
    k = E_p/E_s    # coeff of sine term
    g0 = (hbar^2/(4*e^2))*(1.0/(E_s*L_h))   # coeff of linear term
    φ_left, φ_right = 0.0, -2*π     #DBC
    
    # initial guess is the SG soliton
    φ = @. -4*atan(exp(sqrt(k)*xrange/dx))   #applies initial guess to definied grid
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
        Δφ = J\(-F) #left division, computes -F = J Δϕ
        φ .+= Δφ
    end
    
    @warn "Newton iteration did not reach tol = $tol within $maxiter steps " *
          "(‖F‖ = $(norm(residual!(F, φ))))."
    return xrange, φ
end

#Generating steady state soliton based on variable values declared above
x, ϕ_ss = solve_soliton(E_p, E_s, L_h, L, N)

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




#Allowing for disorder in values of junction energies

#Generates steady state with disorder
function solve_soliton_disorder(E_p_disordered, E_s_disordered, L_h::Real, L::Real, N::Integer; Θ::Function = x -> (x ≥ 0 ? 1.0 : 0.0), tol::Real = 1e-9, maxiter::Integer = 50, verbose::Bool = true)
    
    xrange = range(-L/2, L/2; length=N)
    dx = step(xrange)
    φ_left, φ_right = 0.0, -2*π     #DBC
    
    # initial guess is the SG soliton
    φ = @. -4*atan(exp(sqrt(E_p_disordered[1]/E_s_disordered[1])*xrange/dx))   #applies initial guess to definied grid
    src = Θ.(xrange)   #applies Heaviside function to defined grid
    F  = zeros(N)   #function we want equal to zero (difference between guess and solution)
    
    #Calculates how much the guess is different from the real solution
    function residual!(F, φ)
        F[1] = φ[1] - φ_left
        F[N] = φ[N] - φ_right
        @inbounds for i in 2:(N - 1) #skips check that array element is a valid one to speed up loop
            laplacianDisorder = E_s_disordered[i]*φ[i+1] - (E_s_disordered[i-1] + E_s_disordered[i])*φ[i] + E_s_disordered[i-1]*φ[i-1]
            F[i] = laplacianDisorder - E_p_disordered[i]*sin.(φ[i]) - (hbar^2/(4*e^2*L_h))*φ[i] - 2*π*(hbar^2/(4*e^2*L_h))*src[i]
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
            d[i] = -E_s_disordered[i-1] - E_s_disordered[i] - E_p_disordered[i]*cos(φ[i]) - (hbar^2/(4*e^2*L_h))
            dl[i-1] = E_s_disordered[i-1]
            du[i] = E_s_disordered[i]
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
        Δφ = J\(-F) #left division, computes -F = J Δϕ
        φ .+= Δφ
    end
    
    @warn "Newton iteration did not reach tol = $tol within $maxiter steps " *
          "(‖F‖ = $(norm(residual!(F, φ))))."
    return xrange, φ
end

E_s_sd = 0.02*E_s
E_s_disordered = fill(E_s, N-1) .+ E_s_sd*randn(N-1)
E_p_sd = 0.02*E_p
E_p_disordered = fill(E_p, N) .+ E_p_sd*randn(N)

x, ϕ_ss_disorder = solve_soliton_disorder(E_p_disordered, E_s_disordered, L_h, L, N)
cosTermDisorder = Diagonal(E_p_disordered).*Diagonal(cos.(ϕ_ss_disorder))

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