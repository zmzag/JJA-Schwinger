using LinearAlgebra
using Plots
using CSV, DataFrames

#Variables
L = 100
dx = 0.05
num = 2001

g = 1.5
κ = 0.2

function solve_soliton(L::Real, N::Integer; Θ::Function = x -> (x ≥ 0 ? 1.0 : 0.0), tol::Real = 1e-9, maxiter::Integer = 30, verbose::Bool = true)
    
    xrange = range(-L/2, L/2; length=N)
    dx = step(xrange)
    k = κ*sqrt(π) 
    g0 = g^2 
    φ_left, φ_right = 0.0, -sqrt(π) 

    # initial guess is the SG soliton
    φ = @. -(2/sqrt(π))*atan(exp(sqrt(2*π*κ)*xrange)) 
    src = Θ.(xrange)   #applies Heaviside function to defined grid
    F  = zeros(N)   #function we want equal to zero (difference between guess and solution)
    
    function residual!(F, φ)
        F[1] = φ[1] - φ_left
        F[N] = φ[N] - φ_right
        @inbounds for i in 2:N-1 #skips check that array element is a valid one to speed up loop
            laplacian = (φ[i+1] - 2φ[i] + φ[i-1])/dx^2
            F[i] = laplacian - k*sin(2*sqrt(π)*φ[i]) - g0*φ[i] - sqrt(π)*g0*src[i] 
        end
        return F
    end

    dl = zeros(N - 1)   # sub diagonal of Jacobian
    d  = zeros(N)   # diagonal
    du = zeros(N - 1)   # super diagonal
    
    function jacobian!(dl, d, du, φ)
        d[1] = 1.0
        d[N] = 1.0
        @inbounds for i in 2:N-1
            #d[i] = -2.0/dx^2 - k * cos(φ[i]) - g0
            d[i] = -2.0/dx^2 - 2*sqrt(π)*k*cos(2*sqrt(π)*φ[i]) - g0 #TESTING AGAINST WENTAO'S CODE - DELETE LATER
            dl[i-1] = 1.0/dx^2
            du[i-1] = 1.0/dx^2
        end
        return Tridiagonal(dl, d, du)
    end
    
    for iter in 1:maxiter
        residual!(F, φ)
        resnorm = norm(F)
        verbose && println("iter $iter:  ‖F‖ = $resnorm") #&& can be used for short circuit evaluation - in a && b, the expression b is only evaluated if a evaluates to true
        if resnorm < tol
            verbose && println("Converged in $iter iterations.")
            return xrange, φ
        end
        J  = jacobian!(dl, d, du, φ)
        Δφ = J \ (-F) #left division, computes -F = Δϕ J
        φ .+= Δφ
    end
    
    @warn "Newton iteration did not reach tol = $tol within $maxiter steps " *
          "(‖F‖ = $(norm(residual!(F, φ))))."
    return xrange, φ
end
 
x, ϕ = solve_soliton(L, num)
plt = plot(x,ϕ)

#=
function deriv(x, f)
    h = x[2] - x[1]
    df = similar(f)
    df[1] = (-3f[1] + 4f[2] - f[3])/(2*h)
    df[end] = (3f[end] - 4f[end - 1] + f[end - 2])/(2*h)
    df[2:end-1] = (f[3:end] .- f[1:end - 2])./(2*h)
    return df
end
=#

#=
function fwhm(x, f)
    abs_f = abs.(f)
    half_max = maximum(abs_f)/2
    n = length(f)

    left_x = 0
    for i in 1:n-1
        if abs_f[i] <= half_max && abs_f[i + 1] >= half_max
            t = (half_max - abs_f[i])/(abs_f[i + 1] - abs_f[i])
            left_x = x[i] + t * (x[i + 1] - x[i])
            break
        end
    end

    right_x = 0
    for i in n-1:-1:1
        if abs_f[i] >= half_max && abs_f[i + 1] <= half_max
            t = (half_max - abs_f[i])/(abs_f[i + 1] - abs_f[i])
            right_x = x[i] + t * (x[i + 1] - x[i])
            break
        end
    end
    return right_x - left_x
end

halfmax = maximum(abs.(deriv(x,ϕ))/2)
fullwidth = fwhm(x,deriv(x,ϕ))

display(plot!(plt, x, deriv(x,ϕ), xlims = (-5,5)))
hline!([-halfmax])
vline!([-fullwidth/2])
vline!([fullwidth/2])

println(fullwidth)
=#

df = CSV.read("/Users/zoe/Documents/AnalogSimProject/Atom Steady States/SS_L200_dx0p1_g1p5_k0p2.csv", DataFrame, header=false)
phi_ss_original = df[:, 1]
#phi_ss = phi_ss_original[5001:5:15001] #making L and dx smaller than imported
N = length(phi_ss_original)
xnew = range(-100, step=0.01, length=N) 
display(plot!(plt, xnew,phi_ss_original, ls=:dash))