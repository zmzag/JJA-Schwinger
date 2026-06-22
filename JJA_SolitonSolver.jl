using LinearAlgebra
using Plots

#Variables
L = 100
dx = 0.05
num = 2001

g = 1.5
κ = 0.2
e = 1.0;
c_t = 1.0;
c_l = 100.0;

E_t = κ*dx/2;
E_l = (e^2/(4*π^2))*(1/c_t);
L_t = (2*e/(g^2*dx))*sqrt(2/E_l)*sqrt(c_l/(c_t^2*(4*c_l + c_t)));





function solve_soliton(E_t::Real, E_l::Real, L_t::Real, L::Real, N::Integer; Θ::Function = x -> (x ≥ 0 ? 1.0 : 0.0), tol::Real = 1e-9, maxiter::Integer = 30, verbose::Bool = true)
    
    xrange = range(-L, L; length=N)
    dx = step(xrange)
    #k = E_t/E_l    # coeff of sine term
    k = 0.2*sqrt(π) #TESTING AGAINST WENTAO'S CODE - DELETE LATER
    #g0 = 1.0/(E_l*L_t)   # coeff of linear term
    g0 = 1.5^2 #TESTING AGAINST WENTAO'S CODE - DELETE LATER
    #φ_left, φ_right = 0.0, -2*π     #DBC
    φ_left, φ_right = 0.0, -sqrt(π) #TESTING AGAINST WENTAO'S CODE - DELETE LATER
    
    # initial guess is the SG soliton
    #φ = @. -4*atan(exp(sqrt(k)*xrange))   #applies initial guess to definied grid
    φ = @. -(2/sqrt(π))*atan(exp(sqrt(2*π*0.2)*xrange)) #TESTING AGAINST WENTAO'S CODE - DELETE LATER

    src = Θ.(xrange)   #applies Heaviside function to defined grid
    
    F  = zeros(N)   #function we want equal to zero (difference between guess and solution)
    

    function residual!(F, φ)
        F[1] = φ[1] - φ_left
        F[N] = φ[N] - φ_right
        @inbounds for i in 2:N-1 #skips check that array element is a valid one to speed up loop
            laplacian  = (φ[i+1] - 2φ[i] + φ[i-1])/dx^2
            #F[i] = laplacian - k*sin(φ[i]) - g0*φ[i] - 2*π*g0*src[i]
            F[i] = laplacian - k*sin(2*sqrt(π)*φ[i]) - g0*φ[i] - sqrt(π)*g0*src[i] #TESTING AGAINST WENTAO'S CODE - DELETE LATER
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
            #d[i] = -2.0 / dx^2 - k * cos(φ[i]) - g0
            d[i] = -2.0 / dx^2 - 2*sqrt(π)*cos(2*sqrt(π)*φ[i]) - g0 #TESTING AGAINST WENTAO'S CODE - DELETE LATER
            dl[i-1] = 1.0 / dx^2
            du[i] = 1.0 / dx^2
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
 
x, ϕ = solve_soliton(E_t, E_l, L_t, L, num)
ϕ_free = @. -4 * atan(exp(sqrt(E_t / E_l) * x))
#display(plot(x, [ϕ, ϕ_free], label=["ϕ_ss" "SG soliton"]))
plt = plot(x,ϕ)

df = CSV.read("/Users/zoe/Documents/AnalogSimProject/Atom Steady States/SS_L200_dx0p1_g1p5_k0p2.csv", DataFrame, header=false)
phi_ss_original = df[:, 1]
phi_ss = phi_ss_original[5001:5:15001] #making L and dx smaller than imported
x0 = -50
dx = 0.05
N = length(phi_ss)
xnew = range(x0, step=dx, length=N) 

display(plot!(plt, xnew, phi_ss, ls=:dash))