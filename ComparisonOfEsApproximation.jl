using Plots

e = 1
ω(c_p, c_s) = 1 + c_p/(2*c_s) - sqrt(c_p/c_s + c_p^2/(4*c_s^2))

f(c_p) = e^2/(4*π^2*c_p)   # define your function
g(c_p, c_s) = e^2/(2*π^2)*1/(abs(log(ω(c_p,c_s)))*sqrt(c_p*(4c_s + c_p)))

cp_range = range(1, 10, length=500)   # range of x values
approx = f.(cp_range)                         # broadcast f over x
exact_cs1 = g.(cp_range,0.1)
exact_cs2 = g.(cp_range,1)
exact_cs3 = g.(cp_range,10)

plot(cp_range, approx, label="approximation", xlabel="c_p", ylabel="E_s(c_p)")
plot!(cp_range, exact_cs1, label="exact for c_s=0.1")
plot!(cp_range, exact_cs2, label="exact for c_s=1")
plot!(cp_range, exact_cs3, label="exact for c_s=10")