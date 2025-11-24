module Flash

using ..EOS
using ..Solvers
using ForwardDiff
using NLsolve
using LinearAlgebra
using Statistics
# using LinearSolve
using FixedPointAcceleration
using ExportAll

function rho_α_Q(x; U_spec, V_spec, N_spec, model, numPhases = nothing, Scale = nothing, ScaleFunc = nothing)
    T = x[end]                     # Temperature
    α = clamp(x[model.Nc + 1], 1e-12, 1.0 - 1e-12)  # Vapor fraction (clamped for stability)
    𝐳 = N_spec ./ sum(N_spec)     # Overall composition
    V_mix = V_spec              # Total system volume

    # Compute mixture density (mass basis)
    ρ_mix = sum(N_spec .* model.Mw) / V_mix

    # Extract component-wise mass densities in vapor phase (normalized by ρ_mix)
    ρ_vap_mass = x[1:model.Nc] .* ρ_mix

    # Phase volumes
    V_vap = α * V_mix
    V_liq = V_mix - V_vap

    b_val = EOS.b(𝐳 ; model)
    B = b_val #* sum(N_spec)

    # V should be more than B and less than V_spec - B
    # V_vap = clamp(V_vap, B + 1e-5, V_spec - B - 1e-5)
    # V_liq = V_mix - V_vap

    # extract_value(qty) = ForwardDiff.value.(ForwardDiff.value.(qty))
    # Moles in each phase (from densities and phase volumes)
    N_vap = ρ_vap_mass .* V_vap ./ model.Mw
    N_liq = N_spec .- N_vap

    ρ_liq_mass = N_liq .* model.Mw ./ V_liq
    
   
    indx = 1

    # Compute Helmholtz energies for both phases
    A_vap = EOS.A_wrapper(T, ρ_vap_mass, ρ_liq_mass[indx]; 
                          indx, ρ_mix, V_mix, 𝐳, model)
    A_liq = EOS.A_wrapper(T, ρ_liq_mass, ρ_vap_mass[indx]; 
                          indx, ρ_mix, V_mix, 𝐳, model)

    # Total Helmholtz energy and Q value
    A_total = A_vap + A_liq
    Q = (U_spec - A_total) / T

    return Q
end


# check if ∑cᵢbᵢ < 1, all cᵢ > 0, V > 0
# increement is the perturbation to the current state

function func_wrapper(func, x_scaled, μ, σ)
    x_descaled = z_score_descale(x_scaled, μ, σ)
    return func(x_descaled)
end

function applyScaleDescale(optimizer; Scale, ScaleFunc)

    Scaler, DeScaler = ScaleFunc

    opt(f, x, cons; tol, maxiter, α, useNewtonJulia) = optimizer(f, Scaler(x), cons; tol, maxiter, α, useNewtonJulia)

    ScaledOptimiser(f, x, cons=nothing; tol, maxiter, α, useNewtonJulia) = DeScaler(opt(f, x, cons; tol, maxiter, α, useNewtonJulia))

    return ScaledOptimiser
end

function OptimizeHelmholtz(func, g, H, x, cons=nothing; tol=1e-6, maxiter=300, α=1.0, V_total=nothing, N_total=nothing,useNewtonJulia=true)
    # println("OptimizeHelmholtz, x: ", x)
    # T is stored in x[end] and λ = -1/R but we are maximising S/R, not S
    # lagrangian(x) = func(x)
    
    # # @show x func(x)
    # g(x) = ForwardDiff.gradient(lagrangian, x)
    # H(x) = ForwardDiff.hessian(lagrangian, x)
   
    # @show g(x)
    # sol = nlsolve(func, x, xtol=tol, ftol=tol, iterations=2000, method=:trust_region)
    # sol = nlsolve(g, H, x, xtol=tol, ftol=tol, iterations=4000, method=:newton, linesearch=LineSearches.BackTracking(order=3))
    # converged = sol.x_converged || sol.f_converged
    @show tol
    sol, converged, iterations = Solvers.newton_mixed(func, g, H, x; tol=tol, maxiter)
    # @show sol
    # norm_converged = norm(g(sol.zero))
    # @show sol.iterations sol.x_converged, sol.f_converged, norm_converged
    if !converged        
        @warn "Flash did not converge."
        false, nothing       
    end
    # @show g(sol.zero)
    # Hess = round.(H(sol.zero), digits=2)
    # @show Hess
    return true, sol, iterations
    # return true, sol.zero, sol.iterations
end

convergence(x, y; tol) = norm(y .- x) < tol

@exportAll()
end