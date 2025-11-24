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

# These equations are from Bi. Really bad formulations. It works only when we are super close to the solution.

function S_reduced(x; U_spec, V_spec, N_spec, model, numPhases, Scale, ScaleFunc, λ_penalty=1e1)
    _n = model.Nc
    (; R, doScale) = model
    Scaler, DeScaler = ScaleFunc
    if doScale
        x = DeScaler(x)
    end

    #TODO: Check if β is within bounds 
    β = VapourPhaseFraction(N_spec .- x[1:end-2], N_spec)
    # β_value = β.value

    if false
        # println("β: ", β_value)        
        K = EquilibriumRatio(x[1:_n], N_spec)
        if β_value > 1.0
            # Beta can not be that high, need to reduce x[1:_n]
            x[1:_n] .= N_spec .* (1 - 1e-7)
        elseif β_value <= 0.0
            # Beta can not be that low, need to increase x[1:_n]
            x[1:_n] .= N_spec .* 1e-7
        end

        # x_k_value = x_k.value
        z_frac = N_spec ./ sum(N_spec)
        max_index = argmax(K)
        min_index = argmin(K)
        β_max = (K[max_index] * z_frac[max_index] - 1) ./ (K[max_index] - 1)
        β_min = (z_frac[min_index] - 1) / (K[min_index] - 1)

        K = ForwardDiff.value.(K)
        β_max = ForwardDiff.value(β_max)
        β_min = ForwardDiff.value(β_min)
        # @show K, β_value, β_max, β_min

    end

    U_k_p = convert(eltype(x), U_spec)
    V_k_p = convert(eltype(x), V_spec)
    V_tot = convert(eltype(x), V_spec)
    mole_k_p = copy(N_spec) .* one(eltype(x))  # Ensure mole_k_p matches the type of `x`

    penalty_moles = 0.0
    penalty_vol = 0.0
    S = zero(eltype(x))  # Initialize S with the correct type
    inner_iters = 0
    for k in 0:numPhases-2

        U_k = x[_n+2+k*(_n+2)]
        V_k = x[_n+1+k*(_n+2)]
        mole_k = x[1+k*(_n+2):_n+k*(_n+2)]
        # if V_k / V_tot > 1e-5
        S_UVN, iters = EOS.Property_UVN(U_k, V_k, mole_k; model, ThermoFunc=EOS.S_EOS, verbose=true)
        S += S_UVN
        inner_iters += ForwardDiff.value(iters)
        # end
        U_k_p -= U_k
        V_k_p -= V_k
        mole_k_p = mole_k_p .- mole_k

        penalty_moles += abs(sum(ForwardDiff.value(mole_k[mole_k.<0]))) - sum(ForwardDiff.value(mole_k[mole_k.<0]))
        penalty_vol += abs(ForwardDiff.value(V_k)) - ForwardDiff.value(V_k)
    end
    # @show typeof(inner_iters)
    # if isa(inner_iters, Float64)
    # println(inner_iters)
    # end
    x_star = vcat(mole_k_p, V_k_p, U_k_p)
    S = EOS.TotalQty(x_star; model, ThermoFunc=EOS.S_EOS) + S
    # S_val = S.value
    # if isa(S_val, Float64)
    #     println(" S: ", S_val, " U: ", U_k_p.value, " V: ", V_k_p.value)
    #     # S_1 = Property_UVN(U_spec, V_spec, N_spec; model, ThermoFunc=S_EOS)
    #     # println(" S_1: ", S_1)
    # end
    penalty = λ_penalty * (penalty_moles + penalty_vol)
    p_val = ForwardDiff.value(penalty)
    if p_val > 0
        # println("Penalty: ", p_val)
    end
    S_val = ForwardDiff.value(S)
    # @show p_val S_val
    return -S / R # + penalty
end

function cons_U(x; U_spec, V_spec, N_spec, model, numPhases, Scale, ScaleFunc, doScale=true)
    Scaler, DeScaler = ScaleFunc
    if doScale
        x = DeScaler(x)
    end
    U_k_p = convert(eltype(x), U_spec)
    V_k_p = convert(eltype(x), V_spec)
    V_tot = convert(eltype(x), V_spec)
    mole_k_p = copy(N_spec) .* one(eltype(x))

    # the last value is the lagrange multiplier
    # T = x[end-1]
    T = x[end]
    # x = [N11, N21 ... Nn1, V1, N12, N22 ... Nn2, V2, N13, N23 ... Nn3, V3, T]
    _n = model.Nc
    for k in 0:numPhases-2
        V_k = x[_n+1+k*(_n+1)]
        mole_k = x[1+k*(_n+1):_n+k*(_n+1)]

        if V_k / V_tot > 1e-5
            U_k_p -= U_EOS(T, V_k, mole_k; model)
        end
        V_k_p -= V_k
        mole_k_p = mole_k_p .- mole_k
    end
    U_k_p -= U_EOS(T, V_k_p, mole_k_p; model)

    # Here we are returning the negative of the constraint so that it is confromant with the from of constraint function used in the paper.
    return -U_k_p
end

function S_reduced_constrained(x; U_spec, V_spec, N_spec, model, numPhases, Scale, ScaleFunc)
    _n = model.Nc
    (; R, doScale) = model
    Scaler, DeScaler = ScaleFunc
    if doScale
        x = DeScaler(x)
    end

    U_k_p = convert(eltype(x), U_spec)
    V_k_p = convert(eltype(x), V_spec)
    V_tot = convert(eltype(x), V_spec)
    mole_k_p = copy(N_spec) .* one(eltype(x))  # Ensure mole_k_p matches the type of `x`

    S = zero(eltype(x))  # Initialize S with the correct type

    # the last value is the lagrange multiplier

    # T = x[end-1]
    T = x[end]
    # x = [N11, N21 ... Nn1, V1, N12, N22 ... Nn2, V2, N13, N23 ... Nn3, V3, T]

    for k in 0:numPhases-2

        V_k = x[_n+1+k*(_n+1)]
        mole_k = x[1+k*(_n+1):_n+k*(_n+1)]
        V_k_val = V_k.value
        V_tot_val = V_tot.value
        if isa(V_k_val, Float64) && isa(V_tot_val, Float64)
            if V_k_val / V_tot_val < 1e-2
                # println("Vanishing phase : ", V_k_val / V_tot_val) 
            end
        end
        S += S_EOS(T, V_k, mole_k; model)

        V_k_p -= V_k
        mole_k_p = mole_k_p .- mole_k
    end

    S = S_EOS(T, V_k_p, mole_k_p; model) + S
    S_val = S.value
    if isa(S_val, Float64)

        # println(" S: ", S_val, " U: ", U_k_p.value, " V: ", V_k_p.value)
        # S_1 = S_EOS(T, V_spec, N_spec; model)
    end
    return S
    # return -S / R

end

function accumulate_phase_contributions(x, T, Vs, p::Int, model, N_spec; ThermoFunc)
    n = model.Nc
    A_acc = zero(eltype(x))
    
    mole_k_p = copy(N_spec) .* one(eltype(x))
    
    for k in 1:(p-1)
        V_k = Vs[k]
        mole_k = @view x[((k-1)*n + 1):(k*n)]
        A_acc += ThermoFunc(T, V_k, mole_k; model)
        mole_k_p .-= mole_k
    end
    
    return A_acc, mole_k_p
end

function compute_total_qty(x, T, model, numPhases, ScaleFunc, N_spec, tempSpec, total_volume_known, V_spec = nothing; ThermoFunc = A_EOS)
    
    # Apply scaling if needed
    if model.doScale
        _, DeScaler = ScaleFunc
        x = DeScaler(x)
    end

    # Extract system properties
    # T = x[end]
    n = model.Nc
    Vs = extract_Vs(x, n, numPhases, tempSpec)    
    V_total = sum(Vs)

    # Accumulate Helmholtz energy and compute last phase moles
    A_acc, last_phase_moles = accumulate_phase_contributions(x, T, Vs, numPhases, model, N_spec; ThermoFunc)
    V_last_phase = Vs[end]

    if total_volume_known
        V_last_phase = V_spec - V_total
        
        # Correct the total volume, although it will not be used for volume based flashes
        V_total = V_spec
    end

    # Add final phase contribution
    A_p = ThermoFunc(T, V_last_phase, last_phase_moles; model)
    total_A = A_acc + A_p

    return total_A, V_total, T
end


function compute_total_helmholtz(x, T, model, numPhases, ScaleFunc, N_spec, tempSpec, total_volume_known, V_spec = nothing)
    # Apply scaling if needed
    if model.doScale
        _, DeScaler = ScaleFunc
        x = DeScaler(x)
    end

    # Extract system properties
    # T = x[end]
    n = model.Nc
    Vs = extract_Vs(x, n, numPhases, tempSpec)    
    VsVal = ForwardDiff.value.(ForwardDiff.value.(x))
    # @show VsVal n numPhases tempSpec
    V_total = sum(Vs)

    # Accumulate Helmholtz energy and compute last phase moles
    A_acc, last_phase_moles = accumulate_phase_contributions(x, T, Vs, numPhases, model, N_spec; ThermoFunc = A_EOS)
    V_last_phase = Vs[end]

    if total_volume_known
        V_last_phase = V_spec - V_total
        
        # Correct the total volume, although it will not be used for volume based flashes
        V_total = V_spec
    end

    # Add final phase contribution
    A_p = A_EOS(T, V_last_phase, last_phase_moles; model)
    total_A = A_acc + A_p

    return total_A, V_total, T
end

function VT_Q(x; T_spec, V_spec, N_spec, model, numPhases, Scale, ScaleFunc)
    T = T_spec
    total_A, _, T = compute_total_helmholtz(x, T, model, numPhases, ScaleFunc, N_spec, TemperatureSpecified(), true, V_spec)
    Q = total_A
    return Q

end

function Q_lambda(x, lamdba; U_spec, V_spec, N_spec, model, numPhases, Scale, ScaleFunc)
  
    T = -1.0 / lamdba  # Assuming lambda is the inverse temperature, T = -1/λ 
    UV_Q(vcat(x, T); U_spec, V_spec, N_spec, model, numPhases, Scale, ScaleFunc)
end

function Q_stationarity!(F, full_x, Q_lambda)
    # full_x = [x; lambda] — full decision vector
    x = full_x[1:end-1]
    lambda = full_x[end]
    
    # Gradient of Q w.r.t x
    ∇x = ForwardDiff.gradient(x -> Q_lambda(x, lambda), x)
    
    # Derivative of Q w.r.t lambda
    dQ_dlambda = ForwardDiff.derivative(lambda -> Q_lambda(x, lambda), lambda)
    
    # Combine into stationarity conditions
    F[1:end-1] .= ∇x
    F[end] = dQ_dlambda
end

# TODO: Make it more robust. Not fully working yet
function solve_flash_Q_lambda(func, x, cons=nothing; tol=1e-6, maxiter=300, α=1.0, useNewtonJulia=true)
    
    # full_x0 = vcat(x0, lambda0)
    @show x
    x[end] = -1/x[end]  # Initial guess for lambda, assuming T = -1/lambda 
    result = nlsolve(
        (F, x) -> Q_stationarity!(F, x, func),
        x;
        method = :newton,
        xtol = 1e-8,
        ftol = 1e-8,
        autodiff = :forward
    )
    @show result.zero
    x_star = result.zero[1:end-1]
    lambda_star = result.zero[end]
    T_star = -1 / lambda_star
    return x_star, T_star #, result
end

function UV_Q(x; U_spec, V_spec, N_spec, model, numPhases, Scale, ScaleFunc)
    T = x[end]
    total_A, _, T = compute_total_helmholtz(x, T, model, numPhases, ScaleFunc, N_spec, TemperatureNotSpecified(), true, V_spec)
    Q = (U_spec - total_A)/T
    return Q
end

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
    
    # s_V = S_EOS(T, V_vap, N_vap; model)  
    # s_L = S_EOS(T, V_liq, N_liq; model)  
    
    # S_one = S_EOS(T, V_spec, N_spec; model)    
    # sV_val = extract_value(s_V)
    # sL_val = extract_value(s_L)
    # ΔS = s_V - s_L
    # ΔV = V_vap - V_liq
    # S_two = s_V + s_L
    # dpdT_coex  = ΔS/ΔV
    # dGdT_coex = S_one-S_two + V_spec * dpdT_coex
    # dGdT = extract_value(dGdT_coex)
    # pV = EOS.P_EOS(T, V_vap, N_vap; model)
    # pL = EOS.P_EOS(T, V_liq, N_liq; model)
    # p_val = extract_value(pV)
    # pL_val = extract_value(pL)
    # dp = round((p_val - pL_val)/1e6; digits=2)
    # @show sV_val + sL_val, dGdT

    # component index to compute rho_L in A_wrapper
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

function rho_α_Q_funky_test(x; U_spec, V_spec, N_spec, model, numPhases = nothing, Scale = nothing, ScaleFunc = nothing)
    # Extract temperature and vapor fraction from input
    T = x[end]                        # Temperature
    α = clamp(x[model.Nc + 1], 1e-8, 1 - 1e-8)  # Vapor fraction (clamped)
    𝐳 = N_spec ./ sum(N_spec)        # Overall composition
    V_mix = V_spec                   # Total volume
    ρ_mix = sum(N_spec .* model.Mw) / V_mix  # Mixture density (mass basis)

    ρ_vap_mass = x[1:model.Nc] .* ρ_mix       # Mass densities of vapor components
    V_vap = α * V_mix
    V_liq = V_mix - V_vap

    # Moles in each phase
    N_vap = ρ_vap_mass .* V_vap ./ model.Mw
    N_liq = N_spec .- N_vap

    # EOS evaluations
    pG = EOS.P_EOS(T, V_vap, N_vap; model)
    pL = EOS.P_EOS(T, V_liq, N_liq; model)

    μG = EOS.μ_EOS(T, V_vap, N_vap; model)
    μL = EOS.μ_EOS(T, V_liq, N_liq; model)

    uG = EOS.U_EOS(T, V_vap, N_vap; model)
    uL = EOS.U_EOS(T, V_liq, N_liq; model)

    # Core equilibrium equations
    eqns = vcat(
        (pL - pG) / T,
        (μG .- μL) ./ T,
        (uG + uL - U_spec) / T^2
    )
     # Add to energy equation (or alternatively as an extra equation)

    return eqns
end

function rhoG_rhoL_Q(x; U_spec, V_spec, N_spec, model, numPhases = nothing, Scale = nothing, ScaleFunc = nothing)
    # T = x[end]                     # Temperature
    ϵ = 1e-10
    M⁻¹ = 1 / model.Mw[1]
    T, ρG, ρL = x
    ρ_mix = sum(N_spec .* model.Mw) / V_spec
    denom = max(abs(ρL - ρG), ϵ)
    α = clamp((ρL - ρ_mix) / denom, ϵ, 1.0 - ϵ)

    V_G = α * V_spec
    V_L = (1.0 - α) * V_spec

    N_G = ρG * V_G * M⁻¹        
    N_L = N_spec - N_G

    A_G = EOS.A_EOS(T, V_G, [N_G]; model)
    A_L = EOS.A_EOS(T, V_L, [N_L]; model)
    A_total = A_L + A_G
    
    Q = (U_spec - A_total) / T

    return Q
end

function NG_NL_Q(x; U_spec, V_spec, N_spec, model, numPhases = nothing, Scale = nothing, ScaleFunc = nothing)
    # T = x[end]                     # Temperature
    ϵ = 1e-10
    M⁻¹ = 1 / model.Mw[1]
    T, V_G, N_G = x

    V_L = V_spec - V_G
    N_L = N_spec .- N_G
    ρ_mix = sum(N_spec .* model.Mw) / V_spec
    ρG =  N_G * M⁻¹ / (V_G)
    ρL = N_L * M⁻¹ / (V_L)  
    denom = max(abs(ρL - ρG), ϵ)
    α = clamp((ρL - ρ_mix) / denom, ϵ, 1.0 - ϵ)

    # V_G = α * V_spec
    # V_L = (1.0 - α) * V_spec

    # N_G = ρG * V_G * M⁻¹        
    # N_L = N_spec - N_G

    A_G = EOS.A_EOS(T, V_G, [N_G]; model)
    A_L = EOS.A_EOS(T, V_L, [N_L]; model)
    A_total = A_L + A_G
    
    Q = (U_spec - A_total) / T

    return Q
end

function SV_Q(x; S_spec, V_spec, N_spec, model, numPhases, Scale, ScaleFunc)
    T = x[end]
    total_A, _, T = compute_total_helmholtz(x, T, model, numPhases, ScaleFunc, N_spec, TemperatureNotSpecified(), true, V_spec)
    Q = (total_A + T*S_spec)
    return Q

end

function PS_Q(x; P_spec, S_spec, N_spec, model, numPhases, Scale, ScaleFunc)
    T = x[end]
    total_A, V_total, T = compute_total_helmholtz(x, T, model, numPhases, ScaleFunc, N_spec, TemperatureNotSpecified(), false)
    Q = total_A + P_spec * V_total + T * S_spec
    return Q
end

function PH_Q(x; P_spec, H_spec, N_spec, model, numPhases, Scale, ScaleFunc)
    T = x[end]
    total_A, V_total, T = compute_total_helmholtz(x, T, model, numPhases, ScaleFunc, N_spec, TemperatureNotSpecified(), false)
    Q = total_A + P_spec * V_total - H_spec
    return -Q / T
end

function PT_Q(x; P_spec, T_spec, N_spec, model, numPhases, Scale, ScaleFunc)
    T = T_spec
    total_A, V_total, T = compute_total_helmholtz(x, T, model, numPhases, ScaleFunc, N_spec, TemperatureSpecified(), false)
    Q = total_A + V_total * P_spec
    return Q
end

function VT_A_reduced_constrained(
    x;
    V_spec,
    T_spec,
    N_spec,
    model,
    numPhases,
    Scale,
    ScaleFunc,
)
    _n = model.Nc
    (; R, doScale) = model
    Scaler, DeScaler = ScaleFunc
    doScale = true
    if doScale
        x = DeScaler(x)
    end

    mole_k_p = copy(N_spec) .* one(eltype(x))  # Ensure mole_k_p matches the type of `x`

    Q = zero(eltype(x))  # Initialize S with the correct type
    V_p = convert(eltype(x), V_spec)
    # x = [N11, N21 ... Nn1, V1, N12, N22 ... Nn2, V2, N13, N23 ... Nn3, V3, V4, T]
    pEOS = EOS.P_EOS(T_spec, V_p, mole_k_p; model)

    # pEOS = ForwardDiff.value(pEOS)
    # if isa(pEOS, Float64)
    #     @show pEOS
    # end
    for k = 0:numPhases-2

        V_k = x[_n+1+k*(_n+1)]
        mole_k = x[1+k*(_n+1):_n+k*(_n+1)]

        A_k = A_EOS(T_spec, V_k, mole_k; model)
        # Q += (A_k + V_k * P_spec)
        pEOS = EOS.P_EOS(T_spec, V_k, mole_k; model)

        pEOS = ForwardDiff.value(pEOS)
        # if isa(pEOS, Float64)
        #     @show pEOS
        # end
        Q += A_k
        V_p -= V_k
        mole_k_p .= mole_k_p .- mole_k
    end

    # It could be P_spec as well
    A_p = A_EOS(T_spec, V_p, mole_k_p; model)

    Q = (A_p + Q)

    return Q

end

"""
Given we have `n` components and `p` phases, the vector `x` is structured as follows:
x = [N₁₁, ..., Nₙ₁,   # Phase 1 (n values)
     N₁₂, ..., Nₙ₂,   # Phase 2 (n values)
     ...
     N₁₍ₚ₋₁₎, ..., Nₙ₍ₚ₋₁₎,   # Phase p−1
     V₁, ..., Vₚ,     # Volumes of all p phases
     T]              # Temperature

"""
# Access functions. x is the big array containing all the variables.
# n is number of components, p is number of phases
@inline extract_Ns(x, n, p) = [x[(α - 1)*n + 1 : α*n] for α in 1:(p - 1)]

# @inline function extract_Ns_matrix(x, n, p)
#     @view reshape(x[1:n*(p-1)], n, p-1)
# end

abstract type SpecMode end
struct TemperatureSpecified <: SpecMode end
struct TemperatureNotSpecified <: SpecMode end
struct VolumeSpecified <: SpecMode end

function extract_Vs(x, n, p, ::TemperatureNotSpecified)
    V_start = n * (p - 1) + 1    
    # @view x[V_start : V_start + p - 1 - last_phase_volume_known]
    @view x[V_start : end - 1]
end

function extract_Vs(x, n, p, ::TemperatureSpecified)
    V_start = n * (p - 1) + 1
    # @view x[V_start : V_start + p - 1 - last_phase_volume_known]
    @view x[V_start : end]
end

# @inline extract_T(x) =  x[end]

function SV_A_reduced_constrained(x; V_spec, S_spec, N_spec, model, numPhases, Scale, ScaleFunc)
    _n = model.Nc
    (; R, doScale) = model
    Scaler, DeScaler = ScaleFunc
    doScale = true
    if doScale
        x = DeScaler(x)
    end
   
    mole_k_p = copy(N_spec) .* one(eltype(x))  # Ensure mole_k_p matches the type of `x`

    Q = zero(eltype(x))  # Initialize S with the correct type
    # H_tot = zero(eltype(x))  # Initialize H_tot with the correct type
    # the last value is the lagrange multiplier

    # T = x[end-1]
    # T = T_spec #x[end]
    T = x[end]  
    V_p = x[end-1]  
    
    # x = [N11, N21 ... Nn1, N12, N22 ... Nn2, N13, N23 ... Nn3, V1, V2, ..., Vn-1, T]

    for k in 0:numPhases-2

        V_k = x[_n+1+k*(_n+1)]
        
        mole_k = x[1+k*(_n+1):_n+k*(_n+1)]
                
        A_k = A_EOS(T, V_k, mole_k; model)
        # H_tot += H_EOS(T, V_k, mole_k; model)
        # Q += (A_k + V_k * P_spec)
        Q += A_k

        mole_k_p = mole_k_p .- mole_k
    end
    
    # last phase A
    A_p = A_EOS(T, V_p, mole_k_p; model)
    # H_tot += H_EOS(T, V_p, mole_k_p; model)
    Q = A_p + Q # Total Helmholtz energy is the sum of all Helmholtz energies
    
    # V_total = sum(x[end-numPhases:end-1]) # Total volume is the sum of all volumes    
    Q += (T * S_spec ) # A + T S_spec
    # alpha = 0.1
    return Q #+ alpha * (H_spec - H_tot)^2

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

function Optimize(func, x, cons=nothing; tol=1e-6, maxiter=50, α=1.0)

    # to optimize, we need to to find the root of the gradient of func
    g(x) = ForwardDiff.gradient(func, x)
    H(x) = ForwardDiff.hessian(func, x)

    iter = 0
    x_new = x
    for i in 1:maxiter

        hess = H(x)
        Δx = -(hess \ g(x))
        x_new = x + α * Δx
        if norm(x_new .- x) < tol
            x_new = x
            break
        end
        x = x_new
        iter += 1

    end


    if iter == maxiter
        println("Max iterations reached without convergence.")
    else
        println("Converged in $iter iterations.")
    end

    return x

end

function solveLinearSystem(A, b, Pl)
    prob = LinearProblem(A, b)
    sol = solve(prob, KrylovJL_GMRES(), Pl=Pl)
    sol.u
end

# TODO: Make it work for any numbe of phases . Currently it is hard coded for 2 phases
function EquilibriumRatio(N_L, N_spec)
    N_V = N_spec .- N_L
    x_k_l = N_L ./ sum(N_L)
    x_k_v = N_V ./ sum(N_V)
    K = x_k_v ./ x_k_l
    return K
end

function VapourPhaseFraction(N_vapour, N_spec)
    N_vapour = sum(N_vapour)
    N_tot = sum(N_spec)
    return N_vapour / N_tot
end


function Optimize2(func, x, cons=nothing; tol=1e-10, maxiter=300, α=1.0, useNewtonJulia=true)

    # to optimize, we need to to find the root of the gradient of func
    g(x) = ForwardDiff.gradient(func, x)
    H(x) = ForwardDiff.hessian(func, x)

    # With finite difference, it works only for some cases.

    # g(x) = FiniteDiff.finite_difference_gradient(func, x)
    # H(x) = FiniteDiff.finite_difference_hessian(func, x)

    sol = nlsolve(g, H, x, xtol=tol, ftol=tol, iterations=1000, method=:trust_region)
    return sol.zero
    
end

function OptimizeWithCons(func, x, cons; tol=1e-6, maxiter=300, α=1.0, useNewtonJulia=true)
    # On all the test cases considered, we got a constant value of λ * T. It might be of some use.
    
    λT = -1.0

    # T is stored in x[end] and λ = -1/R but we are maximising S/R, not S
    lagrangian(x) = func(x) + λT / x[end] * cons(x)

    
    # @show g1 g2 g3
    # error("Hurray")
    g(x) = ForwardDiff.gradient(lagrangian, x)
    H(x) = ForwardDiff.hessian(lagrangian, x)

    iter = 1
    # sol = nlsolve(g, H, x, xtol=tol, ftol=tol, iterations=1000, method=:newton, linesearch=LineSearches.BackTracking(order=3))    

    sol = nlsolve(g, H, x, xtol=tol, ftol=tol, iterations=1000, method=:newton, linesearch=LineSearches.BackTracking(order=3))

    converged = sol.x_converged || sol.f_converged
    # @show sol.iterations, sol.zero, converged
    if !converged
        println("Warning: Optimization did not converge.")        
    end
    # sol = nlsolve(g, H, x, xtol=tol, ftol=tol, iterations=1000, method=:trust_region)
    # sol = nlsolve(g, H, x, xtol=tol, ftol=tol, iterations=2000, method=:newton, linesearch=LineSearches.StrongWolfe())
    
    return sol.zero
    
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
    sol = nlsolve(g, H, x, xtol=tol, ftol=tol, iterations=4000, method=:newton, linesearch=LineSearches.BackTracking(order=3))
    converged = sol.x_converged || sol.f_converged
    # norm_converged = norm(g(sol.zero))
    # @show sol.iterations sol.x_converged, sol.f_converged, norm_converged
    if !converged        
        @warn "Flash did not converge."
        false, nothing       
    end
    # @show g(sol.zero)
    # Hess = round.(H(sol.zero), digits=2)
    # @show Hess
    return true, sol.zero, sol.iterations
end

function OptimizeHelmholtzPT(func, x, cons; tol=1e-6, maxiter=300, α=1.0, useNewtonJulia=true)
    
    @show x
    # T is stored in x[end] and λ = -1/R but we are maximising S/R, not S
    lagrangian(x) = func(x)

    g(x) = ForwardDiff.gradient(lagrangian, x)
    H(x) = ForwardDiff.hessian(lagrangian, x)
   
    # sol = nlsolve(g, H, x, xtol=tol, ftol=tol, iterations=1000, method=:trust_region)
    sol = nlsolve(g, H, x, xtol=tol, ftol=tol, iterations=1000, method=:newton, linesearch=LineSearches.BackTracking(order=3))
    @show sol
    return sol.zero
end


convergence(x, y; tol) = norm(y .- x) < tol

@exportAll()
end