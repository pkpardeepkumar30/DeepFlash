# Constants
module EOS

using StaticArrays
using ForwardDiff
using Polynomials
using LinearAlgebra
using Statistics
using NLsolve
# using LinearSolve
# using Krylov
# using QuadGK
# using PositiveFactorizations
using FixedPointAcceleration
# using ..uvn
# using ..Utils

using ExportAll


function PengRobinson(; Mw, ρ_c, T_c, P_c, ω, δ::Matrix{Float64}, α::Matrix{Float64}, components=[], T_c_mix = nothing, P_c_mix = nothing, doScale=false)

    # R = 8.314  # Universal gas constant in J/(mol*K)
    # R = 8.3144621  # Universal gas constant in J/(mol*K)    
    R = 8.31446261815324
    T0 = 298.15 # Reference temperature in K
    P0 = 1e5 # Reference pressure in Pa    
    u0 = -2478.95687512 # J/mol
    Ωₐ = 0.45724
    Ωᵦ = 0.0778
    Δ₁ = 1 + sqrt(2)
    Δ₂ = 1 - sqrt(2)
    # @show ω
    ω_coeffs1 = SA[0.37464, 1.54226, -0.26992]
    ω_coeffs2 = SA[0.379642, 1.48503, -0.164423, 0.016667]
    Nc = length(Mw)

    #  m_i = ω[i] < 0.5 ?
    #       dot(ω_coeffs1, [1.0, ω, ω^2]) :
    #       dot(ω_coeffs2, [1.0, ω, ω^2, ω^3])

    (; R, Nc, Mw, ρ_c, T_c, P_c, T_c_mix, P_c_mix, Ωₐ, Ωᵦ, Δ₁, Δ₂, ω, ω_coeffs1, ω_coeffs2, δ, α, T0, P0, u0, components, doScale)

end

function SRK(; Mw, ρ_c, T_c, P_c, ω, δ::Matrix{Float64}, α::Matrix{Float64}, components=[], doScale=false)

    # R = 8.314  # Universal gas constant in J/(mol*K)
    # R = 8.3144621  # Universal gas constant in J/(mol*K)    
    R = 8.31446261815324
    T0 = 298.15 # Reference temperature in K
    P0 = 1e5 # Reference pressure in Pa    
    u0 = -2478.95687512 # J/mol
    Ωₐ = 0.42748
    Ωᵦ = 0.08664
    Δ₁ = 1
    Δ₂ = 0
    # @show ω
    ω_coeffs1 = [0.48508, 1.55171, -0.15613]
    ω_coeffs2 = [0.48508, 1.55171, -0.15613, 0.0]
    Nc = length(Mw)

    (; R, Nc, Mw, ρ_c, T_c, P_c, Ωₐ, Ωᵦ, Δ₁, Δ₂, ω, ω_coeffs1, ω_coeffs2, δ, α, T0, P0, u0, components, doScale)

end

# Function to calculate a_i(T)
function a_i(T; i, model)
    # @show typeof(model)
    T_c, P_c, ω = (model.T_c[i], model.P_c[i], model.ω[i])
    (; R, ω_coeffs1, ω_coeffs2) = model

    m_i = ω < 0.5 ?
          dot(ω_coeffs1, SA[1.0, ω, ω^2]) :
          dot(ω_coeffs2, SA[1.0, ω, ω^2, ω^3])
    #   0.37464 + 1.54226 * ω - 0.26992 * ω^2 :
    #   0.3796 + 1.485 * ω - 0.1644 * ω^2 + 0.01667 * ω^3

    T_r = T / T_c
    Ωₐ = model.Ωₐ
    multiplier = Ωₐ * (R^2 * T_c^2) / P_c
    a_i = multiplier * (1 + m_i * (1 - sloppysqrt(T_r)))^2
    return a_i
end

function da_i_dT(T; i, model)
    # @show typeof(model)
    T_c, P_c, ω = (model.T_c[i], model.P_c[i], model.ω[i])
    (; R, ω_coeffs1, ω_coeffs2) = model

    m_i = ω < 0.5 ?
          dot(ω_coeffs1, SA[1.0, ω, ω^2]) :
          dot(ω_coeffs2, SA[1.0, ω, ω^2, ω^3])
    #   0.37464 + 1.54226 * ω - 0.26992 * ω^2 :
    #   0.3796 + 1.485 * ω - 0.1644 * ω^2 + 0.01667 * ω^3

    T_r = T / T_c
    Ωₐ = model.Ωₐ
    multiplier = Ωₐ * (R^2 * T_c^2) / P_c
    a_i = -2 * multiplier * (1 + m_i * (1 - sloppysqrt(T_r))) * m_i / sloppysqrt(T_c) / 2 / sloppysqrt(T)
    return a_i
end

# Function to calculate b_i
function b_i(; i, model)
    R, T_c, P_c = (model.R, model.T_c[i], model.P_c[i])

    # Ωᵦ = 0.0778
    b = model.Ωᵦ * (R * T_c) / P_c
    # println("b: ", b)
    return b
end

# Function to calculate a_ij
function a_ij(a_i, a_j, δ_ij)
    return (1 - δ_ij) * sloppysqrt(a_i * a_j)
end

# Function to calculate b_ij
function b_ij(b_i, b_j, l_ij)
    return 0.5*(b_i + b_j) #*(1 - l_ij)
end

# Function to calculate a(T) as a sum over all components with named arguments
function a(T, x; model)
    n = model.Nc
    δ = model.δ

    # ∑∑xᵢxⱼaᵢⱼ = sum(x[i] * x[j] * a_ij(a_i(T; i, model), a_i(T; i=j, model), δ[i, j]) for i in 1:n for j in 1:n)

    # return ∑∑xᵢxⱼaᵢⱼ
    # @show typeof(x) length(x)
    ∑∑xᵢxⱼaᵢⱼ = 0.0
    for i in 1:n
        aᵢ = a_i(T; i, model)
        # println("aᵢ: ", aᵢ)
        for j in 1:n
            if i == j
                aⱼ = aᵢ  # Use aᵢ directly if i == j to avoid recomputation
            else
                aⱼ = a_i(T; i=j, model)
            end
            aᵢⱼ = a_ij(aᵢ, aⱼ, δ[i, j])
            ∑∑xᵢxⱼaᵢⱼ += x[i] * x[j] * aᵢⱼ
            # ∑∑xᵢxⱼaᵢⱼ =  ∑∑xᵢxⱼaᵢⱼ + x[i] * x[j] * aᵢⱼ
        end
    end
    
    return ∑∑xᵢxⱼaᵢⱼ
end

# Function to calculate b(T) as a sum over all components with named arguments
function b_2(x; model)
    n = model.Nc
    l = zero(model.δ)

    # ∑∑xᵢxⱼaᵢⱼ = sum(x[i] * x[j] * a_ij(a_i(T; i, model), a_i(T; i=j, model), δ[i, j]) for i in 1:n for j in 1:n)

    # return ∑∑xᵢxⱼaᵢⱼ
    # @show typeof(x) length(x)
    ∑∑xᵢxⱼbᵢⱼ = 0.0
    for i in 1:n
        bᵢ = b_i(; i, model)
        # println("aᵢ: ", aᵢ)
        for j in 1:n
            if i == j
                bⱼ = bᵢ  # Use aᵢ directly if i == j to avoid recomputation
            else
                bⱼ = b_i(; i=j, model)
            end
            bᵢⱼ = b_ij(bᵢ, bⱼ, l[i, j])
            ∑∑xᵢxⱼbᵢⱼ += x[i] * x[j] * bᵢⱼ
            # ∑∑xᵢxⱼaᵢⱼ =  ∑∑xᵢxⱼaᵢⱼ + x[i] * x[j] * aᵢⱼ
        end
    end
    return ∑∑xᵢxⱼbᵢⱼ
end

# Function to calculate b(x) with named arguments
function b(x; model)

    R, T_c, P_c, nc = (model.R, model.T_c, model.P_c, model.Nc)

    #  ∑xᵢbᵢ = sum(x[i] * 0.0778 * (R * T_c[i]) / P_c[i] for i in 1:nc)
    # summand = @. x * 0.0778 * (R * T_c) / P_c
    # ∑xᵢbᵢ = sum(summand)
    ∑xᵢbᵢ = sum(x[i] .* b_i(; i, model) for i in 1:nc)
    return ∑xᵢbᵢ
end

evalcoeff(coeffs, T) = evalpoly(T, coeffs)

function eval∫coeff(coeffs, T)
    n = length(coeffs)
    div1 = NTuple{n,Int}(1:n)
    ∫poly = coeffs ./ div1
    return evalpoly(T, ∫poly) * T
end

function eval∫coeffT(coeffs, T, lnT=sloppylog(T))
    n = length(coeffs)
    div1 = NTuple{n - 1,Int}(1:(n-1))
    A = first(coeffs)
    coeffs1 = coeffs[2:end]
    ∫polyT = coeffs1 ./ div1
    return evalpoly(T, ∫polyT) * T + A * lnT
end

function a_ideal_clapeyron(V, T, z; model, u0=-2478.95687512, strange_factor=1.0)
    #x = z/sum(z)
    polycoeff = model.α

    V⁻¹ = 1 / V
    V0 = 1.0 # -u0 / model.P0
    res = zero(V + T + first(z))
    Σz = sum(z)
    R̄ = model.R
    RT = R̄ * T
    R̄⁻¹ = 1 / R̄
    RT⁻¹ = 1 / RT
    T0 = 298.15
    lnT0 = sloppylog(T0)
    lnT = sloppylog(T)
    # strange_factor = 0.3677
    n = model.Nc
    for i in 1:n
        coeffs = polycoeff[i, :]
        H = (eval∫coeff(coeffs, T) - eval∫coeff(coeffs, T0)) * RT⁻¹
        TS = (eval∫coeffT(coeffs, T, lnT) - eval∫coeffT(coeffs, T0, lnT0)) * R̄⁻¹
        α₀ᵢ = H - TS + lnT - lnT0
        res += z[i] * α₀ᵢ
        res += z[i] * sloppylog(z[i] * strange_factor * V0 / V) # Look at Reynolds Thermodynamics for this
    end
    return res # we want Aʳ/RT, not Aʳ/nRT
end

function xlogx(x::Real, k=one(x))
    _0 = zero(x)
    iszero(x) && return _0
    ifelse(x > _0, x * Base.sloppylog(max(_0, k * x)), _0 / _0)
end

function approximatelog(x)
    # if x > 1.0
    #     x = 1/x
    #     x = x - 1.0
    # elseif x < 1.0
    #      x = x - 1.0
    # end

    return x - x ^2 / 2 + x^3 / 3 # - x^4 / 4  + x^5 / 5 - x^6 / 6 + x^7 / 7 - x^8 / 8 + x^9 / 9

end

function sloppylog(x)
    # @show x
    # return approximatelog(x-1)
    _0 = abs(eps(eltype(x)))
    # _0 = zero(x)
    iszero(x) && return _0
    ifelse(x > _0, log(max(_0, x)), _0 / _0)
    # log(abs(x))
end

function sloppysqrt(x)
    sqrt.(abs.(x))
end

function U_ideal(T, V, z; model)
    #x = z/sum(z)
    (; α, R, T0, Nc, u0) = model
    
    N = sum(z)

    term1 = -N * R * (T - T0) # -RT0 is the reference, u0
    ∑ = sum

    
    # term2 = 0.0    
    
    # if V > 1e-10
        term2 = sum(1:Nc) do i
            z[i] > 0 && return z[i] * sum(k -> α[i, k+1] * (T^(k + 1) - T0^(k + 1)) / (k + 1), 0:3)
            return 0.0
        end
    # end
    # term2 = ∑(1:Nc) do i
    #     if z[i] > 0 
    #         z[i] * ∑(0:3) do k
    #             α[i, k+1] * (T^(k + 1) - T0^(k + 1)) / (k + 1)
    #         end
    #     else
    #         0.0
    #     end
    # end

    u_ideal = term1 + term2 + N * u0
    return u_ideal
end

function DoesEntropyIncrease(x, increment; model)
    TotalQty(x .+ increment; model, ThermoFunc=S_EOS) > TotalQty(x; model, ThermoFunc=S_EOS)
end

function TotalQty(x; model, ThermoFunc)
    _n = model.Nc
    numPhases = Int(length(x) / (_n + 2))

    S = zero(eltype(x))
    for k in 0:numPhases-1
        U_k = x[_n+2+k*(_n+2)]
        V_k = x[_n+1+k*(_n+2)]
        mole_k = x[1+k*(_n+2):_n+k*(_n+2)]
        S = S + Property_UVN(U_k, V_k, mole_k; model, ThermoFunc)
    end
    return S
end

function TotalChemicalPot(x; model)
    _n = model.Nc
    numPhases = Int(length(x) / (_n + 2))

    S = zeros(eltype(x), _n)
    for k in 0:numPhases-1
        U_k = x[_n+2+k*(_n+2)]
        V_k = x[_n+1+k*(_n+2)]
        mole_k = x[1+k*(_n+2):_n+k*(_n+2)]
        S .= S .+ Property_UVN(U_k, V_k, mole_k; model, ThermoFunc=μ_EOS)
    end
    return S
end


# f_i = P * ϕ * z_frac
# log(f_i) = log(P * ϕ * z_frac) = log(P) + log(ϕ) + log(z_frac)

function fugacity(T, V, z; model)
    z_frac = z ./ sum(z)
    fug_coeff = exp.(lnϕ(T, V, z; model))
    P = P_EOS(T, V, z; model)
    # @show fug_coeff P
    fugacity = P * fug_coeff .* z_frac
    return fugacity

end

function log_fugacity(T, V, z; model)
    z_frac = z ./ sum(z)
    log_fug_coeff = lnϕ(T, V, z; model)
    P = P_EOS(T, V, z; model)
    log_fugacity = sloppylog(P) .+ log_fug_coeff .+ sloppylog.(z_frac)
    return log_fugacity

end

function newton_raphson(f, x0; tol=1e-8, max_iter=1000,)
    x = x0
    h = sqrt(eps(x))
    for i in 1:max_iter
        # Calculate f(x) and f'(x) using finite differences
        fx = f(x)

        dfdx = (f(x + h)[1] - f(x - h)[1]) / h

        # Check if the derivative is too small (to avoid division by zero)
        if norm(dfdx) < tol
            println("Derivative too small, stopping.")
            return (; zero=[x])
        end

        # Newton-Raphson update
        x_new = @. x - fx / dfdx

        # Check for convergence
        if norm(x_new .- x) < tol
            # println("Converged in $i iterations.")
            return (; zero=[x_new])
        end

        # Update x for the next iteration
        x = x_new
    end

    # If the maximum iterations are reached without convergence
    # println("Max iterations reached without convergence.")

    return (; zero=[x])
end

function halleys_method(f, x0; tol=1e-7, max_iter=100)
    x = x0
    for i in 1:max_iter
        # Compute f(x), f'(x), and f''(x)
        fx = f(x)
        fx_prime = ForwardDiff.derivative(f, x)
        fx_double_prime = ForwardDiff.derivative(x -> ForwardDiff.derivative(f, x), x)

        # Calculate the Halley step
        denominator = 2 * (fx_prime^2) - fx * fx_double_prime
        if denominator == 0
            println("Denominator zero; possible convergence issue.")
            break
        end

        h = (2 * fx * fx_prime) / denominator
        x -= h  # Update x

        # Check for convergence
        if abs(h) < tol
            println("Converged to $x in $i iterations.")
            return x
        end
    end
    println("Did not converge within $max_iter iterations.")
    return x
end

function GetTemperatureForSpecifiedUVWithFD(; U, V, z, model, T_guess)

    UFunc(T) = U_EOS(T, V, MVector(z...); model)

    function create_prob(x::SVector)
        T, = x
        UFunc(T) - U
    end
    # out = newton_raphson(prob, T_guess)
    # @show out.iterations
    out = nlsolve(create_prob, [T_guess], method=:newton)
    T = out.zero[1]
end

function GetVolumeForSpecifiedPT(; P, T, z, model, V_guess)
    function prob(x)
        V, = x
        [P_EOS(T, 1.0, z ./ V; model) - P]
    end

    V_guess = convert(eltype(T), V_guess)
    out = nlsolve(prob, [V_guess], ftol=1e-12, xtol=1e-12, method=:newton, linesearch=LineSearches.BackTracking(order=3))
    V = out.zero[1]
end

function GetTemperatureForSpecifiedUV(; U, V, z, model, T_guess, verbose=false)
    UFunc(T) = U_EOS(T, V, MVector(z...); model)

    function create_prob_T(x::MVector)
        T, = x
        UFunc(T) - U
    end

    T_guess = convert(eltype(U), T_guess)
    out = nlsolve(create_prob_T, MVector([T_guess]...), method=:newton)
    if verbose
        return out.zero[1], out.iterations
        # println(out.iterations)
    end
    T = out.zero[1]
end

function Property_UVN(U, V, z; model, ThermoFunc, T_guess=300.0, verbose=false)
    # T = GetTemperatureForSpecifiedUVWithFD(; U, V, z, model, T_guess)
    T = 300.0
    if verbose
        T, iterations =  GetTemperatureForSpecifiedUV(; U, V, z, model, T_guess, verbose)
        return ThermoFunc(T, V, z; model), iterations
    else
        T = GetTemperatureForSpecifiedUV(; U, V, z, model, T_guess, verbose)
        return ThermoFunc(T, V, z; model)
    end
    
end

function Property_PTN(P, T, z; model, ThermoFunc, T_guess=300.0)
    # T = GetTemperatureForSpecifiedUVWithFD(; U, V, z, model, T_guess)
    T = GetTemperatureForSpecifiedUV(; U, V, z, model, T_guess)
    return ThermoFunc(T, V, z; model)
end

function S_ideal(T, V, z; model)

    (; α, R, T0, P0, Nc) = model
    term1 = 0.0
    for Ni in z
        # if Ni > 0
            Ni > 0 && (term1 += R * Ni * sloppylog(V * P0 / (Ni * R * T)))
        # end
    end
    term2 = sum(z[i] > 0 ? z[i] * (α[i, 1] * sloppylog(T / T0) + sum([α[i, k+1] * (T^(k) - T0^(k)) / k for k in 1:3])) : 0 for i in 1:Nc)

    # term1 = R * sum(-Ni * sloppylog((Ni * R * T)) for Ni in z)
    # term2 = sum(z[i] * (α[i, 1] * sloppylog(T) + sum([α[i, k+1] * (T^(k)) / k for k in 1:3])) for i in 1:Nc)
    s_ideal = term1 + term2

    return s_ideal
end

function a_ideal(T, V, z; model, u0=-2478.95687512)

    RT = model.R * T
    # RT = model.R * model.T0

    u_ideal = U_ideal(T, V, z; model) #term1 + term2 + N * u0
    s_ideal = S_ideal(T, V, z; model) #term3 + term4
    return (u_ideal - T * s_ideal) / RT
end

# F = Aʳ / RT, reduced(non-dimensionalised) Helmholtz energy
function a_res(T, V, z; model)
    RT = model.R * T
    # @show T, V
    n = sum(z)    
    if isapprox(n, 0.0; atol = eps(eltype(n))) || isapprox(V, 0.0; atol = eps(eltype(V)))
        return 0.0
    end
    x = z ./ n
    δ₁ = model.Δ₁
    δ₂ = model.Δ₂
    bₘ = b(x; model)
    aₘ = a(T, x; model) # D = a = n²aₘᵢₓ
    term2 = aₘ * n / (RT * bₘ * (δ₁ - δ₂))
    # B = n * bₘ
    # D = n^2 * aₘ
    # @show B.value
    term1 = -n * sloppylog(1 - n * bₘ / V)
    term3 = sloppylog((1 + δ₁ * n * bₘ / V) / (1 + δ₂ * n * bₘ / V))

    return term1 - term2 * term3
end

function massieu_potential(T, V, z; model )
    return U_EOS(T, V, z; model) + T * S_EOS(T, V, z; model)
end

function μ_res_michelsen(i, T, V, z; model)
    R = model.R
    n = sum(z)
    x = z ./ n
    nc = length(z)
    RT = R * T
    δ₁ = model.Δ₁
    δ₂ = model.Δ₂
    # Compute mixture properties
    bₘ = b_2(x; model)
    aₘ = a(T, x; model)

    B = n * bₘ
    D = n^2 * aₘ
    
    f = (1 / (R*B*(δ₁ - δ₂))) * log((V + δ₁ * B) / (V + δ₂ * B))
    g = log((V - B) / V)
    g_B = -1 / (V - B)

    # Compute B_i
    bᵢ = b_i(i=i, model=model)
    sum_zj_bij = 0.0
    l = zero(model.δ)
    for j in 1:nc
        bⱼ = b_i(;i=j, model)
        bᵢⱼ = b_ij(bᵢ, bⱼ, l[i,j])
        sum_zj_bij += z[j] * bᵢⱼ
    end
    B_i = (2 * sum_zj_bij - B) / n

    # Compute D_i
    D_i = 0.0
    aᵢ = a_i(T; i=i, model=model)
    for j in 1:nc
        aⱼ = a_i(T; i=j, model)
        aᵢⱼ = a_ij(aᵢ, aⱼ, model.δ[i,j])
        D_i += z[j] * aᵢⱼ
    end
    D_i *= 2
    # @show D_i
    f_V = (1 / (R * B * (δ₁ - δ₂))) * (1 / (V + δ₁ * B) - 1 / (V + δ₂ * B))
    f_B = -(f + V*f_V) / B
    # Final derivative components
    F_n = -g
    F_B = -n * g_B - (D / T) * f_B
    F_D = -f / T

    ∂F_∂zᵢ = F_n + F_B * B_i + F_D * D_i
    return RT * ∂F_∂zᵢ
end

function chem_pot2(T, V, z; model)
    nc = length(z)
    μ = zeros(nc)
    RT = model.R * T
    for i in 1:nc
        μ[i] = μ_res_michelsen(i, T, V, z; model)        # @show μᵢ
    end
    F_ideal = z -> a_ideal(T, V, z; model)
    ∂F_N_ideal = ForwardDiff.gradient(F_ideal, z)
    μ_ideal = RT * ∂F_N_ideal
    total_μ = μ .+ μ_ideal
    μ2 = EOS.chem_pot(T, V, z; model)
    diff = total_μ .- μ2
    division = total_μ ./ μ2
    @show total_μ μ2
    return nothing
end

function A_EOS(T, V, z; model)

    if is_volume_or_moles_zero(V, z)
        return eps(eltype(V))
    end
    # n = sum(z)
    RT = model.R * T

    a_res_val = a_res(T, V, z; model)
    a_ideal_val = a_ideal(T, V, z; model)
    A = RT * (a_res_val + a_ideal_val)

end

"""
    sound_speed(T, V, z; model)
Calculate the speed of sound in a mixture using the EOS model.
u = (∂p/∂ρ)_s   
where p is the pressure, ρ is the density, and s is the entropy.
"""

function sound_speed(T, V, z; model)
    total_mass  = sum(z .* model.Mw)  # Calculate the total mass of the mixture
    ρ_mix = total_mass / V  # Calculate the mixture density
    pr(vt) = press(vt[2], vt[1], z; model)   # press(T, V, z)
    se(vt) = entropy(vt[2], vt[1], z; model) # entropy(T, V, z)

    V_T = [V, T]
    ∂p_V, ∂p_T = ForwardDiff.gradient(pr, V_T)

    # ∂s_V, ∂s_T = ForwardDiff.gradient(se, V_T)
    # c² = (∂p_V + ∂p_T * (-∂s_V/∂s_T)) *(-V^2 / total_mass)

    C_v = Cv(T, V, z; model)  # Heat capacity at constant volume
    c² = (-∂p_V + (T/C_v)*∂p_T^2) * V / ρ_mix
    if c² < 0
        @warn "Negative sound speed squared: $c² at T=$T, V=$V, z=$z"
        return 1e-2
    end
    c = sqrt( c² )

    return c
end

function sound_speed2(T, V, z; model)
    total_mass  = sum(z .* model.Mw)  # Calculate the total mass of the mixture
    ρ_mix = total_mass / V  # Calculate the mixture density
    pr(vt) = press(vt[2], vt[1], z; model)   # press(T, V, z)
    se(vt) = entropy(vt[2], vt[1], z; model) # entropy(T, V, z)

    V_T = [V, T]
    ∂p_V, ∂p_T = ForwardDiff.gradient(pr, V_T)

    # ∂s_V, ∂s_T = ForwardDiff.gradient(se, V_T)
    # c² = (∂p_V + ∂p_T * (-∂s_V/∂s_T)) *(-V^2 / total_mass)

    C_v = Cv(T, V, z; model)  # Heat capacity at constant volume
    c² = (-∂p_V + (T/C_v)*∂p_T^2) * V / ρ_mix
    
    c = sqrt( c² )

    return c
end

function sound_speed3(T, V, z; model)
    total_mass  = sum(z .* model.Mw)  # Calculate the total mass of the mixture
    ρ_mix = total_mass / V  # Calculate the mixture density

    pr(vt) = press(vt[2], vt[1], z; model)   # press(T, V, z)
    se(vt) = entropy(vt[2], vt[1], z; model) # entropy(T, V, z)

    V_T = [V, T]
    ∂p_V, ∂p_T = ForwardDiff.gradient(pr, V_T)

    ∂ρ_p = -ρ_mix/(∂p_V * V)  # ∂ρ/∂p = -1/(∂p/∂ρ * V)
    ∂ρ_T = (ρ_mix/V)*(∂p_T / ∂p_V)  # ∂ρ/∂T = (ρ/V) * (∂p/∂T / ∂p/∂ρ)

    C_p = Cp(T, V, z; model)
    c_p = C_p / total_mass
    
    inv_c² = ∂ρ_p - (T/(ρ_mix^2 * c_p)) * ∂ρ_T^2

    # ∂s_V, ∂s_T = ForwardDiff.gradient(se, V_T)
    # c² = (∂p_V + ∂p_T * (-∂s_V/∂s_T)) *(-V^2 / total_mass)

      # Heat capacity at constant volume
    # c² = (-∂p_V + (T/C_p)*∂p_T^2) * V / ρ_mix
    
    c = sqrt( 1/inv_c² )

    return c
end

#------------------------------------------#
"""
    compute_phase_A_vol_from_phasic_densities(N_i, M_i, rhoG_i, rhoL_i, Vtot)
    - `N_i` is the total moles of component i in the mixture    
    - `M_i` is the molar mass of component i
    - `rhoG_i` is the density of component i in gas phase (= `NG_i M_i/ V_G`)
    - `rhoL_i` is the density of component i in liquid phase
    - `Vtot` is the total volume

"""
function compute_phase_A_vol_from_phasic_densities(N_i, M_i, rhoG_i, rhoL_i, Vtot)    
    numerator = (Vtot * rhoL_i - N_i * M_i )
    denominator = (rhoL_i - rhoG_i)
  
    # ep = sign(denominator)*1e-12
    Vᴬ = numerator / (denominator)
end

function A_wrapper(T, 𝛒𝐀, ρB; indx = 1, ρ_mix, V_mix, 𝐳, model)

    # n = length(𝐳)
    𝐌 = model.Mw
    𝛒ᴬ = copy(𝛒𝐀)       # ensure we don't modify input
    # 𝛒ᴸ = copy(𝛒𝐋)       # ensure we don't modify input
    𝐍 = compute_mole_vector_from_density(ρ_mix, V_mix, 𝐳, 𝐌)    
    
    Vᴬ = compute_phase_A_vol_from_phasic_densities(𝐍[indx], 𝐌[indx], 𝛒ᴬ[indx], ρB, V_mix)    
    # Vᴸ = V_mix - Vᴳ

    Aᴬ = Vᴬ * A_EOS(T, 1.0, 𝛒ᴬ./ 𝐌; model)
    # Aᴸ = Vᴸ * A_EOS(T, 1.0, 𝛒ᴸ./ 𝐌; model)
    # A = Aᴳ + Aᴸ
    return Aᴬ
end

function A_wrapper2(T, 𝛒𝐀, αA; indx = 1, ρ_mix, V_mix, 𝐳, model)

    # n = length(𝐳)
    𝐌 = model.Mw
    𝛒ᴬ = copy(𝛒𝐀)       # ensure we don't modify input
    # 𝛒ᴸ = copy(𝛒𝐋)       # ensure we don't modify input
    # 𝐍 = compute_mole_vector_from_density(ρ_mix, V_mix, 𝐳, 𝐌)    
    
    Vᴬ = αA * V_mix #compute_phase_A_vol_from_phasic_densities(𝐍[indx], 𝐌[indx], 𝛒ᴬ[indx], ρB, V_mix)    
    # Vᴸ = V_mix - Vᴳ

    Aᴬ = Vᴬ * A_EOS(T, 1.0, 𝛒ᴬ./ 𝐌; model)
    # Aᴸ = Vᴸ * A_EOS(T, 1.0, 𝛒ᴸ./ 𝐌; model)
    # A = Aᴳ + Aᴸ
    return Aᴬ
end

function p_wrapper2(T, 𝛒𝐀;  model)

    # n = length(𝐳)
    𝐌 = model.Mw
    𝛒ᴬ = copy(𝛒𝐀)       # ensure we don't modify input
    # 𝛒ᴸ = copy(𝛒𝐋)       # ensure we don't modify input
    # 𝐍 = compute_mole_vector_from_density(ρ_mix, V_mix, 𝐳, 𝐌)    
    
    # Vᴬ = αA * V_mix #compute_phase_A_vol_from_phasic_densities(𝐍[indx], 𝐌[indx], 𝛒ᴬ[indx], ρB, V_mix)    
    # Vᴸ = V_mix - Vᴳ

    Pᴬ = P_EOS(T, 1.0, 𝛒ᴬ./ 𝐌; model)
    # Aᴸ = Vᴸ * A_EOS(T, 1.0, 𝛒ᴸ./ 𝐌; model)
    # A = Aᴳ + Aᴸ
    return Pᴬ
end


"""
    compute_M_avg(𝐳, 𝐌)
    - `𝐳` is the overall mole fraction vector
    - `𝐌` is the molar mass vector
Computes the average molar mass of a mixture given its mole fraction vector and molar mass vector.
Returns the average molar mass in kg/mol.
"""

function compute_avg_molecular_mass(𝐳, 𝐌)
    
    return sum(𝐳 .* 𝐌)  # average molar mass in kg/mol
end

function compute_mole_vector_from_density(ρ_mix, V, 𝐳, 𝐌)
    m_total = ρ_mix * V                         # total mass in kg
    M_avg = compute_avg_molecular_mass(𝐳, 𝐌)    # average molar mass in kg/mol
    N_total = m_total / M_avg                   # total moles
    𝐍 = N_total .* 𝐳                            # mole vector
    return 𝐍
end

#------------------------------------------------#


function press(T, V, z; model)
    n = sum(z)
    if is_volume_or_moles_zero(V, z)
        return eps(eltype(V))
    end
    RT = model.R * T
    F = V -> a_res(T, V, z; model)
    ∂F_V = ForwardDiff.derivative(F, V)
    # P = -RT * ∂F_V + n * RT / V
    P = -RT * ∂F_V + n * RT / V
    return P

end

function int_energy(T, V, z; model)
    n = sum(z)
    RT = model.R * T
    A = T -> model.R * T * (a_res(T, V, z; model) + a_ideal(T, V, z; model))
    # A = T -> model.R * T * (a_res(T, V, z; model) + a_ideal_clapeyron(T, V, z; model))

    S = entropy(T, V, z; model)

    return A(T) + T * S

end

function entropy(T, V, z; model)
    n = sum(z)
    RT = model.R * T
    F = T -> model.R * T * (a_res(T, V, z; model) + a_ideal(T, V, z; model))
    ∂F_T = ForwardDiff.derivative(F, T)
    # P = -RT * ∂F_V + n * RT / V
    S = -∂F_T
    return S

end

function chem_pot(T, V, z; model, strange_factor=0.36765328)
    n = sum(z)
    # x = z ./ n
    if isapprox(n, 0.0; atol = eps(eltype(n))) || isapprox(V, 0.0; atol = eps(eltype(V)))
        return zeros(length(z))
    end
    RT = model.R * T
    F = z -> a_res(T, V, z; model) + a_ideal(T, V, z; model)
    ∂F_N = ForwardDiff.gradient(F, z)

    return @. ∂F_N * RT

end

function isothermal_compressibility(T, V, z; model)
    
    ∂p∂V = ForwardDiff.derivative(v -> press(T, v, z; model), V)
    
    return -1.0/V/∂p∂V
end

function is_spinodal(T, V, z; model)

    ∂p∂V = ForwardDiff.derivative(v -> press(T, v, z; model), V)
    # isapprox(∂p∂V, 0.0; atol=1e-8)
end

function mechanical_stability(T, V, z; model)
    return isothermal_compressibility(T, V, z; model) >= 0
end

function diffusive_stability(T, V, z; model)
    A(z) = A_EOS(T,V,z; model)
    Hf = ForwardDiff.hessian(A, z)
    λ = eigmin(Hermitian(Hf)) # calculating just minimum eigenvalue more efficient than calculating all & finding min
    return λ > 0    
end

# eq 13, chapter 2, Michelsen and Mollerup
function lnϕ(T, V, z; model)
    RT = model.R * T
    n = sum(z)

    F = z -> a_res(T, V, z; model)
    ∂F_N = ForwardDiff.gradient(F, z)
    p = press(T, V, z; model)
    # @show p, T
    Z = p * V / RT / n
    Z_abs = abs(Z)
    logϕ = ∂F_N .- sloppylog(Z_abs)
    return logϕ
end

function ln_volume_function(T, V, z; model)
    log_phi = lnϕ(T, V, z; model)
    Z = Compressibility(T, V, z; model)
    log_vol_func = @. -(log_phi + sloppylog(Z))

end



# Function to calculate b
# function b(x, b_vec)
#     return sum(x .* b_vec)
# end

# Function to calculate P^(EOS) using the Peng-Robinson EOS
function P_EOS(T, V, z; model)

    (; R) = model
    N = sum(z)

    if is_volume_or_moles_zero(V, z)
        return eps(eltype(V))
    end

    x = z ./ N

    a_val = a(T, x; model)
    b_val = b(x; model)

    # Calculate pressure using Peng-Robinson equation of state
    P = (N * R * T) / (V - b_val * N) - (a_val * N^2) / (V^2 + 2 * b_val * N * V - b_val^2 * N^2)

    return P
end
# μ_res = ∂U_res/∂N - T∂S_res/∂N
# lnϕ = μ_res/RT .- sloppylog(Z)
# F = Aʳ/RT, where Aʳ = U_res - TS_res, is residual Helmholtz energy
# lnϕ = ∂F∂nᵢ - sloppylog(Z)
# Function to calculate the compressibility factor
function Compressibility(T, V, z; model)
    P = P_EOS(T, V, z; model)
    N = sum(z)
    R = model.R
    Z = P * V / (N * R * T)
end

function compute_Z_PR(P, T, z; model)
    # @show P, T, z
    (;R, T_c, P_c, ω, δ) = model
    
    n = model.Nc

    # b_mix = sum(z .* b_i)
    a_mix = a(T, z; model)
    b_mix = b(z; model)
    A = a_mix * P / (R^2 * T^2)
    B = b_mix * P / (R * T)

    # Cubic coefficients: Z^3 + c2 Z^2 + c1 Z + c0 = 0
    coeffs = [1.0,
              -1.0 + B,
              A - 3B^2 - 2B,
              - (A*B - B^2 - B^3)]

    roots_Z = roots(Polynomial(reverse(coeffs)))
    # @show roots_Z
    real_Z = real.(roots_Z[abs.(imag.(roots_Z)) .< 1e-8])
    return sort(real_Z)  # for single-phase: choose max for vapor, min for liquid
end

function μ_EOS(T, V, z; model)
    if is_volume_or_moles_zero(V, z)
        return z .* 0.0
    end
    U(z) = U_EOS(T, V, z; model)
    S(z) = S_EOS(T, V, z; model)

    # args_list = vcat(T, V, z)
    ∂U_N = ForwardDiff.gradient(U, z)
    ∂S_N = ForwardDiff.gradient(S, z)

    # ∂U_T = ∂U[1]
    # ∂U_V = ∂U[2]
    # ∂U_N = ∂U[1:end]

    # ∂S_T = ∂S[1]
    # ∂S_V = ∂S[2]
    # ∂S_N = ∂S[1:end]
    # if isapprox(sum(z), 0.0; atol = 1e-14)
    #     return zeros(length(z))
    # end
    μ = @. ∂U_N - T * ∂S_N
    return μ
end

function Cv(T, V, z; model)
    U(T) = U_EOS(T, V, z; model)
    # S(T) = S_EOS(T, V, z; model)
    # args_list = vcat(T, V, z)
    ∂U_T = ForwardDiff.derivative(U, T)
    # ∂S_T = ForwardDiff.derivative(S, T)
    # ∂U_T = ∂U[1]
    # ∂U_V = ∂U[2]
    # ∂U_N = ∂U[1:end]

    # ∂S_T = ∂S[1]
    # ∂S_V = ∂S[2]
    # ∂S_N = ∂S[1:end]

    
    return ∂U_T #, ∂S_T * T
end

"""
    For speed of sound calculations, we need Cv per unit mass
"""
function Cv_mass(T, V, z; model)
    total_mass = sum(z .* model.Mw)  # Calculate the total mass of the mixture
    Cv(T, V, z; model) / total_mass
end

function Cp(T, V, z; model)
    # U(TV) = U_EOS(TV[1], TV[2], z; model)
    S(TV) = S_EOS(TV[1], TV[2], z; model)
    P(TV) = P_EOS(TV[1], TV[2], z; model)
    TV = [T, V]

    ∂S_T, ∂S_V = ForwardDiff.gradient(S, TV)
    ∂P_T, ∂P_V = ForwardDiff.gradient(P, TV)
    ∂S_T_constant_P = (∂S_T - ∂S_V * ∂P_T / ∂P_V)
    return T * (∂S_T_constant_P)
    
end

"""
    For speed of sound calculations, we need Cv per unit mass
"""

function Cp_mass(T, V, z; model)
    total_mass = sum(z .* model.Mw)  # Calculate the total mass of the mixture
    Cp(T, V, z; model) / total_mass
end

function U_res(T, V, z; model)
    N = sum(z)
    if isapprox(N, 0.0; atol = 1e-14)
        return 0.0
    end

    (; R) = model
    
    x = @. z / N
    sqrt2 = sqrt(2)  # Precompute sqrt(2) for efficiency
    a_val = a(T, x; model)
    b_val = b(x; model)

    # Exact derivative
    ∂T_a = ForwardDiff.derivative(T -> a(T, x; model), T)
    log_arg(Δ) = (V + Δ * b_val * N)
    term1 = N * (T * ∂T_a - a_val) / ((model.Δ₁ - model.Δ₂) * b_val)
    term2 = sloppylog(abs((log_arg(model.Δ₁)) / log_arg(model.Δ₂)))
    u_res = term1 * term2

    return u_res
end

# Function to calculate S^(EOS) with named arguments
function S_res(T, V, z; model)
    (; R) = model
    N = sum(z)
     N = sum(z)
    
    if isapprox(N, 0.0; atol = 1e-14)
        return 0.0
    end

    x = @. z / N

    b_val = b(x; model)
    ∂T_a = ForwardDiff.derivative(T -> a(T, x; model), T)

    term1 = N * R * sloppylog(abs((V - b_val * N) / V))
    term2 = N * ∂T_a / ((model.Δ₁ - model.Δ₂) * b_val) * sloppylog(abs((V + model.Δ₁ * b_val * N) / (V + model.Δ₂ * b_val * N)))

    s_res = term1 + term2
    return s_res
end

function is_volume_or_moles_zero(V, z)
    # Check if the sum of moles is approximately zero
    # @show z, V
    length(z) == 0 && @show z, V
    if isapprox(sum(z), 0.0; atol = 1e-14) || isapprox(V, 0.0; atol = 1e-14)
        return true
    end
    return false
end

# Function to calculate U^(EOS) with named arguments
function U_EOS(T, V, z; model)
    if is_volume_or_moles_zero(V, z)
        return 0.0
    end
    return U_res(T, V, z; model) + U_ideal(T, V, z; model)
end

function H_EOS(T, V, z; model)
    return U_EOS(T, V, z; model) + P_EOS(T, V, z; model) * V
end

function G_EOS(T, V, z; model)
    return A_EOS(T, V, z; model) + T * S_EOS(T, V, z; model)
end

# Function to calculate S^(EOS) with named arguments
function S_EOS(T, V, z; model)
    if is_volume_or_moles_zero(V, z)
        return 0.0
    end
    s_res = S_res(T, V, z; model)
    s_ideal = S_ideal(T, V, z; model)
    # @show s_res, s_ideal
    return s_res + s_ideal
end

@exportAll()
end
