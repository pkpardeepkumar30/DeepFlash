module EosGPU
using CUDA
using KernelAbstractions
using ForwardDiff
using Zygote
using Enzyme
using Flux
using StaticArrays
using ..AutoDiffGPU
# using ForwardDiff
using Polynomials
using LinearAlgebra
using Statistics
# using NLsolve

function PengRobinson_gpu(; Mw::SVector, ρ_c::SVector, T_c::SVector, P_c::SVector, ω::SVector, δ::SMatrix, α::SMatrix, T_c_mix=nothing, P_c_mix=nothing)
    R = 8.31446261815324
    T0 = 298.15 # Reference temperature in K
    P0 = 1e5 # Reference pressure in Pa    
    u0 = -2478.95687512 # J/mol
    Ωa = 0.45724
    Ωb = 0.0778
    Δ1 = 1 + sqrt(2)
    Δ2 = 1 - sqrt(2)
    # @show ω
    ω_coeffs1 = SA[0.37464, 1.54226, -0.26992]
    ω_coeffs2 = SA[0.379642, 1.48503, -0.164423, 0.016667]

    Nc = length(Mw)

    #  m_i = ω[i] < 0.5 ?
    #       dot(ω_coeffs1, [1.0, ω, ω^2]) :
    #       dot(ω_coeffs2, [1.0, ω, ω^2, ω^3])

    (; R, Nc, Mw, ρ_c, T_c, P_c, T_c_mix, P_c_mix, Ωa, Ωb, Δ1, Δ2, ω, ω_coeffs1, ω_coeffs2, δ, α, T0, P0, u0)
end

# GPU-compatible SRK function
function SRK_gpu(; Mw, ρ_c, T_c, P_c, ω, δ::AbstractMatrix{Float64}, α::AbstractMatrix{Float64}, 
                doScale=false)

    R = 8.31446261815324
    T0 = 298.15
    P0 = 1e5
    u0 = -2478.95687512
    Ωa = 0.42748
    Ωb = 0.08664
    Δ1 = 1
    Δ2 = 0
    ω_coeffs1 = SA[0.48508, 1.55171, -0.15613]
    ω_coeffs2 = SA[0.48508, 1.55171, -0.15613, 0.0]
    Nc = length(Mw)

    # Convert arrays to GPU arrays
    Mw_gpu = Mw isa CuArray ? Mw : CuArray(Mw)
    ρ_c_gpu = ρ_c isa CuArray ? ρ_c : CuArray(ρ_c)
    T_c_gpu = T_c isa CuArray ? T_c : CuArray(T_c)
    P_c_gpu = P_c isa CuArray ? P_c : CuArray(P_c)
    ω_gpu = ω isa CuArray ? ω : CuArray(ω)
    δ_gpu = δ isa CuArray ? δ : CuArray(δ)
    α_gpu = α isa CuArray ? α : CuArray(α)

    (; R, Nc, Mw=Mw_gpu, ρ_c=ρ_c_gpu, T_c=T_c_gpu, P_c=P_c_gpu, Ωa, Ωb, Δ1, Δ2, 
        ω=ω_gpu, ω_coeffs1, ω_coeffs2, δ=δ_gpu, α=α_gpu, T0, P0, u0, components, doScale)
end

# GPU-compatible sloppysqrt
@inline function sloppysqrt_gpu(x)
    sqrt(abs(x))
end

# GPU-compatible a_i function
@inline function a_i_gpu(T, i, model_gpu)
    T_c = model_gpu.T_c[i]
    P_c = model_gpu.P_c[i]
    ω = model_gpu.ω[i]
    R = model_gpu.R
    ω_coeffs1 = model_gpu.ω_coeffs1
    ω_coeffs2 = model_gpu.ω_coeffs2

    # # Calculate m_i
    # m_i = if ω < 0.5
    #     dot(ω_coeffs1, SA[1.0, ω, ω^2])
    # else
    #     dot(ω_coeffs2, SA[1.0, ω, ω^2, ω^3])
    # end
    m_i = 0.0f0
    if ω < 0.5f0
        c = model_gpu.ω_coeffs1  # Array-like with 3 elements
        # m = c1 + c2*ω + c3*ω^2
        m_i = ((c[3]*ω) + c[2])*ω + c[1]
    else
        c = model_gpu.ω_coeffs2  # Array-like with 4 elements
        m_i = (((c[4]*ω) + c[3])*ω + c[2])*ω + c[1]
    end

    T_r = T / T_c
    Ωa = model_gpu.Ωa
    multiplier = Ωa * (R^2 * T_c^2) / P_c
    a_i_val = multiplier * (1f0 + m_i * (1 - sloppysqrt_gpu(T_r)))^2
    return a_i_val
end

# GPU-compatible da_i_dT function
@inline function da_i_dT_gpu(T, i, model_gpu)
    T_c = model_gpu.T_c[i]
    P_c = model_gpu.P_c[i]
    ω = model_gpu.ω[i]
    R = model_gpu.R
    ω_coeffs1 = model_gpu.ω_coeffs1
    ω_coeffs2 = model_gpu.ω_coeffs2

    # Calculate m_i
    # m_i = if ω < 0.5
    #     dot(ω_coeffs1, SA[1.0, ω, ω^2])
    # else
    #     dot(ω_coeffs2, SA[1.0, ω, ω^2, ω^3])
    # end

    m_i = 0.0f0
    if ω < 0.5f0
        c = model_gpu.ω_coeffs1  
        # m = c1 + c2*ω + c3*ω^2
        m_i = ((c[3]*ω) + c[2])*ω + c[1]
    else
        c = model_gpu.ω_coeffs2  
        m_i = (((c[4]*ω) + c[3])*ω + c[2])*ω + c[1]
    end

    T_r = T / T_c
    Ωa = model_gpu.Ωa
    multiplier = Ωa * (R^2 * T_c^2) / P_c
    
    # Derivative calculation
    da_i_dT_val = -2 * multiplier * (1 + m_i * (1 - sloppysqrt_gpu(T_r))) * m_i / sloppysqrt_gpu(T_c) / 2 / sloppysqrt_gpu(T)
    return da_i_dT_val
end

# GPU-compatible b_i function
@inline function b_i_gpu(i, model_gpu)
    R = model_gpu.R
    T_c = model_gpu.T_c[i]
    P_c = model_gpu.P_c[i]
    
    b_val = model_gpu.Ωb * (R * T_c) / P_c
    return b_val
end

# GPU-compatible helper functions (no allocations)
@inline function b_ij_gpu(b_i, b_j)
    return 0.5 * (b_i + b_j)
end

@inline function a_ij_gpu(a_i, a_j, δ_ij)
    return (1.0 - δ_ij) * sqrt(a_i * a_j)
end

# GPU-compatible a(T, x) calculation (no allocations)
@inline function a_mixture_gpu(T, z, model)
    Nc = Int(model.Nc)
    δ = model.δ
    result = 0.0
    n_total = sum(z)
    inv_n_total = 1.0 / n_total
    # z[k] * inv_n_total
    for i in 1:Nc
        a_i_val = a_i_gpu(T, i, model)
        x_i = z[i] * inv_n_total
        
        # Diagonal term (j == i)
        a_ii = a_ij_gpu(a_i_val, a_i_val, 0.0)
        result += x_i * x_i * a_ii
        
        # Off-diagonal terms (j > i)
        for j in (i+1):Nc
            a_j_val = a_i_gpu(T, j, model)
            a_ij_val = a_ij_gpu(a_i_val, a_j_val, δ[i, j])
            result += 2.0 * x_i * (z[j] * inv_n_total) * a_ij_val
        end
    end
    
    return result
end

# GPU-compatible b(x) calculation (no allocations)
@inline function b_mixture_linear_gpu(x, model)
    Nc = Int(model.Nc)
    result = 0.0
    
    for i in 1:Nc
        result += x[i] * b_i_gpu(i, model)
    end
    
    return result
end

# GPU-compatible b(x) quadratic mixing rule (no allocations)
@inline function b_mixture_quadratic_gpu(z, model)
    Nc = Int(model.Nc)
    result = 0.0
    n_total = sum(z)
    inv_n_total = 1.0 / n_total

    for i in 1:Nc
        b_i_val = b_i_gpu(i, model)
        x_i = z[i] * inv_n_total
        
        # Diagonal term (j == i)
        b_ii = b_ij_gpu(b_i_val, b_i_val)
        result += x_i * x_i * b_ii
        
        # Off-diagonal terms (j > i)
        for j in (i+1):Nc
            b_j_val = b_i_gpu(j, model)
            b_ij_val = b_ij_gpu(b_i_val, b_j_val)
            result += 2.0 * x_i * (z[j] * inv_n_total) * b_ij_val
        end
    end
    
    return result
end

# GPU-compatible μ_res_michelsen for Peng Robinson (no allocations, single thread per component)
@inline function μ_res_michelsen_gpu(i::Int, T, V, z, model)
    R = model.R
    Nc = Int(model.Nc)
    
    # Compute total moles
    n_total = 0.0
    for k in 1:Nc
        n_total += z[k]
    end
    
    # Handle edge cases
    if n_total < 1e-12 || V < 1e-12
        return 0.0
    end
    
    # Compute mole fractions
    
    # Compute mixture properties
    b_mix = b_mixture_quadratic_gpu(z, model)
    a_mix = a_mixture_gpu(T, z, model)
    
    B = n_total * b_mix
    D = n_total * n_total * a_mix
    
    Δ1, Δ2 = model.Δ1, model.Δ2
    δ_diff = Δ1 - Δ2
    
    # Compute f and g
    V_Δ1B = V + Δ1 * B
    V_Δ2B = V + Δ2 * B
    V_B = V - B
    
    f = 0.0
    if abs(B * δ_diff) > 1e-12 && abs(V_Δ2B) > 1e-12
        f = 1.0 / (R * B * δ_diff) * log(V_Δ1B / V_Δ2B)
    end
    
    g = 0.0
    if abs(V) > 1e-12 && abs(V_B) > 1e-12
        g = log(V_B / V)
    end
    
    g_B = -1.0 / V_B
    
    # Compute B_i
    b_i_val = b_i_gpu(i, model)
    sum_zj_bij = 0.0
    for j in 1:Nc
        b_j_val = b_i_gpu(j, model)
        b_ij_val = b_ij_gpu(b_i_val, b_j_val)
        sum_zj_bij += z[j] * b_ij_val
    end
    B_i = (2.0 * sum_zj_bij - B) / n_total
    
    # Compute D_i
    a_i_val = a_i_gpu(T, i, model)
    sum_zj_aij = 0.0
    for j in 1:Nc
        a_j_val = a_i_gpu(T, j, model)
        a_ij_val = a_ij_gpu(a_i_val, a_j_val, model.δ[i, j])
        sum_zj_aij += z[j] * a_ij_val
    end
    D_i = 2.0 * sum_zj_aij
    
    # Compute derivatives of f
    f_V = 0.0
    f_B = 0.0
    if abs(R * B * δ_diff) > 1e-12
        f_V = 1.0 / (R * B * δ_diff) * (1.0/V_Δ1B - 1.0/V_Δ2B)
        f_B = -(f + V * f_V) / B
    end
    
    # Final derivative components
    F_n = -g
    F_B = -n_total * g_B - (D / T) * f_B
    F_D = -f / T
    
    ∂F_∂zᵢ = F_n + F_B * B_i + F_D * D_i
    return R * T * ∂F_∂zᵢ
end

@inline function a_res_device(T, V, z, model_gpu, Nc::Int)
    # Compute mole total n
    n = zero(T)
    for j in 1:Nc
        n += z[j]
    end

    if n < 1e-12 || V < 1e-12
        return zero(T)
    end

    # compute b_m inline (no temporary array)
    b_m = zero(T)
    for j in 1:Nc
        b_m += z[j] * b_i_gpu(j, model_gpu) / n
    end

    # compute a_m with double loop, compute a_i on the fly (no allocations)
    a_m = zero(T)
    for i in 1:Nc
        ai = a_i_gpu(T, i, model_gpu)
        for j in 1:Nc
            aj = a_i_gpu(T, j, model_gpu)
            δ_ij = model_gpu.δ[i, j]
            aij = (1f0 - δ_ij) * sqrt(max(ai * aj, 0f0))
            a_m += z[i] * z[j] * aij / (n * n)
        end
    end

    RTval = model_gpu.R * T
    δ1 = model_gpu.Δ1
    δ2 = model_gpu.Δ2

    term1 = -n * log(max(1f0 - n * b_m / V, 1e-12))
    term2 = a_m * n / (RTval * b_m * (δ1 - δ2))
    term3 = log(max((1f0 + δ1 * n * b_m / V) / (1f0 + δ2 * n * b_m / V), 1e-12))

    return term1 - term2 * term3
end

@inline function a_res_device_indexed(T, V, z_batch, idx::Int, model_gpu, Nc::Int)
    # same logic as a_res_device, but where z[j] is z_batch[idx, j]
    n = zero(T)
    for j in 1:Nc
        n += z_batch[idx, j]
    end
    if n < 1e-12 || V < 1e-12
        return zero(T)
    end
    b_m = zero(T)
    for j in 1:Nc
        b_m += z_batch[idx, j] * b_i_gpu(j, model_gpu) / n
    end
    a_m = zero(T)
    for i in 1:Nc
        ai = a_i_gpu(T, i, model_gpu)
        for j in 1:Nc
            aj = a_i_gpu(T, j, model_gpu)
            δ_ij = model_gpu.δ[i, j]
            aij = (1f0 - δ_ij) * sqrt(max(ai * aj, 0f0))
            a_m += z_batch[idx, i] * z_batch[idx, j] * aij / (n * n)
        end
    end
    RTval = model_gpu.R * T
    δ1 = model_gpu.Δ1
    δ2 = model_gpu.Δ2
    term1 = -n * log(max(1f0 - n * b_m / V, 1e-12))
    term2 = a_m * n / (RTval * b_m * (δ1 - δ2))
    term3 = log(max((1f0 + δ1 * n * b_m / V) / (1f0 + δ2 * n * b_m / V), 1e-12))
    return term1 - term2 * term3
end


# GPU-compatible sloppylog
function sloppylog_gpu(x)
    return log.(max.(x, 1e-100))  # Avoid log(0)
end

# Inline versions for single-kernel approach (no separate kernel launches)
function U_ideal_gpu(T, V, z, model_gpu)
    α = model_gpu.α
    R = model_gpu.R
    T0 = model_gpu.T0
    Nc = model_gpu.Nc
    u0 = model_gpu.u0
    
    N = 0.0
    @inbounds for i in 1:Int(Nc)
        N += z[i]
    end

    # short-circuit on tiny N or V
    if N < 1e-100 || V < 1e-100
        return 0.0
    end

    # term1
    term1 = -N * R * (T - T0)

    # precompute powers T^1..T^4 and T0^1..T0^4
    T1 = T
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T

    T0_1 = T0
    T0_2 = T0 * T0
    T0_3 = T0_2 * T0
    T0_4 = T0_3 * T0

    term2 = 0.0

    @inbounds for i in 1:Int(Nc)
        zi = z[i]
        if zi > 0.0
            # α row: α[i,1..4] (assumes α has 4 columns)
            # component_sum = Σ_{k=0..3} α[i,k+1] * (T^(k+1) - T0^(k+1)) / (k+1)
            comp = 0.0
            comp += α[i, 1] * (T1  - T0_1)          # k=0, divide by 1
            comp += α[i, 2] * (T2  - T0_2) / 2.0    # k=1
            comp += α[i, 3] * (T3  - T0_3) / 3.0    # k=2
            comp += α[i, 4] * (T4  - T0_4) / 4.0    # k=3

            term2 += zi * comp
        end
    end

    return term1 + term2 + N * u0
end


function S_ideal_gpu(T, V, z, model_gpu)
    α = model_gpu.α
    R = model_gpu.R
    T0 = model_gpu.T0
    P0 = model_gpu.P0
    Nc = model_gpu.Nc
    
    term1 = 0.0f0
    term2 = 0.0f0

    for i in 1:Nc
        zi = z[i]

        if zi > 0f0
            # --- term 1 ---
            log_arg = V * P0 / (zi * R * T)
            log_arg = max(log_arg, 1f-30)
            term1 += R * zi * log(log_arg)

            # --- term 2 ---
            component_sum = 0.0f0

            # α[i,1] * log(T/T0)
            ratio = T / T0
            ratio = max(ratio, 1f-30)
            component_sum += α[i,1] * log(ratio)

            # α[i,2]*(T - T0) / 1  +  α[i,3]*(T^2 - T0^2)/2  + ...
            T1 = T
            T2 = T * T
            T3 = T2 * T

            T01 = T0
            T02 = T0 * T0
            T03 = T02 * T0

            component_sum += α[i,2] * (T1 - T01) / 1
            component_sum += α[i,3] * (T2 - T02) / 2
            component_sum += α[i,4] * (T3 - T03) / 3

            term2 += zi * component_sum
        end
    end

    return term1 + term2

end

# Pure GPU a_ideal function
function a_ideal_gpu(T, V, z_gpu, model_gpu)
    RT = model_gpu.R * T
    α = model_gpu.α
    val = α[1, 2]
    # out[1] = val
    u_ideal = U_ideal_gpu(T, V, z_gpu, model_gpu)
    
    s_ideal = S_ideal_gpu(T, V, z_gpu, model_gpu)
    
    return (u_ideal - T * s_ideal) / RT
    # return nothing
end

# Partial derivative of U_ideal w.r.t z[i]
function dU_dzi(T, V, z, model_gpu, i)
    α = model_gpu.α
    R = model_gpu.R
    T0 = model_gpu.T0
    u0 = model_gpu.u0

    # Sum over α-terms for component i
    comp = α[i,1]*(T - T0) +
           α[i,2]*(T^2 - T0^2)/2 +
           α[i,3]*(T^3 - T0^3)/3 +
           α[i,4]*(T^4 - T0^4)/4

    # Derivative formula
    return -R*(T - T0) + comp + u0
end

# Partial derivative of S_ideal w.r.t z[i]
function dS_dzi(T, V, z, model_gpu, i)
    α = model_gpu.α
    R = model_gpu.R
    T0 = model_gpu.T0
    P0 = model_gpu.P0

    # Avoid log(0)
    zi_safe = max(z[i], 1e-30)

    # Translational term
    term1 = R * (log(V * P0 / (R * T * zi_safe)) - 1)

    # Temperature α-term
    term2 = α[i,1] * log(T / T0) +
            α[i,2] * (T - T0) +
            α[i,3] * (T^2 - T0^2)/2 +
            α[i,4] * (T^3 - T0^3)/3

    return term1 + term2
end

using LinearAlgebra

# Vectorized derivative of U_ideal w.r.t z
function dU_dz_gpu(T, z, model_gpu)
    α = model_gpu.α
    R = model_gpu.R
    T0 = model_gpu.T0
    u0 = model_gpu.u0
    Nc = model_gpu.Nc

    # Precompute powers of T and T0
    T1, T2, T3, T4 = T, T^2, T^3, T^4
    T0_1, T0_2, T0_3, T0_4 = T0, T0^2, T0^3, T0^4

    # α-terms for all components
    comp = α[:,1]*(T1 - T0_1) .+ α[:,2]*(T2 - T0_2)/2 .+
           α[:,3]*(T3 - T0_3)/3 .+ α[:,4]*(T4 - T0_4)/4

    # Derivative: -R*(T-T0) + comp + u0
    return -R*(T - T0) .+ comp .+ u0
end

# Vectorized derivative of S_ideal w.r.t z
function dS_dz_gpu(T, V, z, model_gpu)
    α = model_gpu.α
    R = model_gpu.R
    T0 = model_gpu.T0
    P0 = model_gpu.P0
    Nc = model_gpu.Nc

    # Safe z to avoid log(0)
    z_safe = max.(z, 1e-30)

    # Translational term: R*(log(V*P0 / (R*T*z_i)) - 1)
    term1 = R .* (log.(V * P0 ./ (R * T .* z_safe)) .- 1)

    # α-temperature term
    term2 = α[:,1] .* log(T/T0) .+
            α[:,2] .* (T - T0) .+
            α[:,3] .* (T^2 - T0^2)/2 .+
            α[:,4] .* (T^3 - T0^3)/3

    return term1 .+ term2
end

function μ_ideal_i(i, T, V, z, model_gpu)
    RT = model_gpu.R * T

    dU = dU_dzi(T, V, z, model_gpu, i)
    dS = dS_dzi(T, V, z, model_gpu, i)

    return (dU - T * dS) / RT
end

function μ_tot(T, V, z, model_gpu, mu_out)  
    Nc = model_gpu.Nc
    for j in 1:Nc           
        μ_res = μ_res_michelsen_gpu(j, T, V, z, model_gpu)
        μ_ideal = μ_ideal_i(j, T, V, z, model_gpu)
        RT = model_gpu.R * T
        μ_res + RT * μ_ideal
        out[j] = μ_res + RT * μ_ideal
    end
    return nothing
end

function μ_tot_batch_kernel(z_batch, T_batch, V_batch, model_gpu, mu_out)
    tid = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    M  = size(z_batch, 1)        # number of samples
    Nc = model_gpu.Nc

    if tid > M
        return
    end

    # Load batch-specific data
    T = T_batch[tid]
    V = V_batch[tid]

    # Thread-local views into 2D arrays
    z_i  = @view z_batch[tid, :]
    # xi   = @view x_local[:, tid]
    mui  = @view mu_out[:, tid]

    RT = model_gpu.R * T

    # -------------------------------
    # Compute total moles
    # -------------------------------
    n_total = sum(z_i)

    if n_total < 1e-12 || V < 1e-12
        @inbounds for j in 1:Nc
            mui[j] = 0.0
        end
        return
    end

        
    # -------------------------------
    # Compute chemical potentials
    # -------------------------------
    @inbounds for j in 1:Nc
        μ_res   = μ_res_michelsen_gpu(j, T, V, z_i, model_gpu)
        μ_ideal = μ_ideal_i(j, T, V, z_i, model_gpu)
        mui[j]  = μ_res + RT * μ_ideal
    end

    return
end

function launch_mu_batch(T_single, V_single, z_single;
                         model, M::Int)

    Nc = model.Nc

    # Build batched inputs
    # z_batch = CuArray(reshape(repeat(z_single, M), M, Nc))
    z_batch = CUDA.zeros(Float32, M, Nc)
    for j in 1:Nc
        z_batch[:, j] .= z_single[j]
    end
    # z_batch = CuArray(fill.(z_single', M))
    T_batch = CuArray(fill(Float32(T_single), M))
    V_batch = CuArray(fill(Float32(V_single), M))
    # @show z_batch
    # Outputs
    # x_local = CUDA.zeros(Float32, Nc, M)
    mu_out  = CUDA.zeros(Float32, Nc, M)

    threads = 128
    blocks  = cld(M, threads)
    println("Launching μ_tot_batch_kernel with $blocks blocks of $threads threads")
    @cuda threads=threads blocks=blocks μ_tot_batch_kernel(
        z_batch, T_batch, V_batch, model,
        mu_out
    )

    return mu_out
end


function μ_ideal(T, V, z, model_gpu)
    RT = model_gpu.R * T

    dU_vec = [dU_dzi(T, V, z, model_gpu, i) for i in 1:model_gpu.Nc]
    dS_vec = [dS_dzi(T, V, z, model_gpu, i) for i in 1:model_gpu.Nc]

    return (dU_vec .- T .* dS_vec) ./ RT
end


function forward_ad_on_gpu(f, x)
    N = length(x)
    y = CUDA.zeros(eltype(x), N)
    ∂y = CUDA.zeros(eltype(x), N)
    # @show eltype(x)
    
    threads = 128
    blocks = cld(N, threads)
    @cuda threads=1 blocks=1 AutoDiffGPU.gradient(f, x, y, ∂y)

    return y, ∂y
end


# ----- Test run -----
function test_manual_ad()
    x = cu(Float32[1.0])
    y, dy = forward_ad_on_gpu(x -> exp(x), x)
    println("x = ", Array(x))
    println("y = ", Array(y))   # expected [1+2=3, 8+4=12, 27+6=33]
    println("dy = ", Array(dy)) # derivative g'(x)=3x^2 + 2 => [5, 14, 29]
end

end
