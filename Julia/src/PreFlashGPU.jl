module PreFlashGPU
using CUDA
using Dates
using ..Scalers
using ..EOS
using ..Flash
using ..StabilityGPU
using Random
using StaticArrays
using ForwardDiff
using LinearAlgebra

# GPU version of extract_phase_properties
function extract_phase_properties_gpu(res_gpu, V_spec, N_spec_gpu, model, ρ_mix)
    # Convert to CPU for computation
    res_cpu = Array(res_gpu)
    N_spec_cpu = Array(N_spec_gpu)
    
    nc = model.Nc
    T = res_cpu[end]
    α = res_cpu[nc + 1]
    V_G = α * V_spec
    ρG = res_cpu[1:nc] .* ρ_mix
    N_G = ρG .* V_G ./ model.Mw
    N_L = N_spec_cpu .- N_G
    V_L = V_spec - V_G
    
    # Return GPU arrays where appropriate
    return T, α, V_G, cu(N_G), V_L, cu(N_L)
end

# GPU version of is_physically_valid_phase
function is_physically_valid_phase_gpu(T, V, N_gpu; N_spec_gpu, V_spec, T_upper_limit=410, T_lower_limit=100)
    # Convert to CPU for checks
    N_cpu = Array(N_gpu)
    N_spec_cpu = Array(N_spec_gpu)
    
    all(N_cpu .≥ 0) && all(N_cpu .≤ N_spec_cpu) && (1e-10 ≤ V ≤ V_spec - 1e-10) && (T_lower_limit < T < T_upper_limit)
end

# GPU version of is_physically_valid_single_phase (unchanged as it's scalar)
function is_physically_valid_single_phase_gpu(T; T_upper_limit=410, T_lower_limit=100)
    (T_lower_limit < T < T_upper_limit)
end

# GPU version of is_valid_two_phase
function is_valid_two_phase_gpu(T, α, V_G, N_G_gpu, V_L, N_L_gpu, N_spec_gpu, V_spec, model, S_one)
    # Physical constraints
    valid_phase = is_physically_valid_phase_gpu(T, V_G, N_G_gpu; N_spec_gpu, V_spec)
    
    # Entropy validation with CPU EOS calls
    valid_entropy = if valid_phase
        N_G_cpu = Array(N_G_gpu)
        N_L_cpu = Array(N_L_gpu)
        
        S_two = EOS.S_EOS(T, V_G, N_G_cpu; model) + EOS.S_EOS(T, V_L, N_L_cpu; model)
        S_two ≥ S_one
    else
        false
    end
    
    return valid_phase && valid_entropy
end

# GPU version of flash_trivial_solution (unchanged as it's scalar)
function flash_trivial_solution_gpu(α; tol=1e-8)
    α < tol || α > (1 - tol) 
end

function attempt_two_phase_flash_gpu(initial_guess_gpu, U_spec, V_spec, N_spec_gpu, model,
                                    Scale, T_stab, S_one, numPhases)
    # Convert GPU arrays to CPU for flash computation
    initial_guess_cpu = Array(initial_guess_gpu)
    N_spec_cpu = Array(N_spec_gpu)
    Scale_cpu = Array(Scale)
    ρ_mix = sum(N_spec_cpu .* model.Mw) / V_spec
    cons(_) = nothing  # Dummy constraint function

    x_spec_cpu = vcat(U_spec, V_spec, N_spec_cpu)
    
    # Flash optimization happens on CPU
    func_to_optimize(x) = Flash.rho_α_Q(x; U_spec, V_spec, N_spec=N_spec_cpu, model, numPhases, Scale=Scale_cpu)
    g(x) = ForwardDiff.gradient(func_to_optimize, x)
    H(x) = ForwardDiff.hessian(func_to_optimize, x)
    
    flash_converged, res_cpu, iterations = Flash.OptimizeHelmholtz(func_to_optimize,
        g, H,
        initial_guess_cpu, 
        cons; 
        tol=1e-6, 
        maxiter=1000,
        V_total=V_spec, 
        N_total=N_spec_cpu
    )
   
    if !flash_converged 
        return false, cu(vcat(N_spec_cpu, V_spec, T_stab))  # Return GPU array
    end
    
    # Extract properties (returns some GPU arrays)
    T, α, V_G, N_G_gpu, V_L, N_L_gpu = extract_phase_properties_gpu(
        cu(res_cpu), V_spec, N_spec_gpu, model, ρ_mix
    )
    
    # EOS calls on CPU with converted arrays
    N_G_cpu = Array(N_G_gpu)
    N_L_cpu = Array(N_L_gpu)
    U_G = EOS.U_EOS(T, V_G, N_G_cpu; model)
    U_L = EOS.U_EOS(T, V_L, N_L_cpu; model)
    
    is_trivial = flash_trivial_solution_gpu(α) 
    is_valid = is_valid_two_phase_gpu(T, α, V_G, N_G_gpu, V_L, N_L_gpu, N_spec_gpu, V_spec, model, S_one)

    if is_valid && !is_trivial
        return true, cu(vcat(N_G_cpu, V_G, T))  # Return GPU array
    end
    return false, cu(vcat(N_G_cpu, V_G, T))  # Return GPU array
end

function stability_analysis_fallback_gpu(U_spec, V_spec, N_spec_gpu, model, T_stab, S_one, 
                                        Scale, numPhases)
    
    # Convert GPU array to CPU for stability analysis
    N_spec_cpu = Array(N_spec_gpu)
    
    # CPU stability analysis call
    stab = StabilityGPU.VT_stabilityAnalysis_gpu(; model, T_spec=T_stab, V_spec, z_spec_gpu=N_spec_gpu)
    
    is_uncertain = abs(stab.D_trial) ≤ 1e-4  
    
    if !stab.isunstable
        is_good_single_phase = is_physically_valid_single_phase_gpu(T_stab; T_upper_limit=600)
        if is_good_single_phase
            # Return GPU array for single-phase solution
            return true, CuArray(vcat(N_spec_cpu, V_spec, T_stab))
        else
            @warn "Single-phase solution is not valid, trying stability-based initialization."
            return false, CuArray(vcat(N_spec_cpu, V_spec, T_stab))
        end
    end
    
    # CPU call for IG3
    x0 = Stability.IG3(stab.c_sol, T_stab, U_spec, V_spec, N_spec_cpu; model, verbose=false)    
    
    ρ_mix = sum(N_spec_cpu .* model.Mw) / V_spec
    
    # Extract and validate stability-based initial guess
    N_G = x0[1:model.Nc]
    V_G = x0[model.Nc+1]
    
    if !(all(0 .≤ N_G .≤ N_spec_cpu)) || !(0 < V_G < V_spec)
        return false, CuArray(vcat(N_spec_cpu, V_spec, T_stab))
    end
    
    # Compute initial vapor fraction
    N_L = N_spec_cpu .- N_G
    V_L = V_spec - V_G
    ρG = N_G .* model.Mw ./ V_G
    ρL = N_L .* model.Mw ./ V_L

    ϵ = 1e-12  # Small value to avoid division by zero
    denominator = abs(ρL[1] - ρG[1]) < ϵ ? ϵ : ρL[1] - ρG[1]

    # Total mixture density of the first component
    ρ_1 = N_spec_cpu[1] * model.Mw[1] / V_spec
    
    α_est = (ρL[1] - ρ_1) / denominator
    α_est = clamp(α_est, 1e-8, 1-1e-8)  # Avoid extreme values
    
    x_init = vcat(ρG ./ ρ_mix, α_est, T_stab)
    
    if sum(ρG) < 1e-8
        @warn "Vapor density is too low."
    end
    if sum(ρG) > 1000
        @warn "Vapor density is too high."        
    end
    
    # Convert to GPU and attempt flash
    x_init_gpu = CuArray(x_init)
    
    # Attempt flash with stability-based guess (GPU version)
    success, result = attempt_two_phase_flash_gpu(
        x_init_gpu, U_spec, V_spec, N_spec_gpu, model, Scale, 
        T_stab, S_one, numPhases
    )
    
    success && return true, result
    
    return false, CuArray(vcat(N_spec_cpu, V_spec, T_stab))
end

function compute_single_phase_state_gpu(U_spec, V_spec, N_spec_gpu, model; T_guess=300.0)
    # Convert GPU array to CPU for EOS calls
    N_spec_cpu = Array(N_spec_gpu)
    
    # First attempt with given T_guess (if provided)
    if T_guess !== nothing
        T_single = EOS.GetTemperatureForSpecifiedUV(; 
            U=U_spec, V=V_spec, z=N_spec_cpu, model, T_guess, verbose=false
        )
        if is_physically_valid_single_phase_gpu(T_single; T_upper_limit=4000, T_lower_limit=50)
            return T_single
        end
    end

    # Try incrementally from 300 K to 2000 K in steps of 100 K
    for T_init in 300:100:2000
        T_single = EOS.GetTemperatureForSpecifiedUV(; 
            U=U_spec, V=V_spec, z=N_spec_cpu, model, T_guess=T_init, verbose=false
        )
        if is_physically_valid_single_phase_gpu(T_single; T_upper_limit=4000, T_lower_limit=50)
            @info "Single-phase state found at T = $T_single K using initial guess $T_init K"
            return T_single
        end
    end

    @error "Single-phase state computation yielded unphysical temperature for all trial guesses (300–2000 K)."
    return NaN  # Indicate failure to find a valid temperature
end

function solve_rho_Q_from_UVN_gpu(U_spec, V_spec, N_spec_gpu; model, singlePhaseSure=false, 
                                  ϵ=0.0, x_guess=nothing, numPhases=2)

    # Convert GPU array to CPU for single phase computation
    N_spec_cpu = Array(N_spec_gpu)
    
    T_stab = compute_single_phase_state_gpu(U_spec, V_spec, N_spec_gpu, model)
    
    if singlePhaseSure
        # If single-phase solution is guaranteed, return it directly
        status = is_physically_valid_single_phase_gpu(T_stab)        
        result = (status = status, flash_result = CuArray(vcat(N_spec_cpu, V_spec, T_stab)))
        return result
    end
    
    # CPU EOS call for entropy
    S_one = EOS.S_EOS(T_stab, V_spec, N_spec_cpu; model)
    ρ_mix = sum(N_spec_cpu .* model.Mw) / V_spec
    Scale = vcat(N_spec_cpu, V_spec, model.T_c)
    
    # First attempt: Use provided initial guess if available
    if x_guess !== nothing
        # Convert guess to GPU
        x_guess_gpu = CuArray(x_guess)
        
        success, flash_result = attempt_two_phase_flash_gpu(
            x_guess_gpu, U_spec, V_spec, N_spec_gpu, model, Scale, 
            T_stab, S_one, numPhases
        )

        if success
            @info "Flash with Initial guess succeeded"
            result = (status = success, flash_result = flash_result)
            return result
        end
    end
    
    # Second attempt: Stability-based initialization (GPU version)
    success, flash_result = stability_analysis_fallback_gpu(
        U_spec, V_spec, N_spec_gpu, model, T_stab, S_one, 
        Scale, numPhases
    )
    
    result = (success = success, flash_result = flash_result)
    
    return result   
end

# High-level flash calculation interface - GPU version
function flash_calculation_gpu(U_spec, V_spec, N_spec_gpu; digits::Int=3, atol::Float64=1e-8,
                              model, singlePhaseSure=false, x_guess=nothing)
    
    status, flash_result = solve_rho_Q_from_UVN_gpu(U_spec, V_spec, N_spec_gpu; 
                                                   model, singlePhaseSure, x_guess)
    return (status, flash_result)
end

end