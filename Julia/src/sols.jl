module Sols
using Dates
using ..Scalers
using ..EOS
using ..Flash
using ..Stability
using Random
using StaticArrays
using ForwardDiff
using LinearAlgebra
using SHA

function ThermoProperty(func, prob, sol, model; digits=6)
    V_spec, N_spec = prob.T.V, prob.T.N
    
    n = model.Nc
    N1 = sol[1:n]
    V1 = sol[n + 1] 
    T = prob.T_spec
    N2 = prob.T.N .- N1
    V2 = V_spec .- V1 < eps(Float64) ? eps(Float64) : V_spec .- V1
    @show V1, V2, N1, N2
    Q_II = func(T, V1, N1; model) + func(T, V2, N2; model)
    # Q_I = func(T_spec, V_spec, N_spec; model)
    Q_II
    # @show  round(S_I, digits=digits) round(S_II, digits=digits)
end


function perturb_solution_gaussian(x; σ=0.01)
    seed_value = 12345  #  can change this value to use a different seed
    Random.seed!(seed_value)
    magnitudes = abs.(x)
    σs = σ .* magnitudes
    # noise = x .+ σ .* randn()

    # Generate Gaussian white noise for each variable
    # Scaling noise by the magnitude of each variable
    noise = σs .* randn(length(x))

    perturbed_solution = x .+ noise
end

function perturb_solution(x; ϵ=0.1)
    x_perturbed = @. x + ϵ * x
    return x_perturbed
end

function compute_single_phase_state(U_spec, V_spec, N_spec, model; T_guess=300.0)
    # First attempt with given T_guess (if provided)
    if T_guess !== nothing
        T_single = EOS.GetTemperatureForSpecifiedUV(; 
            U=U_spec, V=V_spec, z=N_spec, model, T_guess, verbose=false
        )
        if is_physically_valid_single_phase(T_single; T_upper_limit=4000, T_lower_limit=50)
            return T_single
        end
    end

    # Try incrementally from 300 K to 2000 K in steps of 100 K
    for T_init in 300:100:2000
        T_single = EOS.GetTemperatureForSpecifiedUV(; 
            U=U_spec, V=V_spec, z=N_spec, model, T_guess=T_init, verbose=false
        )
        if is_physically_valid_single_phase(T_single; T_upper_limit=4000, T_lower_limit=50)
            @info "Single-phase state found at T = $T_single K using initial guess $T_init K"
            return T_single
        end
    end

    @error "Single-phase state computation yielded unphysical temperature for all trial guesses (300–2000 K)."
    return NaN  # Indicate failure to find a valid temperature
end

function extract_phase_properties(res, V_spec, N_spec, model, ρ_mix)
    nc = model.Nc
    T = res[end]
    α = res[nc + 1]
    V_G = α * V_spec
    ρG = res[1:nc] .* ρ_mix
    N_G = ρG .* V_G ./ model.Mw
    N_L = N_spec .- N_G
    V_L = V_spec - V_G
    return T, α, V_G, N_G, V_L, N_L
end

function is_physically_valid_phase(T, V, N; N_spec, V_spec, T_upper_limit = 410, T_lower_limit = 100)
    all(N .≥ 0) && all(N .≤ N_spec) && (1e-10 ≤ V ≤ V_spec - 1e-10) && (T_lower_limit < T < T_upper_limit)
end

# TODO: Add more physical constraints for single-phase validity
# Currently, only temperature limits are checked
# Check pressure. It should be positive and within a reasonable range
function is_physically_valid_single_phase(T ; T_upper_limit = 410, T_lower_limit = 100)
    (T_lower_limit < T < T_upper_limit)
end

function is_valid_two_phase(T, α, V_G, N_G, V_L, N_L, N_spec, V_spec, model, S_one)
    # Physical constraints
    valid_phase = is_physically_valid_phase(T, V_G, N_G; N_spec, V_spec)
                # all(N_G .≥ 0) && all(N_L .≥ 0) &&
                #   (1e-8 ≤ V_G ≤ V_spec - 1e-8) &&
                #   (100.0 < T < 410.0)
    
    # Entropy validation
    valid_entropy = if valid_phase
        S_two = EOS.S_EOS(T, V_G, N_G; model) + EOS.S_EOS(T, V_L, N_L; model)
        S_two ≥ S_one
    else
        false
    end
    
    return valid_phase && valid_entropy
end

function flash_trivial_solution(α; tol=1e-8)
    α < tol || α > (1 - tol) 
end

function attempt_two_phase_flash(initial_guess, U_spec, V_spec, N_spec, model,
                                 Scale, T_stab, S_one, numPhases, useNewtonJulia=nothing)
    ρ_mix = sum(N_spec .* model.Mw) / V_spec
    cons(_) = nothing  # Dummy constraint function
    
    x_spec = vcat(U_spec, V_spec, N_spec)
    # @info "Attempting two-phase flash with initial guess: $initial_guess"
    func_to_optimize(x) = Flash.rho_α_Q(x; U_spec, V_spec, N_spec, model, numPhases, Scale)
    g(x) = ForwardDiff.gradient(func_to_optimize, x)
    H(x) = ForwardDiff.hessian(func_to_optimize, x)

    flash_converged, res, iterations = Flash.OptimizeHelmholtz(func_to_optimize,
        g, H,
        initial_guess, 
        cons; 
        tol=1e-6, 
        maxiter=1000,
        V_total=V_spec, 
        N_total=N_spec
    )
    # @show H(res)
    # @show extra_info
    # @show iterations
    if !flash_converged 
        return false, nothing  # Flash did not converge
    end
    T, α, V_G, N_G, V_L, N_L = extract_phase_properties(res, V_spec, N_spec, model, ρ_mix)
    
    U_G = EOS.U_EOS(T, V_G, N_G; model)
    U_L = EOS.U_EOS(T, V_L, N_L; model)
    # @show  α
    is_trivial = flash_trivial_solution(α) 
    # is_all_liq_trivial = flash_trivial_solution(α) 
    # is_trivial = is_all_gas_trivial || is_all_liq_trivial
    # @info "Trivial: $is_trivial, α = $α"
    # @info "User GuessFlash: $res, converged: $flash_converged, Trivial: $is_trivial, α = $α"
    # @info "Flash: $res, converged: $flash_converged, Trivial: $is_trivial, α = $α"
    is_valid = is_valid_two_phase(T, α, V_G, N_G, V_L, N_L, N_spec, V_spec, model, S_one)
    # @info "Is two phase trivial solution: $is_trivial, is valid: $is_valid"
    if is_valid && !is_trivial
        return true, vcat(N_G, V_G, T)
    end
    return false, nothing
end

function func_uvn(x; TVN_sol,  model)
    U_spec = x[1]
    V_spec = x[2]
    N_spec = x[3:end]
    kw_uvn = (model=model, numPhases=2, Scale=vcat(N_spec, V_spec, 300.0), ScaleFunc=Scalers.NoScale())
    Flash.UV_Q(TVN_sol; U_spec, V_spec, N_spec, kw_uvn...)
end

function solve_UVFlash_QFuncVer3(U_spec, V_spec, N_spec; initial_guess = nothing, model, ϵ=0.0, stability_cache=nothing, useNewtonJulia=true)
    # @show U_spec, V_spec, N_spec
    nc = model.Nc
    T_stab = compute_single_phase_state(U_spec, V_spec, N_spec, model)
    if isnothing(initial_guess)
        # @info "No initial guess provided, computing stability-based initialization."
        
        stab = Stability.VT_stabilityAnalysis(; model, T_spec=T_stab, V_spec, z_spec=N_spec, stability_cache = nothing)
        if !stab.isunstable
            is_good_single_phase = is_physically_valid_single_phase(T_stab)
            if is_good_single_phase
                # @info "Single-phase solution is valid, returning it directly. Temperature: $T_stab"
                single_phase_sol = vcat(N_spec, V_spec, T_stab)
                # @show single_phase_sol
                return (status = :success, res = single_phase_sol, Jacobian = nothing)
            else
                @warn "Single-phase solution is not valid. Trying stability-based initialization."
                # T_stab = 300.0  # Default temperature
                # return vcat(N_spec, V_spec, T_stab)
            end
        end
        # @show stab
        initial_guess = Stability.IG3(stab.c, T_stab, U_spec, V_spec, N_spec; model, verbose=false)         
        append!(initial_guess, T_stab)
    end   
        
    Scale = vcat(N_spec, V_spec, 300.0)  
    kw = (U_spec=U_spec, V_spec=V_spec, N_spec=N_spec, model=model, numPhases=2, Scale=Scale, ScaleFunc=Scalers.NoScale())
    kw_uvn = (model=model, numPhases=2, Scale=Scale, ScaleFunc=Scalers.NoScale())
    # optimizer = Flash.applyScaleDescale(Flash.OptimizeHelmholtz; Scale=kw.Scale, ScaleFunc=kw.ScaleFunc)
    
    func_to_optimize1(x) = Flash.UV_Q(x; kw...)
    g(x) = ForwardDiff.gradient(func_to_optimize1, x)
    H(x) = ForwardDiff.hessian(func_to_optimize1, x)

    
    # g_uvn(x; y) = ForwardDiff.gradient(func_uvn, x; y=y)
    # H_uvn(x; y) = ForwardDiff.hessian(func_uvn, x; y=y)

    cons(x) = nothing  # Dummy constraint function
    success, res = Flash.OptimizeHelmholtz(func_to_optimize1,
        g, H,
        initial_guess, 
        cons; 
        tol=1e-6, 
        maxiter=1000,
        V_total=V_spec, 
        N_total=N_spec
    )
    # @show g(res)
    # # func_uvn(x) = Flash.UV_Q(res; U_spec = x[1], V_spec = x[2], N_spec = x[3:end], kw_uvn...)
    # myFunc(x) = func_uvn(x; TVN_sol = res, model)
    # g_uvn(x) = ForwardDiff.gradient(myFunc, x)
    # J_uvn(x) = ForwardDiff.jacobian(g_uvn, x)
    # H_uvn(x) = ForwardDiff.hessian(myFunc, x)
    # @show g_uvn(vcat(U_spec, V_spec, N_spec))
    # @show J_uvn(vcat(U_spec, V_spec, N_spec))
    # @show H_uvn(vcat(U_spec, V_spec, N_spec))

    func_to_optimize(y, U, V, N) = begin
        kw = (U_spec=U, V_spec=V, N_spec=N, model=model,
              numPhases=2, Scale=vcat(N,V,300.0), ScaleFunc=Scalers.NoScale())
        Flash.UV_Q(y; kw...)
    end

    # -----------------------------------------
    # Hessian wrt y (G_y = ∂g/∂y)
    G_y = ForwardDiff.hessian(y -> func_to_optimize(y, U_spec, V_spec, N_spec), res)
    
    # -----------------------------------------
    # Partial wrt parameters (G_p = ∂g/∂(U,V,N))
    # Avoid nested gradient calls by evaluating gradient w.r.t y inside the jacobian of parameters
    function g_wrt_y(y, p)
        U, V, N = p[1], p[2], p[3:end]
        return ForwardDiff.gradient(y -> func_to_optimize(y, U, V, N), y)
    end

    p_vec = vcat(U_spec, V_spec, N_spec)
    G_p = ForwardDiff.jacobian(p -> g_wrt_y(res, p), p_vec)
    
    # -----------------------------------------
    # Implicit differentiation Jacobian
    J = - G_y \ G_p
    # @show J
    # sol = H(res) \ J
    # @show sol
    # success, res = optimizer(func_to_optimize, g, H, initial_guess; tol=1e-8, maxiter=300, α=1.0, useNewtonJulia)
    T = res[end]
    V_G = res[nc+1]
    N_G  = res[1:nc]
    V_L = V_spec - V_G
    N_L = N_spec .- N_G
    α = V_G / V_spec  # Extract α from result
    is_trivial = flash_trivial_solution(α)        
    S_one = EOS.S_EOS(T_stab, V_spec, N_spec; model)
    is_valid = is_valid_two_phase(T, α, V_G, N_G, V_L, N_L, N_spec, V_spec, model, S_one)
    
    if success && !is_trivial && is_valid
        (status = :success, res = res, Jacobian = J)
    else
        (status = :failed, res = res, Jacobian = J)
    end

end

function solve_UVFlash_QFuncVer2(U_spec, V_spec, N_spec; initial_guess = nothing, model, ϵ=0.0, stability_cache=nothing, useNewtonJulia=true, maxiter=300)
    # @show U_spec, V_spec, N_spec
    
    nc = model.Nc
    T_stab = compute_single_phase_state(U_spec, V_spec, N_spec, model)
    if isnothing(initial_guess)
        # @info "No initial guess provided, computing stability-based initialization."
        
        stab = Stability.VT_stabilityAnalysis(; model, T_spec=T_stab, V_spec, z_spec=N_spec, stability_cache = nothing)
        if !stab.isunstable
            is_good_single_phase = is_physically_valid_single_phase(T_stab)
            if is_good_single_phase
                # @info "Single-phase solution is valid, returning it directly. Temperature: $T_stab"
                single_phase_sol = vcat(N_spec, V_spec, T_stab)
                # @show single_phase_sol
                return (status = :success, res = single_phase_sol, iterations = 0, Jacobian = nothing)
            else
                @warn "Single-phase solution is not valid. Trying stability-based initialization."
                # T_stab = 300.0  # Default temperature
                # return vcat(N_spec, V_spec, T_stab)
            end
        end
        # @show stab
        initial_guess = Stability.IG3(stab.c, T_stab, U_spec, V_spec, N_spec; model, verbose=false)         
        append!(initial_guess, T_stab)
    end   
        
    Scale = vcat(N_spec, V_spec, 300.0)  
    kw = (U_spec=U_spec, V_spec=V_spec, N_spec=N_spec, model=model, numPhases=2, Scale=Scale, ScaleFunc=Scalers.NoScale())
    kw_uvn = (model=model, numPhases=2, Scale=Scale, ScaleFunc=Scalers.NoScale())
    # optimizer = Flash.applyScaleDescale(Flash.OptimizeHelmholtz; Scale=kw.Scale, ScaleFunc=kw.ScaleFunc)
    
    func_to_optimize1(x) = Flash.UV_Q(x; kw...)
    g(x) = ForwardDiff.gradient(func_to_optimize1, x)
    H(x) = ForwardDiff.hessian(func_to_optimize1, x)


    cons(x) = nothing  # Dummy constraint function
    success, res, iterations = Flash.OptimizeHelmholtz(func_to_optimize1,
        g, H,
        initial_guess, 
        cons; 
        tol=1e-10, 
        maxiter,
        V_total=V_spec, 
        N_total=N_spec
    )

   
    # success, res = optimizer(func_to_optimize, g, H, initial_guess; tol=1e-8, maxiter=300, α=1.0, useNewtonJulia)
    T = res[end]
    V_G = res[nc+1]
    N_G  = res[1:nc]
    V_L = V_spec - V_G
    N_L = N_spec .- N_G
    α = V_G / V_spec  # Extract α from result
    is_trivial = flash_trivial_solution(α)        
    S_one = EOS.S_EOS(T_stab, V_spec, N_spec; model)
    is_valid = is_valid_two_phase(T, α, V_G, N_G, V_L, N_L, N_spec, V_spec, model, S_one)
    
    if success && !is_trivial && is_valid
        (status = :success, res = res, iterations = iterations, Jacobian = nothing)
    else
        (status = :failed, res = res, iterations = iterations, Jacobian = nothing)
    end

end

function stability_analysis_fallback(U_spec, V_spec, N_spec, model, T_stab, S_one, 
                                     Scale, numPhases, useNewtonJulia, stability_cache)
                                     
    stab = Stability.VT_stabilityAnalysis(; model, T_spec=T_stab, V_spec, z_spec=N_spec, stability_cache)
    # initial_approximations = Stability.generate_all_initial_approximations(T_stab, V_spec, N_spec, model)
    # N_G = initial_approximations[1][2:end]
    # V_G = initial_approximations[1][1]
    
    is_uncertain = abs(stab.bestD) ≤ 1e-4  
    # @info "Stability analysis result: D = $(stab.bestD), is_uncertain = $is_uncertain, stab.isunstable = $(stab.isunstable), T_stab = $T_stab K"
    if !stab.isunstable
        is_good_single_phase = is_physically_valid_single_phase(T_stab; T_upper_limit = 600)
        if is_good_single_phase
            # @info "Single-phase solution is valid, returning it directly."
            return true, vcat(N_spec, V_spec, T_stab)
        else
            @warn "Single-phase solution is not valid, trying stability-based initialization."
            # T_stab = 300.0  # Default temperature
            return false, vcat(N_spec, V_spec, T_stab)
        end
    end
    x0 = Stability.IG3(stab.c, T_stab, U_spec, V_spec, N_spec; model, verbose=false)    
    ρ_mix = sum(N_spec .* model.Mw) / V_spec
    # @show x0    
    # Extract and validate stability-based initial guess
    N_G = x0[1:model.Nc]
    V_G = x0[model.Nc+1]
    if !(all( 0 .≤ N_G .≤ N_spec)) || !(0 < V_G < V_spec)
        return false, nothing
    end
    
    # Compute initial vapor fraction
    N_L = N_spec .- N_G
    V_L = V_spec - V_G
    ρG = N_G .* model.Mw ./ V_G
    ρL = N_L .* model.Mw ./ V_L

    ϵ = 1e-12  # Small value to avoid division by zero

    denominator = abs(ρL[1] - ρG[1]) < ϵ ? ϵ : ρL[1] - ρG[1]

    # total mixture density of the first component
    ρ_1= N_spec[1] * model.Mw[1] / V_spec
    
    α_est = (ρL[1] - ρ_1) / denominator
    α_est = clamp(α_est, 1e-8, 1-1e-8)  # Avoid extreme values
    
    x_init = vcat(ρG ./ ρ_mix, α_est, T_stab)
    if sum(ρG) < 1e-8
        @warn "Vapor density is too low."
    end
    if sum(ρG) > 1000
        @warn "Vapor density is too high."        
    end
    
    # Attempt flash with stability-based guess
    success, result = attempt_two_phase_flash(
        x_init, U_spec, V_spec, N_spec, model, Scale, 
        T_stab, S_one, numPhases, useNewtonJulia
    )
    success && return true, result
    
    return false, nothing
end

function solve_rho_Q_from_UVN(U_spec, V_spec, N_spec::MVector; model, singlePhaseSure=false, 
                              ϵ=0.0, x_guess=nothing, useNewtonJulia=true, factor=1.0, numPhases=2, stability_cache=nothing)
    # @show U_spec, V_spec, N_spec
    # Compute single-phase reference state

    T_stab = compute_single_phase_state(U_spec, V_spec, N_spec, model)
    
    if singlePhaseSure
        # If single-phase solution is guaranteed, return it directly
        is_good_single_phase = is_physically_valid_single_phase(T_stab)
        status = is_good_single_phase ? :success : :failed
        result = (status = status, flash_result = vcat(N_spec, V_spec, T_stab))
        # use_cache && store_uvn_flash_result(stability_cache, U_spec, V_spec, N_spec, result)
        return result
    end
    S_one = EOS.S_EOS(T_stab, V_spec, N_spec; model)
    ρ_mix = sum(N_spec .* model.Mw) / V_spec
    Scale = vcat(N_spec, V_spec, model.T_c)
    
    # First attempt: Use provided initial guess if available
    if x_guess !== nothing
        try
            success, flash_result = attempt_two_phase_flash(
            x_guess, U_spec, V_spec, N_spec, model, Scale, 
            T_stab, S_one, numPhases, useNewtonJulia)

        if success
            @info "Flash with Initial guess succeeded"
            result = (status = :success, flash_result = flash_result)
            # use_cache && store_uvn_flash_result(stability_cache, U_spec, V_spec, N_spec, result)
            return result
        else
            # @warn "Initial guess failed; proceeding with stability analysis for cell $(stability_cache.cell_index[1]) at iteration $(stability_cache.it[1]) "
        end
        catch
            @warn "Poor initial guess provided; proceeding with stability analysis."
        end
        
    end
    
    # Second attempt: Stability-based initialization
    success, flash_result = stability_analysis_fallback(
        U_spec, V_spec, N_spec, model, T_stab, S_one, 
        Scale, numPhases, useNewtonJulia, stability_cache
    )
    if success
        result = (status = :success, flash_result = flash_result)
        # use_cache && store_uvn_flash_result(stability_cache, U_spec, V_spec, N_spec, result)
        return result
    end
    
    return (status = :failed, flash_result = vcat(N_spec, V_spec, T_stab))  # No valid solution found

    # Fallback: Single-phase solution
    # is_good_single_phase = is_physically_valid_single_phase(T_stab)
    # status = is_good_single_phase ? :success : :failed
    # result_single_phase = (status = status, flash_result = vcat(N_spec, V_spec, T_stab))
    # use_cache && store_uvn_flash_result(stability_cache, U_spec, V_spec, N_spec, result_single_phase)
    # return result_single_phase
end


# High-level flash calculation interface
function flash_calculation(U_spec, V_spec, N_spec::MVector; digits::Int=3, atol::Float64=1e-8,
    model, singlePhaseSure=false, x_guess=nothing, stability_cache = nothing)
    
    status, flash_result = solve_rho_Q_from_UVN(U_spec, V_spec, N_spec; model, singlePhaseSure, x_guess, stability_cache)
    return (status, flash_result)
    
end


end