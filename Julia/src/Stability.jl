module Stability

using LinearAlgebra
using ForwardDiff
using ExportAll
# using FiniteDiff
using NLsolve
using StaticArrays
using ..EOS
using ..CubicFuncs
using ..Solvers
# using ..Solvers
using Random
using FixedPointAcceleration
using SHA
using CUDA
using Distributions

function sample_simplex(d::Int, n::Int; α_low=0.1, α_high=10.0, rng::AbstractRNG=Random.GLOBAL_RNG)
    n_uniform = max(1, floor(Int, n / 3))
    n_corners = max(1, floor(Int, n / 3))
    n_center = n - n_uniform - n_corners

    uniform_pts = rand(rng, Dirichlet(d, 1.0), n_uniform)
    corner_pts = rand(rng, Dirichlet(d, α_low), n_corners)
    center_pts = rand(rng, Dirichlet(d, α_high), n_center)
    
    return hcat(uniform_pts, corner_pts, center_pts)
end

# d is the dimension of the simplex, n is the number of points to sample
# f is a function that takes a point in the simplex and returns a scalar value. f will be TPD function here.
function sample_simplex_condition(d::Int, n::Int, f::Function; scale,
                 seed::Int=42, α_low=0.1, α_high=10.0,
                 max_trials=100_000, batch_size=1000, verbose=true)
    
    rng = MersenneTwister(seed)
    accepted_points = Vector{Vector{Float64}}()
    accepted_values = Float64[]
    total_trials = 0

    while length(accepted_points) < n && total_trials < max_trials
        # Generate batch of candidates
        candidates = sample_simplex(d, batch_size; α_low, α_high, rng)
        
        # Process each candidate in batch
        for i in 1:batch_size
            scaled_x = candidates[:, i]
            # Normalise to ensure we have the correct scale
            x = scaled_x .* scale
            fx = f(x)
            
            if fx > 0
                push!(accepted_points, x)
                push!(accepted_values, fx)
                # Stop if we have enough points
                length(accepted_points) >= n && break
            end
        end
        
        total_trials += batch_size
        # verbose && println("Trials: $total_trials, Accepted: $(length(accepted_points))")
    end

    # Handle case where no points were found
    if isempty(accepted_points)
        @warn "No points satisfying f(x) > 0 found"
        return Matrix{Float64}(undef, d, 0), Float64[]
    end

    # Sort by f(x) descending
    sorted_order = sortperm(accepted_values, rev=true)
    sorted_points = hcat(accepted_points[sorted_order]...)
    sorted_values = accepted_values[sorted_order]

    # Return exactly min(n, num_accepted) points with their values
    n_final = min(n, length(accepted_points))
    return sorted_points[:, 1:n_final], sorted_values[1:n_final]
end

# c is the vector of concentrations, c_i = N_i / V, V is the volume of the phase p
function stability(c::MVector; T_spec, V_spec, N_spec::MVector, model)
    
    c_spec = N_spec ./ V_spec
    μ_ref = μ_EOS(T_spec, 1.0, c_spec; model)
    function create_prob(y::MVector) 
        x = exp.(y)        
        Δμs = μ_EOS(T_spec, 1.0, x; model) - μ_ref    
        # MVector(Δμs...)         
        # chem_pot_diff = chem_pot(T_spec, 1.0, x; model) .- chem_pot(T_spec, 1.0, c_spec; model)
    end
    
    # f = create_prob
    g(x) = ForwardDiff.jacobian(create_prob, x)
    H(x) = ForwardDiff.hessian(create_prob, x)
    x0 = MVector{length(c)}(log.(c))
    x, converged, iters = Solvers.newton_stability(create_prob, g, H, x0; tol=1e-8, maxiter=50)
    sol1 = exp.(x)
    
    # sol2 = nlsolve(create_prob, log.(c), xtol=1e-8, ftol=1e-8, method=:newton, linesearch=LineSearches.BackTracking(order=3))
    # exp.(sol2.zero)
    # sol.zero
end

function generate_perturbed_guesses2(c_spec, k; T_spec, model, noise_level=0.1)
    n = length(c_spec)
    guesses = []
    noise_cpu = noise_level * randn(n, k)
    for idx in 1:k
        # Add random Gaussian noise and ensure positivity
        
        # Normalize to preserve total concentration (L∞ stable)
        # c_norm = c * (sum(c_spec) / sum(c))
        # noise[i, idx]
        for i in 1:n
            # Add random Gaussian noise and ensure positivity
            perturbed_val = c_spec[i] * (1.0 + noise_cpu[i, idx])
            perturbed_val = max(perturbed_val, 1e-12)
            c_perturbed[i] = perturbed_val
        end
        # c_perturbed = max.(c_spec .* (1 .+ noise_level * randn(n)), 1e-12)
        # c_perturbed = c_perturbed * (sum(c_spec) / sum(c_perturbed))
        push!(guesses, c_perturbed)
        c_perturbed = c_perturbed ./ sum(c_perturbed)
        push!(guesses, c_perturbed)
    end

    return guesses
end

function generate_perturbed_guesses_orig(c_spec, k; T_spec, model, noise_level=0.1)
    n = length(c_spec)
    guesses = []
    # noise_cpu = noise_level * randn(n, k)
    for _ in 1:k
        # Add random Gaussian noise and ensure positivity
        
        # Normalize to preserve total concentration (L∞ stable)
        # c_norm = c * (sum(c_spec) / sum(c))
        # noise[i, idx]        
        c_perturbed = max.(c_spec .* (1 .+ noise_level * randn(n)), 1e-12)
        c_perturbed = c_perturbed * (sum(c_spec) / sum(c_perturbed))
        push!(guesses, c_perturbed)
        c_perturbed = c_perturbed ./ sum(c_perturbed)
        push!(guesses, c_perturbed)
    end

    return guesses
end

function generate_perturbed_guesses(c_spec, k; T_spec, model, noise_level=0.1, rng_seed=nothing)
    n = length(c_spec)
    guesses = Vector{Vector{Float64}}()
    
    # Pre-generate Gaussian noise: each column is a noise vector
    if rng_seed !== nothing
        Random.seed!(rng_seed)
    end
    noise_cpu = noise_level .* randn(n, k)
    # @show noise_cpu
    for j in 1:k
        # Get precomputed noise for this sample
        δ = @view noise_cpu[:, j]

        # Apply noise: c_spec .* (1 .+ δ)
        c_perturbed = similar(c_spec)
        @inbounds for i in 1:n
            c_perturbed[i] = max(c_spec[i] * (1 + δ[i]), 1e-12)
        end

        # Normalize to preserve total concentration
        scale = sum(c_spec) / sum(c_perturbed)
        @inbounds for i in 1:n
            c_perturbed[i] *= scale
        end

        push!(guesses, copy(c_perturbed))

        # Normalize to sum = 1 version
        s = sum(c_perturbed)
        c_norm = (1/s) .* c_perturbed
        push!(guesses, c_norm)
    end

    return guesses
end



function is_approximately_equal(a, b; atol=1e-1)
    return length(a) == length(b) && all(abs.(a .- b) .< atol)
end

# Helper function to compute c_spec
compute_c_spec(V_spec, z_spec) = z_spec ./ V_spec

# Generate all initial approximations
function generate_all_initial_approximations(T_spec, V_spec, z_spec, model, rng_seed=nothing)
    c_spec = compute_c_spec(V_spec, z_spec)
      
    simplex = generate_smejkal_simplex_based_approximations(; T_spec, model)    
    saturation = initialize_phase_stability(T_spec, V_spec, z_spec; model)
    perturbed = generate_perturbed_guesses(c_spec, 10; T_spec, model, noise_level=0.1, rng_seed)
    
    # error("Initial approximations should be a matrix with each column as a point in the simplex")
    return vcat(simplex, saturation, perturbed)
end



function is_trivial_solution(c, c_spec; tol=1e-4)
    rel_error = norm(c .- c_spec, Inf) / norm(c_spec, Inf)
    return rel_error < tol
end

# Create feasibility check function
function make_feasibility_check(model, ϵ=0.0)
    n = model.Nc
    bi = [b_i(; i=i, model) for i in 1:n]
    return x -> sum(bi .* x) <= 1.0 && all(x .>= ϵ) && !any(isnan.(x))
end



function process_trial_point(c::MVector, T_spec, V_spec, z_spec::MVector, model, digits, feasibility_check, c_spec::MVector)   
    α = -100.0
    # c = c_trial    
    # try
        c_trial = stability(MVector(c...); T_spec, V_spec, N_spec=z_spec, model)        
    # catch e        
    #     @error "Newton failed: ", e
    #     return (status=:failed, error_message = e, c=nothing, D_trial=nothing, trivial=nothing, feasible=nothing, α =nothing)
    # end
    
    # @show "Processed trial point: $c with α = $α"
    # Compute D_trial and check solution properties
    U_trial = U_EOS(T_spec, 1.0, c_trial; model)
    D_trial = VT_D(c_trial; T_spec, V_spec, z_spec, model)
    
    trivial = is_trivial_solution(c_trial, c_spec)
    feasible = feasibility_check(c_trial)
    # @info "D_trial = $D_trial for composition $c_trial, trivial=$trivial, feasible=$feasible"
    return (; c=c_trial, D_trial, α = -100.0, trivial, feasible)
end

# Post-process composition and D value
function postprocess_solution(c::MVector, D_trial)
    return c, D_trial
    
    # n = length(c)
    # c_processed = [abs(comp) < 1e-4 ? eps(comp) : comp for comp in c]
    # D_processed = D_trial < 0 && abs(D_trial) < 1e-4 ? eps(abs(D_trial)) : D_trial
    # return c_processed, D_processed
end

# Main stability analysis function
function VT_stabilityAnalysis(; T_spec, V_spec, z_spec::MVector, model)
    c_spec = compute_c_spec(V_spec, z_spec)
    n = model.Nc
    digits = 2    

    # Generate initial approximations and setup
    initial_approximations = generate_all_initial_approximations(T_spec, V_spec, z_spec, model)
    feasibility_check = make_feasibility_check(model, 1e-8)    
    isunstable = false
    
    c_sol = c_spec    
    bestD = -Inf    
    first_counter = -1
    counter = 1
    
    # Process each trial point
    for x in initial_approximations        
        c_trial = MVector(x[1:end]...)
        trial = process_trial_point(c_trial, T_spec, V_spec, z_spec, model, digits, feasibility_check, MVector(c_spec...))               
        
        # Process converged results
        # c_trial, D_trial = postprocess_solution(MVector(trial.c...), trial.D_trial)
        D_trial = trial.D_trial
        # Check for valid unstable solution
        if !isnan(D_trial) && !trial.trivial && trial.feasible && D_trial >= 0 
            isunstable = true
            # @info "Valid unstable solution found: c_trial = $c_trial, D_trial = $D_trial"
            first_counter == -1 && (first_counter = counter)
                        
            if bestD < D_trial
                bestD = D_trial
                c_sol = trial.c
            end
        end

        counter += 1
    end

    result = (; T_trial=T_spec, D_trial=bestD, isunstable, c_sol, c_spec, iterations=first_counter)
    
    return result
    
end


function initialize_density_guess(T, x, Pini, phase; model)
    n_total = 1.0          # Assume normalized phase mole number
    N_trial = x .* n_total
    #get_PR_volume(P, T, z; model, phase = :unknown)
    V_roots = CubicFuncs.get_PR_volume(Pini, T, N_trial; model=model, phase)  # Should return all real roots

    if isempty(V_roots)
        @warn "No valid volume roots found for trial composition"
        return []
    end

    # Discard middle root if 3 exist
    if length(V_roots) == 3
        V_candidates = [minimum(V_roots), maximum(V_roots)]
    else
        V_candidates = V_roots
    end

    # Choose root with minimum Gibbs energy
    G_values = [EOS.G_EOS(T, V, N_trial; model=model) for V in V_candidates]
    V_best = V_candidates[argmin(G_values)]

    d0 = x ./ V_best
    u0 = EOS.U_EOS(T, V_best, N_trial; model=model)
    
    return [d0]
    # return [vcat(d0, u0)]
end

function initialize_phase_stability(T_spec, Vspec, N0; model)
    ∑ = sum
    z = N0 ./ ∑(N0)                # Mole fractions
    n_total = ∑(N0)
    n_components = length(N0)

    guesses = []
    P_c = model.P_c
    T_c = model.T_c
    ω = model.ω
    compute_K(T_spec, P_spec) = log.(P_c ./ P_spec) .+ (5.373 * (1 .+ ω) .* (1 .- T_c ./ T_spec))
    #### 1. Try to compute initial pressure from EoS (explicit or numerical)
    # try
        P0 = EOS.press(T_spec, Vspec, N0; model=model)
        if P0 > 0
            # Use K-values at (T_spec, P0)
            
            lnK0 = compute_K(T_spec, P0)
            K0 = exp.(lnK0)
            # Type L: vapor-like trial phase, Eq 46 Nichita 2017, xV0 = mole fractions
            xV0 = z .* K0
            xV0 ./= ∑(xV0)
            push!(guesses, initialize_density_guess(T_spec, xV0, P0, :liquid; model))

            # Type V: liquid-like trial phase
            xL0 = z ./ K0
            xL0 ./= ∑(xL0)
            push!(guesses, initialize_density_guess(T_spec, xL0, P0, :vapor; model))
        end
    # catch
    #     @warn "Failed to compute P0 from EoS. Falling back to Wilson estimates."
    # end

    #### 2. If P0 ≤ 0 or failed, use Wilson correlation
    # Wilson’s equation
    Psat = similar(z)
    for i in 1:n_components
        Psat[i] = P_c[i] * exp(5.373 * (1 + ω[i]) * (1 - T_c[i] / T_spec))
    end
    
    # Type L (vapor trial)
    Pini_L = ∑(z .* Psat)
    xV0 = z .* Psat
    xV0 ./= ∑(xV0)
    push!(guesses, initialize_density_guess(T_spec, xV0, Pini_L, :liquid; model))

    # Type V (liquid trial)
    Pini_V = 1 / ∑(z ./ Psat)
    xL0 = z ./ Psat
    xL0 ./= ∑(xL0)
    push!(guesses, initialize_density_guess(T_spec, xL0, Pini_V, :vapor; model))

    return vcat(guesses...)
end

function P_sat(T; model)
    n = model.Nc
    PSat = zeros(n)
    for i in 1:n
        PSat[i] = model.P_c[i] * exp(5.373 * (1 + model.ω[i]) * (1 - model.T_c[i] / T))
    end
    return PSat
end

function isFeasible_orig(x, increment=zero.(x); model, verbose=false)
    n = model.Nc
    # increment = randn(Float64, length(x)) * 1.0
    x_new = x #.+ increment
    Nᴵ = x_new[1:n]
    Nᴵᴵ = x_new[n+3:2*n+2]

    _b_i = [b_i(; i, model) for i = 1:n]
    # Volume and molar quantity must be positive
    for i in 1:(2*n+4)
        if i != (n + 2) && i != (2 * n + 4)
            if x_new[i] < 0.0
                verbose && println("Volume or molar quantity is negative.")
                return false
            end
        end
    end

    # Castier puts this condition, but Mikyskya doesn't

    # if any(x-> x < 0, Nᴵᴵ .- Nᴵ)
    #     return false
    # end

    # ∑Nᵢ'bᵢ' < V' for both phases
    # Sum from Peng first phase
    ∑Nᵢᴵbᵢᴵ = sum(Nᴵ .* _b_i)
    if ∑Nᵢᴵbᵢᴵ >= x_new[n+1]
        verbose && println("Sum from Peng first is greater than V1.")
        return false
    end

    # Sum from Peng second phase
    ∑Nᵢᴵᴵbᵢᴵᴵ = sum(Nᴵᴵ .* _b_i)
    if ∑Nᵢᴵᴵbᵢᴵᴵ >= x_new[2*n+3]
        verbose && println("Sum from Peng second is greater than V2.")
        return false
    end

    return true
end

function isFeasible(x, increment=zero.(x); model, verbose=false)
    n = model.Nc
    x_new = x

    Nᴵ = x_new[1:n]
    Vᴵ = x_new[n+1]
    Nᴵᴵ = x_new[n+2:2n+1]
    Vᴵᴵ = x_new[2n+2]
    _b_i = [b_i(; i, model) for i = 1:n]

    if !all(x_new .>= 0.0)
        verbose && println("Negative molar quantity.")
        return false
    end

    # ∑Nᵢ'bᵢ' < V' for both phases
    ∑Nᵢᴵbᵢᴵ = sum(Nᴵ .* _b_i)
    if ∑Nᵢᴵbᵢᴵ >= Vᴵ
        verbose && println("Sum from Peng first is greater than V1.")
        return false
    end

    ∑Nᵢᴵᴵbᵢᴵᴵ = sum(Nᴵᴵ .* _b_i)
    if ∑Nᵢᴵᴵbᵢᴵᴵ >= Vᴵᴵ
        verbose && println("Sum from Peng second is greater than V2.")
        return false
    end

    return true
end

function InitialGuessFromStabilityResult(x, T_trial; U_spec, V_spec, N_spec, model, verbose=true)
    n = model.Nc
    result = zeros(eltype(x), 2 * n + 2)  # removed 2 energy entries

    cPrime = x[1:n]

    bi = [b_i(; i=i, model) for i in 1:n]
    smallest_volume = 1.01 * maximum(bi)

    V_trial = 0.5 * V_spec
    TPrime = T_trial #GetTemperatureForSpecifiedUV(; U=U_spec, V=V_spec, z=N_spec, model, T_guess=300.0)
    moleTrial = cPrime .* V_trial

    SOne = S_EOS(T_trial, V_spec, N_spec; model)
    diff = 0.0
    iters = 200

    lambda = 1.0
    while iters > 0
        S_trial = S_EOS(T_trial, V_trial, moleTrial; model)

        U_trial = U_EOS(T_trial, V_trial, moleTrial; model)
        U_bulk = U_spec - U_trial
        V_bulk = V_spec - V_trial
        N_bulk = N_spec .- moleTrial
        T_bulk = GetTemperatureForSpecifiedUV(; U=U_bulk, V=V_bulk, z=N_bulk, model, T_guess=300.0)
        S_bulk = S_EOS(T_bulk, V_bulk, N_bulk; model)

        STwo = S_trial + S_bulk

        diff = STwo - SOne

        result[1:n] = moleTrial        
        result[n+1] = V_trial
        result[n+2:2n+1] = N_spec .- moleTrial
        result[2n+2] = V_spec - V_trial

        iters -= 1

        feasible = isFeasible(result; model, verbose)
        verbose && @show STwo, SOne

        if diff > 0.0 && feasible
            verbose && println("Feasible solution found.")
            verbose && @show result
            break
        end

        if (V_trial / V_spec) < 1e-8
            verbose && println("Failed to find feasible solution (V_trial too small).")
            break
        end

        V_trial /= 2.0
        moleTrial = cPrime .* V_trial
    end

    if diff < 0
        isapproximately_zero = isapprox(diff, 0.0; atol=1e-2)
        isfeasible = isFeasible(result; model, verbose)
        if isapproximately_zero && isfeasible
            return result[n+2:end]
        else
            verbose && println("Failed to find feasible solution.")
            return "Failed to find feasible solution."
        end
    end

    return result[n+2:end]
end

function IG3(c, T, U_spec, V_spec, z_spec; model, factor=10, verbose=false)    
    res = Stability.InitialGuessFromStabilityResult(c, T; U_spec, V_spec, N_spec=z_spec, model, verbose)
end


function coverSimplex(; model)
    _n = model.Nc
    X = [zeros(_n + 1) for _ in 1:(_n+2)]

    bi = [b_i(; i=i, model) for i in 1:_n]
    # @show 1/3bi[1], 1/3bi[2]
    # @show 1/6bi[1],  1/6bi[2]
    # @show 2/3bi[1],  2/3bi[2]
    # bi = [0.02236, 0.02214]
    # barycenter of the simplex
    barycenter = zeros(_n)
    
    #bi = ones(_n)  # Placeholder: Replace with the actual values of _b_i
    for i in 1:_n
        barycenter[i] = 1.0 / ((_n + 1) * bi[i])
    end

    # Simplex top vertices V
    V = [zeros(_n) for _ in 1:(_n+1)]
    for i in 1:_n
        V[i][i] = 1.0 / bi[i]
    end

    # New approximations X
    X[1] = copy(barycenter)
    for i in 2:(_n+2)

        X[i] = 0.5 * (V[i-1] .+ barycenter)
        
    end
    # @show X
    return X
end



# TODO
function generate_smejkal_simplex_based_approximations2(;T_spec, model)
    initial_concentrations = coverSimplex(; model)
    
    temperature_range = [T_spec]
    # temperature_range = collect(100:50.0:400)
    numTemps = length(temperature_range)
    numConc = length(initial_concentrations)
    initial_approximations = [zeros(Float64, length(initial_concentrations[1]) + 1) for _ in 1:numTemps*numConc]
    counter = 1

    
    for j in 1:numConc
        c_trial = initial_concentrations[j]
        for i in 1:numTemps
            T_trial = temperature_range[i]

            # Note that we need internal energy density, not total internal energy. Hence,we use concentrations and not moles
            u_trial = U_EOS(T_trial, 1.0, c_trial; model)
            trial = vcat(c_trial, u_trial)
            
            initial_approximations[counter] = trial
            counter += 1
        end
    end

    initial_approximations

end

function generate_smejkal_simplex_based_approximations(; T_spec, model)
    initial_concentrations = coverSimplex(; model)
    temperature_range = [T_spec]
    numTemps = length(temperature_range)
    numConc = length(initial_concentrations)

    # Twice as many because we alternate normalized and unnormalized
    # initial_approximations = Vector{Vector{Float64}}(undef, 2 * numTemps * numConc)
    T_trial = temperature_range[1]
    c_trial_raw = initial_concentrations[1]
    # u_unnorm = U_EOS(T_trial, 1.0, c_trial_raw; model)
    # trial_unnorm = vcat(c_trial_raw, u_unnorm)
    trial_unnorm = vcat(c_trial_raw)
    initial_approximations = [Vector{eltype(trial_unnorm)}(undef, length(initial_concentrations[1]) + 1) for _ in 1:(2 * numTemps * numConc)]

    # initial_approximations = [MVector{length(initial_concentrations[1]) + 1, eltype(trial_unnorm)}(undef)
    # for _ in 1:(2 * numTemps * numConc)]


    counter = 1

    for j in 1:numConc
        c_trial_raw = initial_concentrations[j]

        # Normalized version (preserve total concentration = 1.0)
        c_trial_norm = c_trial_raw ./ sum(c_trial_raw)

        for i in 1:numTemps
            T_trial = temperature_range[i]
            # Unnormalized trial
            trial_unnorm = vcat(c_trial_raw)
            initial_approximations[counter] = trial_unnorm
            counter += 1

            # Normalized trial
            # u_norm = U_EOS(T_trial, 1.0, c_trial_norm; model)
            # trial_norm = vcat(c_trial_norm, u_norm)
            trial_norm = vcat(c_trial_norm)
            initial_approximations[counter] = trial_norm
            
            counter += 1
        end
    end
    
    
    return initial_approximations
end

# x is vector of convcentrations, c_i = N_i / V, V is the volume of the phase p
function D(x; U_spec, V_spec, z_spec, model)
    _n = model.Nc

    result = 0.0
    u_trial = x[_n+1]
    c_trial = x[1:_n]

    T_spec = GetTemperatureForSpecifiedUVWithFD(; U=U_spec, V=V_spec, z=z_spec, model, T_guess=300.0)
    P_spec = P_EOS(T_spec, 1.0, z_spec ./ V_spec; model)
    μ_spec = μ_EOS(T_spec, V_spec, z_spec; model)

    T_trial = GetTemperatureForSpecifiedUVWithFD(; U=u_trial, V=1.0, z=c_trial, model, T_guess=300.0)
    P_trial = P_EOS(T_trial, 1.0, c_trial; model)
    μ_trial = μ_EOS(T_trial, 1.0, c_trial; model)

    # @show T_spec, T_trial, P_spec, P_trial, μ_spec, μ_trial
    result = (1.0 / T_trial - 1.0 / T_spec) * u_trial + (P_trial / T_trial - P_spec / T_spec)

    for i in 1:_n
        result -= (μ_trial[i] / T_trial - μ_spec[i] / T_spec) * c_trial[i]
    end

    return result
end

# x is vector of convcentrations, c_i = N_i / V, V is the volume of the phase p
function VT_D(x; T_spec, V_spec, z_spec, model)
    _n = model.Nc

    result = 0.0    
    c_trial = x[1:_n]
    # T_spec = GetTemperatureForSpecifiedUVWithFD(; U=U_spec, V=V_spec, z=z_spec, model, T_guess=300.0)
    P_spec = P_EOS(T_spec, 1.0, z_spec ./ V_spec; model)
    μ_spec = μ_EOS(T_spec, V_spec, z_spec; model)

    # Temperature is given for VT stability analysis
    T_trial = T_spec #GetTemperatureForSpecifiedUVWithFD(; U=u_trial, V=1.0, z=c_trial, model, T_guess=300.0)
    P_trial = P_EOS(T_trial, 1.0, c_trial; model)
    μ_trial = μ_EOS(T_trial, 1.0, c_trial; model)

    # log_fug = log_fugacity(T_spec, 1.0, c_trial; model) .- log_fugacity(T_spec, 1.0, z_spec ./ V_spec; model)
    
    # We need the pressure  term for the VT stability analysis to give the correct result
    result = (P_trial - P_spec) / ( T_spec)

    for i in 1:_n
        result -= (μ_trial[i] / T_trial - μ_spec[i] / T_spec) * c_trial[i]
        # result -= (log_fug[i]) * c_trial[i]
    end

    return result
end


@exportAll

end