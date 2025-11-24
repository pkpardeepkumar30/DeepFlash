module Stability

using LinearAlgebra
using ForwardDiff
using ExportAll
# using FiniteDiff
using NLsolve
using StaticArrays
using ..EOS
using ..CubicFuncs
# using ..Solvers
using Random
using FixedPointAcceleration
using SHA

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
    # @show c, T_spec, V_spec, N_spec
    c_spec = N_spec ./ V_spec
    μ_ref = μ_EOS(T_spec, 1.0, c_spec; model)
    function create_prob(y::MVector)
        x = exp.(y)        
        Δμs = μ_EOS(T_spec, 1.0, x; model) - μ_ref
        # chem_pot_diff = chem_pot(T_spec, 1.0, x; model) .- chem_pot(T_spec, 1.0, c_spec; model)
    end
    sol = nlsolve(create_prob, log.(c), xtol=1e-8, ftol=1e-8, method=:newton, linesearch=LineSearches.BackTracking(order=3))
    # sol = nlsolve(create_prob, c, xtol=1e-8, ftol=1e-8, method=:newton, linesearch=LineSearches.BackTracking(order=3))
    # MVector(exp.(sol.zero)...)
    exp.(sol.zero)
    # sol.zero
end


# c is the vector of concentrations, c_i = N_i / V, V is the volume of the phase p
function stability2(c; T_spec, V_spec, N_spec, model)

    function create_prob(x)
        # x = exp.(y)
        # @show x
        log_fugacity(T_spec, 1.0, x; model) - log_fugacity(T_spec, 1.0, N_spec ./ V_spec; model)
    end

    sol = nlsolve(create_prob, c, xtol=1e-8, ftol=1e-8, method=:newton, linesearch=LineSearches.BackTracking(order=3))
    sol.zero
    # exp.(sol.zero)

end

function stability_test_uv(c; U_spec, V_spec, N_spec, model)

    cz = c_spec = N_spec ./ V_spec

    function prob(x)
        ForwardDiff.gradient(x -> D(x; U_spec, V_spec, z_spec=N_spec, model), x)
    end
    sol = nlsolve(prob, c, xtol=1e-8, ftol=1e-8, method=:newton, linesearch=LineSearches.BackTracking(order=3))
    # @show sol.zero
    # uvn.solvebyNew(-hess, g(x); PreCondition=true)
    sol.zero[1:end-1]
end

function stability_test(c; T_spec, V_spec, N_spec, model)

    cz = c_spec = N_spec ./ V_spec

    function prob(x)
        cz .* exp.(ln_volume_function(T_spec, 1.0, x; model)) ./ exp.(ln_volume_function(T_spec, 1.0, cz; model)) .- x
    end
    # picard(prob, c)
    # @show c
    sol = nlsolve(prob, c, xtol=1e-8, ftol=1e-8, method=:newton, linesearch=LineSearches.BackTracking(order=3))

    # uvn.solvebyNew(-hess, g(x); PreCondition=true)
    sol.zero
end

function stability_test_log(c; T_spec, V_spec, N_spec, model)

    cz = c_spec = N_spec ./ V_spec

    function prob(y)
        # make sure that x is positive
        x = exp.(y)
        
        # log.(cz ./ x) .+ ln_volume_function(T_spec, 1.0, x; model) .- ln_volume_function(T_spec, 1.0, cz; model)
        log.(x ./ cz) .+ ln_volume_function(T_spec, 1.0, cz; model) .- ln_volume_function(T_spec, 1.0, x; model)
    end
    # sol = picard(prob, log.(c))
    # exp.(sol)
    sol = nlsolve(prob, log.(c), xtol=1e-8, ftol=1e-8)
    exp.(sol.zero)
end

function generate_perturbed_guesses(c_spec, k; T_spec, model, noise_level=0.1)
    n = length(c_spec)
    guesses = []
    for _ in 1:k
        # Add random Gaussian noise and ensure positivity
        
        # Normalize to preserve total concentration (L∞ stable)
        # c_norm = c * (sum(c_spec) / sum(c))
        c_perturbed = max.(c_spec .* (1 .+ noise_level * randn(n)), 1e-12)
        c_perturbed = c_perturbed * (sum(c_spec) / sum(c_perturbed))
        push!(guesses, c_perturbed)
        c_perturbed = c_perturbed ./ sum(c_perturbed)
        U_trial = U_EOS(T_spec, 1.0, c_perturbed; model)
        # push!(guesses, vcat(c_perturbed, U_trial))
        push!(guesses, c_perturbed)
    end

    return guesses
end

function is_trivial_solution(c, c_spec; tol=1e-4)
    rel_error = norm(c .- c_spec, Inf) / norm(c_spec, Inf)
    return rel_error < tol
end

function stabilityAnalysis(; U_spec, V_spec, z_spec, model, stability_cache=nothing)
    # @show U_spec, V_spec, z_spec
    T_spec = GetTemperatureForSpecifiedUV(; U=U_spec, V=V_spec, z=z_spec, model, T_guess=300.0)
    c_spec = z_spec ./ V_spec

    # This function generates approximation and their normalised versions. It is important for avoiding trivial solutions.
    simplex_based_initial_approximations = generate_smejkal_simplex_based_approximations(; T_spec, model)
    saturation_based_approximations = initialize_phase_stability(T_spec, V_spec, z_spec; model)    
    gaussian_noise_based_approximations = generate_perturbed_guesses(c_spec, 10; T_spec, model,  noise_level=15)
    initial_approximations = vcat(simplex_based_initial_approximations, saturation_based_approximations, gaussian_noise_based_approximations)
    
    n = model.Nc
    # c_perturbed = c_spec .* (1 .+ 0.1 * randn(n))
    # c_perturbed = c_perturbed / sum(c_perturbed)
    
    # push!(more_approximations, vcat([146.0, 736.0], U_spec))
    # push!(more_approximations, vcat([140.0, 750.0], U_spec))
    # push!(more_approximations, vcat([150.0, 720.0], U_spec))
    # for approximation in initial_approximations
    #     println("Approximation: ", approximation)
    # end
    # println("initial_approximations: ", initial_approximations)
    T_trial = T_spec
    V_trial = 1.0
    counter = 1
    first_counter = -1
    # c_7 = [0.274, 2.305, 0.656, 0.249, 0.184, 0.0021, 0.0797]
    Ds = []
    cs = []
    c_2 = [146.1705,  736.4988]
    c_sol = nothing
    n = model.Nc
    isunstable = false
    # @show D(vcat(c_2, U_EOS(T_trial, 1.0, c_2; model)) ; U_spec, V_spec, z_spec, model)
    D_trial = -Inf
    bi = [b_i(; i=i, model) for i in 1:n]
    feasibility_check(x) = sum(bi .* x) <= 1.0 && all(x .>= 0)
    for x in initial_approximations
        try
            c_trial = x[1:end]
            c = nothing
            try
                c = stability(c_trial; T_spec=T_spec, V_spec=V_spec, N_spec=z_spec, model)

                U_trial = U_EOS(T_trial, 1.0, c; model)
                D_trial = D(vcat(c, U_trial); U_spec, V_spec, z_spec, model=model)
                # isvalid = feasibility_check(c) 
                if is_trivial_solution(c, c_spec)
                    # @info "Skipping trivial solution or convergence failure"
                    throw(ErrorException("Trivial solution detected"))
                end
            catch e
                # @info "Skipping trivial solution or convergence failure"
                continue
            end
            # U_trial = U_EOS(T_trial, 1.0, c; model)
            # D_trial = D(vcat(c, U_trial); U_spec, V_spec, z_spec, model=model)
            # S_Trial = S_EOS(T_trial, 1.0, c; model)
          
            # if any abs(c) is less than 1e-12, then set it to zero
            c = [abs(c[i]) < 1e-8 ? eps(c[i]) : c[i] for i in 1:n]            
            D_trial = abs(D_trial) < 1e-8 ? 0.0 : D_trial

            # if isapprox(D_trial, 0.0; atol=1e-6) && all(c .> 0)
            # @show sum(bi .* c) && all(x .>= 0) 
            if D_trial >= 0 && feasibility_check(c) 
                isunstable = true
                # println("Found unstable solution with D_trial: ", D_trial, " and c: ", c)
                # println("D_trial: ", D_trial, "c': ", c)
                push!(cs, c)
                append!(Ds, D_trial)
                # println("c': ", c)
                c_sol = c
                if first_counter == -1
                    first_counter = counter
                end
                break
            end
        catch e
            @show e
            println("Error: ", e)
        end
        counter += 1
    end
    # println("Number of initial approximations used: ", counter)
    # display(c_sol)
    # @show isunstable
    if isunstable
        bestD = maximum(Ds)
        argmaxD = argmax(Ds)
        c_sol = cs[argmaxD]
        return (; T_trial, D_trial, isunstable, c_sol, c=c_sol, bestD, c_spec, cs, Ds, iterations=first_counter)
    else        
        return (; T_trial, D_trial, isunstable, Ds, c_spec)
    end

end

function is_approximately_equal(a, b; atol=1e-1)
    return length(a) == length(b) && all(abs.(a .- b) .< atol)
end

function array_hash_key(arr::Vector{Float64}; digits::Int=3)
    rounded = round.(arr; digits=digits)
    return bytes2hex(sha1(reinterpret(UInt8, rounded)))
end

function array_hash_key_efficient(arr::Vector{Float64}; digits::Int=3)
    scale = 10.0^digits
    int_arr = round.(Int64, arr .* scale)
    key = mod(hash(int_arr), 10^8)
    return key
end

# Good cache (lookup result)
function lookup_good_trial(cache, arr::Vector{Float64}; digits=3)
    key = array_hash_key(arr; digits=digits)
    return get(cache.good_trials, key, nothing)
end

function store_good_trial!(cache, TVNC_TrialKey::Vector{Float64}, c_good::Vector{Float64}; digits=3)
    cache.good_trials[array_hash_key(TVNC_TrialKey; digits=digits)] = c_good
end

function store_stability_result!(cache, TVNKey::Vector{Float64}, res; digits=3)
    cache.results[array_hash_key(TVNKey; digits=digits)] = res
end

# Bad cache
function is_in_cache(cache::Set{String}, arr::Vector{Float64}; digits=3)
    return array_hash_key(arr; digits=digits) in cache
end

function add_to_cache(cache::Set{String}, arr::Vector{Float64}; digits=3)
    push!(cache, array_hash_key(arr; digits=digits))
end

function is_near_bad_trial(trial::Vector{Float64}, bad_list::Vector{Vector{Float64}}; tol::Float64=50.0)
    any(bad -> norm(trial .- bad) < tol, bad_list)
end

function add_bad_trial!(bad_list::Vector{Vector{Float64}}, trial::Vector{Float64})
    push!(bad_list, trial)
end

# Helper function to compute c_spec
compute_c_spec(V_spec, z_spec) = z_spec ./ V_spec

# Generate all initial approximations
function generate_all_initial_approximations(T_spec, V_spec, z_spec, model)
    c_spec = compute_c_spec(V_spec, z_spec)
    # d = length(c_spec)
    # tpd(x) = begin
    #     V = x[1]
    #     N = x[2:end]
    #     c = N ./ V
    #     VT_D(c; T_spec, V_spec, z_spec, model)
    # end
    # scale = vcat(V_spec, z_spec)
    # points, d_vals = sample_simplex_condition(1+d, 20, tpd; scale)
    # point_vecs = [points[:, i] for i in 1:size(points, 2)]
    # return point_vecs
    
    simplex = generate_smejkal_simplex_based_approximations(; T_spec, model)    
    saturation = initialize_phase_stability(T_spec, V_spec, z_spec; model)
    perturbed = generate_perturbed_guesses(c_spec, 10; T_spec, model, noise_level=0.1)
    # @show perturbed
    # error("Initial approximations should be a matrix with each column as a point in the simplex")
    return vcat(simplex, saturation, perturbed)
end

# Build trial combination key for caching
build_trial_key(T_spec, V_spec, z_spec, c_trial) = vcat(T_spec, V_spec, z_spec, c_trial)

# Create feasibility check function
function make_feasibility_check(model, ϵ=0.0)
    n = model.Nc
    bi = [b_i(; i=i, model) for i in 1:n]
    return x -> sum(bi .* x) <= 1.0 && all(x .>= ϵ)
end


function process_trial_point(c_trial::MVector, T_spec, V_spec, z_spec::MVector, model, stability_cache, digits, feasibility_check, c_spec::MVector)
    tried_combination = build_trial_key(T_spec, V_spec, z_spec, c_trial)
    use_cache = stability_cache !== nothing && stability_cache.use_cache

    # Check bad cache
    if use_cache && is_in_cache(stability_cache.cache, tried_combination; digits)
        return (status=:bad_cache, error_message = "Bad trial", c=nothing, D_trial=nothing, trivial=nothing, feasible=nothing)
    end

    # Check good cache
    cache_found = false
    cached_c = use_cache ? lookup_good_trial(stability_cache, tried_combination; digits) : nothing
    if cached_c !== nothing
        c = cached_c
        cache_found = true
    end
    α = -100.0
    # Compute stability if not cached
    if !cache_found
        try
            c = stability(MVector(c_trial...); T_spec, V_spec, N_spec=z_spec, model)
            
        catch e
            # println("Stability computation failed for trial point: $c_trial with error: ", e)
            # @warn "Stability computation failed for trial point: $c_trial" error=e
            use_cache && add_to_cache(stability_cache.cache, tried_combination; digits)
            use_cache && add_bad_trial!(stability_cache.bad_trials, tried_combination)
            return (status=:failed, error_message = e, c=nothing, D_trial=nothing, trivial=nothing, feasible=nothing, α =nothing)
        end
    end
    # @show "Processed trial point: $c with α = $α"
    # Compute D_trial and check solution properties
    U_trial = U_EOS(T_spec, 1.0, c; model)
    D_trial = VT_D(c; T_spec, V_spec, z_spec, model)
    trivial = is_trivial_solution(c, c_spec)
    feasible = feasibility_check(c)
    return (;status=:converged, error_message = nothing, c, D_trial, α = -100.0, trivial, feasible, cache_found, tried_combination)
end

# Post-process composition and D value
function postprocess_solution(c::MVector, D_trial)
    return c, D_trial
    
    n = length(c)
    c_processed = [abs(comp) < 1e-4 ? eps(comp) : comp for comp in c]
    D_processed = D_trial < 0 && abs(D_trial) < 1e-4 ? eps(abs(D_trial)) : D_trial
    return c_processed, D_processed
end

# Main stability analysis function
function VT_stabilityAnalysis(; T_spec, V_spec, z_spec::MVector, model, stability_cache=nothing)
    c_spec = compute_c_spec(V_spec, z_spec)
    n = model.Nc
    digits = 2
    use_cache = stability_cache !== nothing && stability_cache.use_cache ? true : false 
    
    # Check final result cache
    if use_cache
        final_key = array_hash_key(vcat(T_spec, V_spec, z_spec); digits)
        haskey(stability_cache.results, final_key) && return stability_cache.results[final_key]
    end

    # Generate initial approximations and setup
    initial_approximations = generate_all_initial_approximations(T_spec, V_spec, z_spec, model)

    feasibility_check = make_feasibility_check(model, 1e-8)
    DSinglePhase = Any[]
    isunstable = false
    Ds = Any[]
    cs = Any[]
    αs = Any[]
    first_counter = -1
    counter = 1
    α = -100.0
    # Process each trial point
    for x in initial_approximations
        c_trial = MVector(x[1:end]...)
        # N_trial = x[2:end]
        # V_trial = x[1]
        # c_trial = N_trial ./ V_trial
        # error("Processing trial point: $c_trial")
        trial_result = process_trial_point(c_trial, T_spec, V_spec, z_spec, model, stability_cache, digits, feasibility_check, MVector(c_spec...))
        
        # Handle bad cache points
        if trial_result.status == :bad_cache
            # println("Skipping known bad c_trial: ", c_trial)
            continue
        # Handle failed computations
        elseif trial_result.status == :failed
            # println("Newton failed to converge for c_trial: $c_trial, T_spec = $T_spec, V_spec = $V_spec, N_spec = $z_spec with error: ", trial_result.error_message)
            continue
        # Process converged results
        elseif trial_result.status == :converged
            push!(DSinglePhase, trial_result.D_trial)
            c, D_trial = postprocess_solution(MVector(trial_result.c...), trial_result.D_trial)
            
            # Check for valid unstable solution
            if !trial_result.trivial && trial_result.feasible && D_trial >= 0
                isunstable = true
                if use_cache && !trial_result.cache_found
                    store_good_trial!(stability_cache, trial_result.tried_combination, c; digits)
                end
                push!(cs, c)
                push!(Ds, D_trial)
                push!(αs, trial_result.α)
                first_counter == -1 && (first_counter = counter)
            end
        end
        counter += 1
    end

   
    # Prepare results
    result = if isunstable
        bestD, idx = findmax(Ds)
        c_sol = cs[idx]
        α_sol = αs[idx]
        (; T_trial=T_spec, D_trial=Ds[idx], isunstable, c_sol, c_spec, c=c_sol, 
         bestD, cs, Ds, α_sol, DSinglePhase, cSinglePhase=Float64[], iterations=first_counter)
    else
        bestD = -Inf
        (; T_trial=T_spec, D_trial=bestD, Ds, αs, α_sol = NaN, bestD, c_sol=nothing, c_spec, 
         cs, cSinglePhase=Float64[], DSinglePhase, isunstable)
    end

    # Cache final result
    if use_cache
        # error("Storing result in cache with key: ", stability_cache)
        TVN_key = array_hash_key(vcat(T_spec, V_spec, z_spec); digits)
        stability_cache.results[TVN_key] = result
    end

    return result
end

function VT_stabilityAnalysis_orig(; T_spec, V_spec, z_spec, model, stability_cache=nothing)
    
    c_spec = z_spec ./ V_spec
    
   # This function generates approximation and their normalised versions. It is important for avoiding trivial solutions.
    simplex_based_initial_approximations = generate_smejkal_simplex_based_approximations(; T_spec, model)
    saturation_based_approximations = initialize_phase_stability(T_spec, V_spec, z_spec; model)    
    gaussian_noise_based_approximations = generate_perturbed_guesses(c_spec, 10; T_spec, model,  noise_level=0.1)
    initial_approximations = vcat(simplex_based_initial_approximations, saturation_based_approximations, gaussian_noise_based_approximations)
    
    # for approximation in initial_approximations
    #     println("Approximation: ", approximation[1:2] ./ sum(approximation[1:2]))
    # end
    # println("initial_approximations: ", initial_approximations)
    T_trial = T_spec
    V_trial = 1.0
    counter = 1
    first_counter = -1
    # c_7 = [0.274, 2.305, 0.656, 0.249, 0.184, 0.0021, 0.0797]
    Ds = []
    DSinglePhase = []
    cSinglePhase = []
    cs = []
    # c_2 = [146.1705,  736.4988]
    c_sol = nothing
    n = model.Nc
    isunstable = false
    D_trial = -Inf
    bestD = -Inf
    ϵ = 0.0 #1e-8
    bi = [b_i(; i=i, model) for i in 1:n]
    feasibility_check(x) = sum(bi .* x) <= 1.0 && all(x .>= ϵ)
    for x in initial_approximations
        try
            c_trial = x[1:end]  

            # don't try bad c_trial if cache is provided
 
            c = nothing
            try
                c = stability(c_trial; T_spec=T_spec, V_spec=V_spec, N_spec=z_spec, model)

                U_trial = U_EOS(T_trial, 1.0, c; model)
                D_trial = VT_D(c; T_spec, V_spec, z_spec, model=model)
                
                if is_trivial_solution(c, c_spec)
                    # @info "Skipping trivial solution or convergence failure, c = $c, D_trial = $D_trial"
                    # throw(ErrorException("Trivial solution detected"))
                    continue
                end
                append!(DSinglePhase, D_trial)
                if stability_cache !== nothing
                    store_good_trial!(stability_cache, arr, c; digits=digits)
                end
            catch e
                println("Newton failed to converge for c_trial: $c_trial, T_spec = $T_spec, V_spec = $V_spec, N_spec = $z_spec with error: ", e)
                if stability_cache !== nothing
                    add_to_cache(stability_cache.cache, arr; digits=digits)
                    add_bad_trial!(stability_cache.bad_trials, arr)
                end
                continue
            end
           
            # if any abs(c) is less than 1e-12, then set it to zero
            c = [abs(c[i]) < 1e-4 ? eps(c[i]) : c[i] for i in 1:n]            
            
            # if 0 < D_trial < 1e-10
            #     continue
            # end
            # D_trial = abs(D_trial) < 1e-8 ? 0.0 : D_trial
            if D_trial < 0 
                D_trial = abs(D_trial) < 1e-4 ? eps(abs(D_trial)) : D_trial
            end
            
            if D_trial >= 0 && feasibility_check(c) 
                isunstable = true
                # println("D_trial: ", D_trial, "c': ", c)
                push!(cs, c)
                append!(Ds, D_trial)
                # println("c': ", c)
                c_sol = c
                if first_counter == -1
                    first_counter = counter
                end
                # break
            end
        catch e
            println("Error: ", e)
        end
        counter += 1
    end
    
    if isunstable 
        bestD = maximum(Ds)
        argmaxD = argmax(Ds)
        c_sol = cs[argmaxD]
        # if isapprox(c_sol, c_spec; atol = 1e-10)
        #     isunstable = false
        #     cs = c_sol = []
        #     Ds = [-Inf]
        #     D_trial = -Inf
        #     bestD = -Inf
        # else
            return (; T_trial, D_trial, isunstable, c_sol, c_spec, c=cs[argmaxD], bestD, cs, Ds, DSinglePhase, cSinglePhase, iterations=first_counter)
        # end
    end
    
    return (; T_trial, D_trial, Ds, bestD, c_sol, c_spec, cs, cSinglePhase, DSinglePhase, isunstable)

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
    try
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
    catch
        @warn "Failed to compute P0 from EoS. Falling back to Wilson estimates."
    end

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

#TODO:  This is only for two-phase specific stability analysis. To make it for general stability analysis, we need to change the result array so that its signature/layout matches the one accepted by Sols.jl which in turn passes it to the Flash.jl
# result = [c₁, c₂, ..., cₙ, V₁, U₁, c₁ᴵᴵ, c₂ᴵᴵ, ..., cₙᴵᴵ, V₂, U₂]
"""
    InitialGuessFromStabilityResult(x, T_trial; U_spec, V_spec, N_spec, model)
    

"""
function InitialGuessFromStabilityResult_orig(x, T_trial; U_spec, V_spec, N_spec, model, verbose=true)
    n = model.Nc
    result = zeros(Float64, 2 * n + 4)
    # println("Initial guess from stability result: ", x)
    cPrime = x[1:n]
    U_trial = U_EOS(T_trial, 1.0, cPrime; model)
    x[n+1] = U_trial

    bi = [b_i(; i=i, model) for i in 1:n]
    smallest_volume = 1.01*maximum(bi)
    # Helmholtz function-based global phase stability test and its link to the isothermal–isochoric flash problem, Castier(2014)
    VPrime = 0.5 * V_spec
    UPrime = U_trial * VPrime
    TPrime = GetTemperatureForSpecifiedUV(; U=U_spec, V=V_spec, z=N_spec, model, T_guess=300.0)
    # @show N_spec U_spec V_spec
    molePrime = cPrime .* VPrime

    # error()
    SOne = S_EOS(TPrime, V_spec, N_spec; model)
    diff = 0.0
    iters = 200

    # func(x) = D(x; U_spec, V_spec, z_spec = N_spec, model)
    # g(x) = FiniteDiff.finite_difference_gradient(func, x)
    # H(x) = FiniteDiff.finite_difference_hessian(func, x)

    lambda = 1.0
    while iters > 0 # VPrime / V_spec > sqrt(eps(VPrime / V_spec))
        S_trial = Property_UVN(UPrime, VPrime, molePrime; model, ThermoFunc=S_EOS)
        S_bulk = Property_UVN(U_spec - UPrime, V_spec - VPrime, N_spec .- molePrime; model, ThermoFunc=S_EOS)
        
        # S_trial2 = S_EOS(TPrime, VPrime, molePrime; model)
        # S_bulk2 = S_EOS(TPrime, V_spec - VPrime, N_spec .- molePrime; model)
        # @show S_trial, S_trial2, S_bulk, S_bulk2
        STwo = S_trial + S_bulk
        STwo = Property_UVN(UPrime, VPrime, molePrime; model, ThermoFunc=S_EOS) + Property_UVN(U_spec - UPrime, V_spec - VPrime, N_spec .- molePrime; model, ThermoFunc=S_EOS)
        # STwo = S_EOS(TPrime, VPrime, molePrime; model) +
        #          S_EOS(TPrime, V_spec - VPrime, N_spec .- molePrime; model)
        T1 = GetTemperatureForSpecifiedUV(; U=U_spec - UPrime, V=V_spec - VPrime, z=N_spec .- molePrime, model, T_guess=300.0)
        T2 = GetTemperatureForSpecifiedUV(; U=UPrime, V=VPrime, z=molePrime, model, T_guess=300.0)
        @show T1, T2
        diff = STwo - SOne
        result[1:n] = molePrime
        result[n+1] = VPrime
        result[n+2] = UPrime
        result[n+3:2*n+2] = N_spec .- molePrime
        result[2*n+3] = V_spec - VPrime
        result[2*n+4] = U_spec - UPrime

        iters -= 1
        # @show diff, VPrime, UPrime, STwo, SOne        
        # add incipient phase
        # Nᴵᴵ = N_spec .* 1e-6

        # @show increment[1:n]
        # @show increment[n+1:2*n]
        # increment = 1e-12 #abs.(randn(2 * n + 4)) * 1e-6
        feasible = isFeasible_orig(result; model, verbose)

        # @show feasible, VPrime
        verbose &&  @show STwo, SOne
        S_diff = STwo - SOne
        if S_diff > 0.0 && feasible
        # if S_diff > abs(SOne)*0.01 && feasible
            verbose && println("Feasible solution found.")
            verbose && @show feasible, VPrime, UPrime, STwo, SOne
            verbose && @show result
            # @show result[1:n]
            # @show result[n+3:2n+2]
            break
        end

        if (VPrime / V_spec) < 1e-8
            verbose && println("Failed to find feasible solution.")
            break
        end


        VPrime /= 2.0
        UPrime = U_trial * VPrime
        molePrime = cPrime .* VPrime

    end

    if diff < 0
        isapproximately_zero = isapprox(diff, 0.0; atol=1e-2)
        isfeasible = isFeasible_orig(result; model, verbose)
        if isapproximately_zero && isfeasible
            # result[end] = TPrime
            return result[n+3:end]
        else
            verbose && println("Failed to find feasible solution.")
            return "Failed to find feasible solution."
        end
        
    end

    # result[end] = TPrime
    return result[n+3:end]
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

function InitialGuessFromStabilityResult_Doubling(x, T_trial; U_spec, V_spec, N_spec, model, verbose=false)
    n = model.Nc
    result = zeros(Float64, 2 * n + 4)
    cPrime = x[1:n]
    U_trial = U_EOS(T_trial, 1.0, cPrime; model)
    x[n+1] = U_trial

    bi = [b_i(; i=i, model) for i in 1:n]
    println("bi: ", bi)
    VPrime = 1.1 * maximum(bi)
    # VPrime = 1e-6  # Start with tiny volume
    UPrime = x[n+1] * VPrime
    TPrime = GetTemperatureForSpecifiedUV(; U=U_spec, V=V_spec, z=N_spec, model, T_guess=300.0)
    molePrime = cPrime .* VPrime

    SOne = S_EOS(TPrime, V_spec, N_spec; model)
    diff = 0.0
    iters = 200

    lambda = 1.0
    while iters > 0
        STwo = Property_UVN(UPrime, VPrime, molePrime; model, ThermoFunc=S_EOS) + 
               Property_UVN(U_spec - UPrime, V_spec - VPrime, N_spec .- molePrime; model, ThermoFunc=S_EOS)

        diff = STwo - SOne
        result[1:n] = molePrime
        result[n+1] = VPrime
        result[n+2] = UPrime
        result[n+3:2*n+2] = N_spec .- molePrime
        result[2*n+3] = V_spec - VPrime
        result[2*n+4] = U_spec - UPrime

        iters -= 1
        feasible = isFeasible(result; model, verbose)
        # verbose && @show STwo, SOne
        S_diff = STwo - SOne

        if S_diff > 0.0 && feasible
            verbose && println("Feasible solution found in iteration: ", iters)
            break
        end

        if (VPrime / V_spec) > 0.5
            verbose && println("Failed to find feasible solution.")
            break
        end

        VPrime = 2.0 * VPrime  # DOUBLE the volume
        UPrime = x[n+1] * VPrime
        molePrime = cPrime .* VPrime
    end

    if diff < 0
        isapproximately_zero = isapprox(diff, 0.0; atol=1e-2)
        isfeasible = isFeasible(result; model, verbose)
        if isapproximately_zero && isfeasible
            # result[end] = TPrime
            return result[n+3:end]
        else
            verbose && println("Failed to find feasible solution.")
            return "Failed to find feasible solution."
        end
    end

    # result[end] = TPrime
    return result[n+3:end]
end


function IG(problem; model, factor=10)
    prob = problem()
    U_spec, V_spec, z_spec = prob.T.U, prob.T.V, prob.T.N
    sol = Stability.stabilityAnalysis(; model, U_spec, V_spec, z_spec)
    res = Stability.InitialGuessFromStabilityResult(vcat(sol.c, -1e6), sol.T_trial; U_spec, V_spec, N_spec=z_spec, model)

end

function IG2(c, T, problem; model, factor=10, verbose=false)
    prob = problem()
    U_spec, V_spec, z_spec = prob.T.U, prob.T.V, prob.T.N
    @show c, T
    res = Stability.InitialGuessFromStabilityResult(c, T; U_spec, V_spec, N_spec=z_spec, model, verbose)
end

function IG3(c, T, U_spec, V_spec, z_spec; model, factor=10, verbose=false)    
    res = Stability.InitialGuessFromStabilityResult(c, T; U_spec, V_spec, N_spec=z_spec, model, verbose)
end

# SSI
function picard(f, x0; λ=0.8)
    # g(x) = x .+ f(x)
    g(x) = f(x) .- x
    sol = fixed_point(g, x0; Dampening=λ)
    # @show sol
    fixedpoint = sol.FixedPoint_
    # println("Residue: ", norm(f(fixedpoint)))
    # @show fixedpoint
    fixedpoint
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
            # u_unnorm = U_EOS(T_trial, 1.0, c_trial_raw; model)
            # trial_unnorm = vcat(c_trial_raw, u_unnorm)
            trial_unnorm = vcat(c_trial_raw)
            # extract_qty(qty) = ForwardDiff.value.(ForwardDiff.value.(qty))
            # @show typeof(trial_unnorm) typeof(initial_approximations)
            initial_approximations[counter] = trial_unnorm
            counter += 1

            # Normalized trial
            u_norm = U_EOS(T_trial, 1.0, c_trial_norm; model)
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