module StabilityGPU

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

# c is the vector of concentrations, c_i = N_i / V, V is the volume of the phase p
function stability_gpu(c::MVector; T_spec, V_spec, N_spec::MVector, model)
    
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

function generate_perturbed_guesses_gpu(c_spec, k; T_spec, model, noise_level=0.1, rng_seed=nothing)
    n = length(c_spec)
    
    # Pre-allocate results on GPU - we need 2*k guesses (both unnormalized and normalized)
    total_guesses = 2 * k
    guesses_gpu = CUDA.zeros(Float64, n, total_guesses)
    
    # Generate random noise with controlled seed
    if rng_seed !== nothing
        Random.seed!(rng_seed)
    end
    
    # Generate noise on CPU (matching what CPU version would generate)
    noise_cpu = noise_level * randn(n, k)
    noise_gpu = cu(noise_cpu)  # Transfer to GPU
    # @show noise_cpu
    # Generate random noise directly on GPU
    # noise_gpu = noise_level * CUDA.randn(Float64, n, k)
    
    # Calculate total of original spec on GPU
    total_spec = sum(c_spec)
    
    # Launch kernel to generate perturbed compositions
    threads = 256
    blocks = cld(k, threads)
    c_spec_gpu = CUDA.cu(c_spec)
    @cuda threads=threads blocks=blocks generate_perturbed_kernel!(
        guesses_gpu, c_spec_gpu, noise_gpu, n, k, noise_level, total_spec
    )
    
    CUDA.synchronize()
    
    return guesses_gpu  # Returns GPU array of shape (n, 2*k)
end

function generate_perturbed_kernel!(guesses, c_spec, noise, n, k, noise_level, total_spec)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= k
        # Calculate unnormalized perturbed composition
        local_sum = 0.0
        for i in 1:n
            # Add random Gaussian noise and ensure positivity
            perturbed_val = c_spec[i] * (1.0 + noise[i, idx])
            perturbed_val = max(perturbed_val, 1e-12)
            guesses[i, 2*idx - 1] = perturbed_val  # Unnormalized goes to odd indices
            local_sum += perturbed_val
        end
        
        # Normalize to preserve total concentration (L∞ stable)
        normalization_factor = total_spec / local_sum
        normalized_sum = 0.0
        
        for i in 1:n
            guesses[i, 2*idx - 1] *= normalization_factor
            normalized_sum += guesses[i, 2*idx - 1]
        end
        
        # Calculate normalized version (sum to 1)
        for i in 1:n
            guesses[i, 2*idx] = guesses[i, 2*idx - 1] / normalized_sum
        end
    end
    return nothing
end

function generate_all_initial_approximations_gpu(T_spec, V_spec, z_spec_gpu, model, rng_seed=nothing)
    # Compute c_spec on GPU
    c_spec_gpu = compute_c_spec_gpu(V_spec, z_spec_gpu)
    
    # Generate all approximations on GPU
    simplex_gpu = generate_smejkal_simplex_based_approximations_gpu(; T_spec, model)
    saturation_gpu = initialize_phase_stability_gpu(T_spec, V_spec, z_spec_gpu; model)
    perturbed_gpu = generate_perturbed_guesses_gpu(Array(c_spec_gpu), 10; T_spec, model, noise_level=0.1, rng_seed)
    # println("simplex_gpu size: ", size(simplex_gpu), " saturation_gpu size: ", size(saturation_gpu), " perturbed_gpu size: ", size(perturbed_gpu))
    
    all_approximations_gpu = hcat(simplex_gpu, saturation_gpu, perturbed_gpu)
    
    return all_approximations_gpu  # Returns GPU array of shape (n, total_approximations)
end

function is_trivial_solution_gpu(c_gpu, c_spec_gpu; tol=1e-4)
    # Convert to CPU for computation
    c_cpu = Array(c_gpu)
    c_spec_cpu = Array(c_spec_gpu)
    
    rel_error = norm(c_cpu .- c_spec_cpu, Inf) / norm(c_spec_cpu, Inf)
    return rel_error < tol
end

# GPU version of make_feasibility_check
function make_feasibility_check_gpu(model, ϵ=0.0)
    n = model.Nc
    bi = [b_i(; i=i, model) for i in 1:n]  # CPU computation
    bi_gpu = cu(bi)  # Transfer to GPU
    
    return x_gpu -> begin
        x_cpu = Array(x_gpu)
        return sum(bi .* x_cpu) <= 1.0 && all(x_cpu .>= ϵ) && !any(isnan.(x_cpu))
    end
end

# GPU version of process_trial_point with CPU stability call
function process_trial_point_gpu(c_trial_gpu, T_spec, V_spec, z_spec_gpu, model, digits, feasibility_check, c_spec_gpu)   
    # Convert all GPU arrays to CPU for stability computation
    c_trial_cpu = Array(c_trial_gpu)
    z_spec_cpu = Array(z_spec_gpu)
    c_spec_cpu = Array(c_spec_gpu)
    
    # Call stability on CPU (convert to MVector for CPU function)
    c_cpu = stability_gpu(MVector(c_trial_cpu...); T_spec, V_spec, N_spec=MVector(z_spec_cpu...), model)
    
    # Convert result back to GPU
    c_gpu = cu(Vector(c_cpu))
    
    # Compute properties using GPU functions where possible
    D_trial = VT_D_gpu(c_gpu; T_spec, V_spec, z_spec_gpu, model)
    
    trivial = is_trivial_solution_gpu(c_gpu, c_spec_gpu)
    feasible = feasibility_check(c_gpu)
    # @info "D_trial = $D_trial for composition $c_gpu, trivial=$trivial, feasible=$feasible"
    return (; c=c_gpu, D_trial, α = -100.0, trivial, feasible)
end

compute_c_spec_gpu(V_spec, z_spec) = z_spec ./ V_spec

function VT_stabilityAnalysis_gpu(; T_spec, V_spec, z_spec_gpu, model)
    # Compute c_spec on GPU
    c_spec_gpu = compute_c_spec_gpu(V_spec, z_spec_gpu)
    n = length(z_spec_gpu)
    digits = 2    

    # Generate initial approximations on GPU
    initial_approximations_gpu = generate_all_initial_approximations_gpu(T_spec, V_spec, z_spec_gpu, model)
    feasibility_check = make_feasibility_check_gpu(model, 1e-8)    
    isunstable = false
    
    c_sol_gpu = c_spec_gpu    
    bestD = -Inf    
    first_counter = -1
    counter = 1
    
    num_approximations = size(initial_approximations_gpu, 2)
    
    println("Processing $num_approximations trial points on GPU with CPU stability calls...")
    
    # Process each trial point
    for i in 1:num_approximations
        c_trial_gpu = @view initial_approximations_gpu[:, i]
        
        # Process trial point (stability called on CPU internally)
        trial = process_trial_point_gpu(c_trial_gpu, T_spec, V_spec, z_spec_gpu, model, digits, feasibility_check, c_spec_gpu)
        
        D_trial = trial.D_trial
        
        # Check for valid unstable solution
        if !isnan(D_trial) && !trial.trivial && trial.feasible && D_trial >= 0
            isunstable = true
 
            first_counter == -1 && (first_counter = counter)
                        
            if bestD < D_trial
                bestD = D_trial
                c_sol_gpu = trial.c
            end
        end

        counter += 1
        
        # Progress reporting for large numbers of approximations
        if num_approximations > 50 && i % 10 == 0
            println("  Processed $i/$num_approximations trial points...")
        end
    end

    result = (; T_trial=T_spec, D_trial=bestD, isunstable, c_sol=Array(c_sol_gpu), c_spec=Array(c_spec_gpu), iterations=first_counter)
    
    return result
end

function initialize_density_guess_gpu(T, x_gpu, Pini, phase; model)
    # Convert GPU arrays to CPU for EOS calls
    x_cpu = Array(x_gpu)
    _n = length(x_cpu)
    
    n_total = 1.0          # Assume normalized phase mole number
    N_trial = x_cpu .* n_total
    
    # CPU call for volume roots
    V_roots = CubicFuncs.get_PR_volume(Pini, T, N_trial; model=model, phase)
    
    if isempty(V_roots)
        @warn "No valid volume roots found for trial composition"
        return CUDA.zeros(Float64, _n, 0)  # Empty GPU array
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

    d0_cpu = x_cpu ./ V_best
    # u0 = EOS.U_EOS(T, V_best, N_trial; model=model)
    
    # Convert back to GPU
    d0_gpu = cu(d0_cpu)
    
    return d0_gpu
end

function initialize_phase_stability_gpu(T_spec, Vspec, N0_gpu; model)
    # Convert GPU arrays to CPU for calculations
    N0_cpu = Array(N0_gpu)
    
    ∑ = sum
    z_cpu = N0_cpu ./ ∑(N0_cpu)                # Mole fractions
    n_total = ∑(N0_cpu)
    n_components = length(N0_cpu)

    guesses_gpu = CUDA.zeros(Float64, n_components, 0)  # Start with empty GPU array
    
    P_c = model.P_c
    T_c = model.T_c
    ω = model.ω
    compute_K(T_spec, P_spec) = log.(P_c ./ P_spec) .+ (5.373 * (1 .+ ω) .* (1 .- T_c ./ T_spec))
    
    #### 1. Try to compute initial pressure from EoS (explicit or numerical)
    try
        P0 = EOS.press(T_spec, Vspec, N0_cpu; model=model)
        if P0 > 0
            # Use K-values at (T_spec, P0)
            lnK0 = compute_K(T_spec, P0)
            K0 = exp.(lnK0)
            
            # Type L: vapor-like trial phase
            xV0_cpu = z_cpu .* K0
            xV0_cpu ./= ∑(xV0_cpu)
            xV0_gpu = cu(xV0_cpu)
            guess1 = initialize_density_guess_gpu(T_spec, xV0_gpu, P0, :liquid; model)
            
            # Type V: liquid-like trial phase
            xL0_cpu = z_cpu ./ K0
            xL0_cpu ./= ∑(xL0_cpu)
            xL0_gpu = cu(xL0_cpu)
            guess2 = initialize_density_guess_gpu(T_spec, xL0_gpu, P0, :vapor; model)
            
            # Combine guesses
            if !isempty(guess1) && !isempty(guess2)
                guesses_gpu = hcat(guesses_gpu, guess1, guess2)
            elseif !isempty(guess1)
                guesses_gpu = hcat(guesses_gpu, guess1)
            elseif !isempty(guess2)
                guesses_gpu = hcat(guesses_gpu, guess2)
            end
        end
    catch e
        @warn "Failed to compute P0 from EoS. Falling back to Wilson estimates. Error: $e"
    end

    #### 2. If P0 ≤ 0 or failed, use Wilson correlation
    # Wilson's equation
    Psat = similar(z_cpu)
    for i in 1:n_components
        Psat[i] = P_c[i] * exp(5.373 * (1 + ω[i]) * (1 - T_c[i] / T_spec))
    end
    
    # Type L (vapor trial)
    Pini_L = ∑(z_cpu .* Psat)
    xV0_cpu = z_cpu .* Psat
    xV0_cpu ./= ∑(xV0_cpu)
    xV0_gpu = cu(xV0_cpu)
    guess3 = initialize_density_guess_gpu(T_spec, xV0_gpu, Pini_L, :liquid; model)
    
    # Type V (liquid trial)
    Pini_V = 1 / ∑(z_cpu ./ Psat)
    xL0_cpu = z_cpu ./ Psat
    xL0_cpu ./= ∑(xL0_cpu)
    xL0_gpu = cu(xL0_cpu)
    guess4 = initialize_density_guess_gpu(T_spec, xL0_gpu, Pini_V, :vapor; model)
    
    # Combine all guesses
    for guess in [guess3, guess4]
        if !isempty(guess)
            guesses_gpu = hcat(guesses_gpu, guess)
        end
    end
    
    return guesses_gpu  # Returns GPU array of shape (n_components, num_guesses)
end

function coverSimplex_gpu(; model)
    _n = model.Nc
    
    # Pre-allocate all arrays on GPU - we need _n + 2 points
    X_gpu = CUDA.zeros(Float64, _n, _n + 2)  # Correct: _n × (_n + 2)
    
    # Calculate bi on CPU first 
    bi_cpu = [b_i(; i=i, model) for i in 1:_n]
    bi_gpu = cu(bi_cpu)  # Transfer to GPU
    
    # Barycenter on GPU
    barycenter = CUDA.zeros(Float64, _n)
    
    # Launch kernel to compute barycenter
    threads = 256
    blocks = cld(_n, threads)
    @cuda threads=threads blocks=blocks compute_barycenter_kernel!(barycenter, bi_gpu, _n)
    
    # Simplex vertices V (shape: _n × (_n + 1))
    V_gpu = CUDA.zeros(Float64, _n, _n + 1)
    
    # Compute vertices
    @cuda threads=threads blocks=blocks compute_vertices_kernel!(V_gpu, bi_gpu, _n)
    
    # Compute approximations X
    total_points = _n + 2
    @cuda threads=threads blocks=blocks compute_approximations_kernel!(
        X_gpu, V_gpu, barycenter, _n, total_points
    )
    
    CUDA.synchronize()
    return X_gpu  # Returns matrix of shape _n × (_n + 2)
end

function compute_barycenter_kernel!(barycenter, bi, _n)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= _n
        barycenter[idx] = 1.0 / ((_n + 1) * bi[idx])
    end
    return nothing
end

function compute_vertices_kernel!(V, bi, _n)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total_indices = _n * (_n + 1)
    
    if idx <= total_indices
        i = ((idx - 1) % _n) + 1  # coordinate index (0-based to 1-based)
        j = (idx - 1) ÷ _n + 1    # vertex index (1-based)
        
        if j <= _n  # First _n vertices are unit vectors scaled by 1/bi[i]
            if i == j
                V[i, j] = 1.0 / bi[i]
            else
                V[i, j] = 0.0
            end
        else  
            # Vertex _n+1 is all zeros (already initialized)
            V[i, j] = 0.0
        end
    end
    return nothing
end

function compute_approximations_kernel!(X, V, barycenter, _n, total_points)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total_indices = _n * total_points
    
    if idx <= total_indices
        i = ((idx - 1) % _n) + 1  # coordinate index  
        j = (idx - 1) ÷ _n + 1    # point index
        
        if j == 1
            # First point is barycenter
            X[i, j] = barycenter[i]
        elseif j <= _n + 1
            # Points 2 to _n+1 are 0.5*(V[j-1] + barycenter)
            X[i, j] = 0.5 * (V[i, j-1] + barycenter[i])
        else
            # Point _n+2 is 0.5*(V[_n+1] + barycenter) = 0.5*barycenter
            # Since V[_n+1] is all zeros
            X[i, j] = 0.5 * barycenter[i]
        end
    end
    return nothing
end

function generate_smejkal_simplex_based_approximations_gpu(; T_spec, model)
    _n = model.Nc
    
    # Use our GPU coverSimplex
    initial_concentrations_gpu = coverSimplex_gpu(; model)
    numConc = _n + 2
    total_guesses = 2 * numConc
    
    # Step 1: Generate all compositions on GPU (without energies)
    # We'll create just the compositions first, matching CPU structure
    compositions_gpu = CUDA.zeros(Float64, _n, total_guesses)
    initial_approximations = CUDA.zeros(Float64, _n, total_guesses)
    threads = 256
    blocks = cld(total_guesses, threads)
    
    @cuda threads=threads blocks=blocks generate_compositions_kernel!(
        initial_approximations, initial_concentrations_gpu, _n, numConc, total_guesses
    )
    
    CUDA.synchronize()
    
    # Step 2: Convert to CPU and create the final structure matching CPU version
    # compositions_cpu = Array(compositions_gpu)
    
    # Create the same structure as CPU version: Vector{Vector{Float64}}
    # initial_approximations = Vector{Vector{Float64}}(undef, total_guesses)
    

    # for idx in 1:total_guesses
    #     # Extract composition for this guess
    #     composition = compositions_cpu[:, idx]
        
    #     # Create the vector (composition only, matching CPU structure)
    #     initial_approximations[idx] = copy(composition)
    # end
    
    return initial_approximations
end

function generate_compositions_kernel!(compositions, initial_concentrations, 
                                     _n, numConc, total_guesses)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    
    if idx <= total_guesses
        conc_idx = (idx - 1) ÷ 2 + 1
        is_normalized = (idx % 2) == 0  # Even indices are normalized
        
        if is_normalized
            # Normalized version - calculate sum and normalize
            total = 0.0
            for i in 1:_n
                total += initial_concentrations[i, conc_idx]
            end
            
            for i in 1:_n
                compositions[i, idx] = initial_concentrations[i, conc_idx] / total
            end
        else
            # Unnormalized version - copy directly
            for i in 1:_n
                compositions[i, idx] = initial_concentrations[i, conc_idx]
            end
        end
    end
    return nothing
end

function VT_D_gpu(x_gpu; T_spec, V_spec, z_spec_gpu, model)
    # Convert to CPU for EOS calls
    x_cpu = Array(x_gpu)
    z_spec_cpu = Array(z_spec_gpu)
    
    _n = model.Nc
    result = 0.0    
    c_trial = x_cpu[1:_n]
    
    # CPU EOS calls
    P_spec = P_EOS(T_spec, 1.0, z_spec_cpu ./ V_spec; model)
    μ_spec = μ_EOS(T_spec, V_spec, z_spec_cpu; model)

    T_trial = T_spec
    P_trial = P_EOS(T_trial, 1.0, c_trial; model)
    μ_trial = μ_EOS(T_trial, 1.0, c_trial; model)

    result = (P_trial - P_spec) / T_spec

    for i in 1:_n
        result -= (μ_trial[i] / T_trial - μ_spec[i] / T_spec) * c_trial[i]
    end

    return result
end

# GPU version of isFeasible
function isFeasible_gpu(x_gpu, increment=zero(Array(x_gpu)); model, verbose=false)
    # Convert to CPU for computation
    x_cpu = Array(x_gpu)
    n = model.Nc
    
    x_new = x_cpu
    Nᴵ = x_new[1:n]
    Vᴵ = x_new[n+1]
    Nᴵᴵ = x_new[n+2:2n+1]
    Vᴵᴵ = x_new[2n+2]
    _b_i = [b_i(; i, model) for i = 1:n]  # CPU computation

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

function InitialGuessFromStabilityResult_gpu(x_gpu, T_trial; U_spec, V_spec, N_spec_gpu, model, verbose=true)
    # Convert GPU arrays to CPU for EOS calls
    x_cpu = Array(x_gpu)
    N_spec_cpu = Array(N_spec_gpu)
    
    n = model.Nc
    result = zeros(eltype(x_cpu), 2 * n + 2)  # removed 2 energy entries

    cPrime = x_cpu[1:n]

    bi = [b_i(; i=i, model) for i in 1:n]  # CPU computation
    smallest_volume = 1.01 * maximum(bi)

    V_trial = 0.5 * V_spec
    TPrime = T_trial  # CPU call
    moleTrial = cPrime .* V_trial

    SOne = S_EOS(T_trial, V_spec, N_spec_cpu; model)  # CPU call
    diff = 0.0
    iters = 200

    lambda = 1.0
    while iters > 0
        S_trial = S_EOS(T_trial, V_trial, moleTrial; model)  # CPU call

        U_trial = U_EOS(T_trial, V_trial, moleTrial; model)  # CPU call
        U_bulk = U_spec - U_trial
        V_bulk = V_spec - V_trial
        N_bulk = N_spec_cpu .- moleTrial
        T_bulk = GetTemperatureForSpecifiedUV(; U=U_bulk, V=V_bulk, z=N_bulk, model, T_guess=300.0)  # CPU call
        S_bulk = S_EOS(T_bulk, V_bulk, N_bulk; model)  # CPU call

        STwo = S_trial + S_bulk

        diff = STwo - SOne

        result[1:n] = moleTrial        
        result[n+1] = V_trial
        result[n+2:2n+1] = N_spec_cpu .- moleTrial
        result[2n+2] = V_spec - V_trial

        iters -= 1

        feasible = isFeasible_gpu(cu(result); model, verbose)
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
        isfeasible = isFeasible_gpu(cu(result); model, verbose)
        if isapproximately_zero && isfeasible
            return cu(result[n+2:end])  # Return GPU array
        else
            verbose && println("Failed to find feasible solution.")
            return "Failed to find feasible solution."
        end
    end

    return cu(result[n+2:end])  # Return GPU array
end

function IG3_gpu(c_gpu, T, U_spec, V_spec, z_spec_gpu; model, factor=10, verbose=false)    
    res = InitialGuessFromStabilityResult_gpu(c_gpu, T; U_spec, V_spec, N_spec_gpu=z_spec_gpu, model, verbose)
    return res
end

end