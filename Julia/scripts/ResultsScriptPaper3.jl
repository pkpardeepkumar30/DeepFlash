if false
    include("../src/MultiComponent.jl")
    include("../src/PreFlashGPU.jl")
    include("../src/PreFlash.jl")
    using .MultiComponent
end

# cd("Pipelines/scripts/")
using Pkg
Pkg.activate("./Julia/")
using Revise  # Load Revise AFTER activating environment but BEFORE loading package

using CO2Transport
using CO2Transport.MultiComponent
# using Clapeyron

using NLsolve
using StaticArrays
using Printf
using Statistics
using CUDA
using Random
using InteractiveUtils   # for code_typed
using Cthulhu # for introspection

kwargs_1_4 = (;
    Mw = [16.04e-3, 34.083e-3],
    T_c = [190.4, 373.2],  # Critical temperatures in K
    P_c = [46.0e5, 89.4e5],  # Critical pressures in Pa    
    ρ_c = [10139.0, 10190.0],
    ω = [0.011, 0.081], #0.0810919       # Acentric factors
    # ω=[0.011, 0.008], #0.0810919       # Acentric factors
    δ = [0.0 0.083; 0.083 0.0],
    α = [
        19.25 5.213e-2 1.197e-5 -1.132e-8
        31.94 1.463e-3 2.432e-5 -1.176e-8
    ],
)

kwargs_56 = (
    Mw = [30.07e-3, 42.081e-3, 44.097e-3, 58.124e-3, 58.124e-3, 72.15e-3],  # Molecular weights [kg/mol]
    T_c = [305.4, 364.9, 369.8, 408.2, 425.2, 469.7],  # Critical temperatures in K
    P_c = [48.8e5, 46.0e5, 42.5e5, 36.5e5, 38.0e5, 33.7e5],  # Critical pressures in Pa
    ρ_c = [10139.0, 10190.0, 10030.0, 9995.0, 9810.0, 9650.0],  # Example critical densities (assumed values)
    ω = [0.099, 0.144, 0.153, 0.183, 0.199, 0.251],  # Acentric factors
    δ = zeros(6, 6),  # Interaction parameter matrix (assumed to be zero)
    α = [
        5.409 1.781e-1 -6.938e-5 8.713e-9
        3.710 2.345e-1 -1.160e-4 2.205e-8
        -4.224 3.063e-1 -1.586e-4 3.215e-8
        -1.390 3.847e-1 -1.846e-4 2.895e-8
        9.487 3.313e-1 -1.108e-4 -2.822e-9
        -3.626 4.873e-1 -2.580e-4 5.305e-8
    ],  # Correlation coefficients for Cp_ig
    # U = -16272506.4,  # Internal energy [J]
    # V = 479845.0e-6,  # Volume [m^3]
    # N = [10.8, 360.8, 146.5, 233.0, 233.0, 15.9]  # Molar quantities [mol]
)



model_1_4 = MultiComponent.EOS.PengRobinson(; kwargs_1_4..., doScale = true);
model_5_6 = MultiComponent.EOS.PengRobinson(; kwargs_56..., doScale = true);
# model_CO2 = MultiComponent.EOS.PengRobinson(; kwargs_CO2..., doScale = true);

function run_flash_calculations(prob, model)
    p1 = prob().T
    PreFlash.flash_calculation(p1.U, p1.V, MVector((p1.N)...); model)
end

function test_gpu_coverSimplex(model)
    println("Testing GPU coverSimplex...")
    
    # Test GPU version
    X_gpu = StabilityGPU.coverSimplex_gpu(; model)
    
    # Test CPU version for comparison
    X_cpu = Stability.coverSimplex(; model)
    
    # Convert GPU result to CPU for comparison
    X_gpu_cpu = Array(X_gpu)
    
    println("GPU result shape: ", size(X_gpu))
    println("CPU result length: ", length(X_cpu))
    
    # Compare results
    for i in 1:length(X_cpu)
        cpu_point = X_cpu[i]
        gpu_point = X_gpu_cpu[:, i]
        
        println("Point $i:")
        println("  CPU: ", cpu_point)
        println("  GPU: ", gpu_point)
        println("  Max difference: ", maximum(abs.(cpu_point - gpu_point)))
    end
    
    return X_gpu
end

function quick_test_smejkal(prob; model)
    println("Quick test for single system...")
    testcase = prob()
    N_spec = MVector((testcase.T.N)...)
    U_spec = testcase.T.U
    V_spec = testcase.T.V
    T_spec = PreFlash.compute_single_phase_state(U_spec, V_spec, N_spec, model)
    # GPU version
   println("Running GPU version...")
    approximations_gpu = StabilityGPU.generate_smejkal_simplex_based_approximations_gpu(; T_spec, model)
    
    # Test CPU version for comparison
    println("Running CPU version...")
    approximations_cpu = Stability.generate_smejkal_simplex_based_approximations(; T_spec, model)
    
    # Convert GPU result to CPU for comparison
    approximations_gpu_cpu = Array(approximations_gpu)
    
   _n = model.Nc
    numConc = _n + 2
    total_guesses = 2 * numConc
    
    println("\nResults Summary:")
    println("GPU result type: ", typeof(approximations_gpu))
    println("GPU result length: ", length(approximations_gpu))
    println("CPU result length: ", length(approximations_cpu))
    println("Expected total guesses: ", total_guesses)
    
    # Compare results
    max_diff = 0.0
    all_close = true
    
    println("\nDetailed Comparison:")
    for i in 1:total_guesses
        gpu_point = approximations_gpu[i]
        cpu_point = approximations_cpu[i]
        
        # Check dimensions
        if length(gpu_point) != length(cpu_point)
            println("ERROR: Dimension mismatch at guess $i")
            println("  GPU length: ", length(gpu_point))
            println("  CPU length: ", length(cpu_point))
            all_close = false
            continue
        end
        
        diff = maximum(abs.(gpu_point - cpu_point))
        max_diff = max(max_diff, diff)
        
        if i <= 4  # Print first few for inspection
            println("Guess $i:")
            println("  GPU: ", round.(gpu_point, digits=6))
            println("  CPU: ", round.(cpu_point, digits=6))
            println("  Difference: $diff")
        end
        
        if diff > 1e-3
            all_close = false
        end
    end
    
    println("\nTest Summary:")
    println("Maximum difference: ", max_diff)
    println("All points close: ", all_close)
    
    
    return approximations_gpu, approximations_cpu, all_close
end

function quick_test_saturation_based_guesses_gpu(prob; model)
    println("Quick test for saturation based stability initialization...")
    testcase = prob()
    z_spec = MVector((testcase.T.N)...)
    U_spec = testcase.T.U
    V_spec = testcase.T.V
    T_spec = PreFlash.compute_single_phase_state(U_spec, V_spec, z_spec, model)
    # Create test composition
    _n = model.Nc
    
    # GPU version
    guesses_gpu = StabilityGPU.initialize_phase_stability_gpu(T_spec, V_spec, z_spec; model)
    guesses_gpu_cpu = Array(guesses_gpu)
    
    # CPU version  
    guesses_cpu = Stability.initialize_phase_stability(T_spec, V_spec, z_spec; model)
    return guesses_cpu, guesses_gpu
    num_gpu = size(guesses_gpu, 2)
    num_cpu = length(guesses_cpu)
    
    println("GPU guesses: ", num_gpu)
    println("CPU guesses: ", num_cpu)
    
    # Quick comparison
    if num_gpu > 0 && num_cpu > 0
        min_guesses = min(num_gpu, num_cpu)
        println("\nFirst guess comparison:")
        gpu_first = guesses_gpu_cpu[:, 1]
        cpu_first = guesses_cpu[1]
        
        diff = maximum(abs.(gpu_first - cpu_first))
        status = diff > 1e-3 ? "❌" : "✅"
        println("$status First guess - Max difference: $diff")
        
        if diff > 1e-3
            println("  GPU: ", gpu_first)
            println("  CPU: ", cpu_first)
        end
    end
    
    return guesses_gpu
end

function quick_test_perturbed_guesses_gpu(prob; k, model, noise_level=0.1, rng_seed=12345)
    println("Quick test for perturbed guesses generation...")
    
    testcase = prob()
    z_spec = MVector((testcase.T.N)...)
    U_spec = testcase.T.U
    V_spec = testcase.T.V
    T_spec = PreFlash.compute_single_phase_state(U_spec, V_spec, z_spec, model)

    c_spec = z_spec ./ V_spec
    # Create test composition
    # Convert to GPU
    c_spec_gpu = CUDA.CuArray(c_spec)
    
    # GPU version
    guesses_gpu = StabilityGPU.generate_perturbed_guesses_gpu(c_spec_gpu, k; T_spec, model, noise_level, rng_seed)
    guesses_gpu_cpu = Array(guesses_gpu)
    
    # CPU version  
    Random.seed!(rng_seed)
    guesses_cpu = Stability.generate_perturbed_guesses_orig(c_spec, k; T_spec, model, noise_level)
    
    # return guesses_cpu, guesses_gpu
    println("GPU shape: ", size(guesses_gpu))
    println("CPU number of guesses: ", length(guesses_cpu))
    
    # Quick comparison of first guess
    if k > 0
        gpu_first = guesses_gpu_cpu[:, 1]
        cpu_first = guesses_cpu[1]
        
        diff = maximum(abs.(gpu_first - cpu_first))
        gpu_sum = sum(gpu_first)
        cpu_sum = sum(cpu_first)
        
        status = diff > 1e-3 ? "❌" : "✅"
        println("$status First guess comparison:")
        println("  Max difference: $diff")
        println("  GPU sum: $gpu_sum, CPU sum: $cpu_sum")
        
        if diff > 1e-3
            println("  GPU: ", round.(gpu_first, digits=6))
            println("  CPU: ", round.(cpu_first, digits=6))
        end
    end
    
    return guesses_gpu
end

function quick_test_generate_all_initial_approximations_gpu(prob; model, rng_seed=12345)
    println("Testing GPU generate_all_initial_approximations...")
    
    testcase = prob()
    z_spec = MVector((testcase.T.N)...)
    U_spec = testcase.T.U
    V_spec = testcase.T.V
    T_spec = PreFlash.compute_single_phase_state(U_spec, V_spec, z_spec, model)

    c_spec = z_spec ./ V_spec
    # Convert to GPU
    c_spec_gpu = CuArray(c_spec)
    z_spec_gpu = CuArray(z_spec)
    n = length(z_spec)
    
    # Test GPU version
    println("Running GPU version...")
    all_approx_gpu = StabilityGPU.generate_all_initial_approximations_gpu(T_spec, V_spec, z_spec_gpu, model, rng_seed)
    
    # Test CPU version
    println("Running CPU version...")
    if rng_seed !== nothing
        Random.seed!(rng_seed)
    end
    all_approx_cpu = Stability.generate_all_initial_approximations(T_spec, V_spec, z_spec, model, rng_seed)
    
    # Convert GPU result for comparison
    all_approx_gpu_cpu = Array(all_approx_gpu)
    # return all_approx_cpu, all_approx_gpu_cpu
    println("\nFinal Comparison:")
    println("GPU shape: ", size(all_approx_gpu))
    println("CPU number of approximations: ", length(all_approx_cpu))
    
    total_gpu = size(all_approx_gpu, 2)
    total_cpu = length(all_approx_cpu)
    
    # Convert CPU to matrix for easier comparison
    all_approx_cpu_matrix = Matrix{Float64}(undef, n, total_cpu)
    for i in 1:total_cpu
        all_approx_cpu_matrix[:, i] = all_approx_cpu[i]
    end
    
    # Compare
    if total_gpu == total_cpu
        println("✅ Same number of approximations: $total_gpu")
        
        # Check if values match
        max_diff = 0.0
        for i in 1:total_gpu
            
            diff = maximum(abs.(all_approx_gpu_cpu[:, i] - all_approx_cpu_matrix[:, i]))
            @info i, all_approx_gpu_cpu[:, i],  all_approx_cpu_matrix[:, i]
            max_diff = max(max_diff, diff)
        end
        println("Maximum difference: ", max_diff)
        println("All match: ", max_diff < 1e-3)
    else
        println("❌ Different number of approximations: GPU=$total_gpu, CPU=$total_cpu")
    end
    
    return all_approx_gpu, all_approx_cpu, total_gpu == total_cpu
end

function test_VT_stabilityAnalysis_gpu(prob; model)
    println("Testing GPU VT_stabilityAnalysis with CPU stability calls...")
    testCase = prob()
    z_spec = MVector((testCase.T.N)...)
    U_spec = testCase.T.U
    V_spec = testCase.T.V
    T_spec = PreFlash.compute_single_phase_state(U_spec, V_spec, z_spec, model)
    # Convert to GPU
    z_spec_gpu = CuArray(z_spec)
    
    # Test GPU version
    println("Running GPU version...")
    result_gpu = StabilityGPU.VT_stabilityAnalysis_gpu(; T_spec, V_spec, z_spec_gpu, model)
    
    # Test CPU version for comparison
    println("Running CPU version...")
    result_cpu = Stability.VT_stabilityAnalysis(; T_spec, V_spec, z_spec=MVector(z_spec...), model)
    return result_cpu, result_gpu
    println("\nResults Comparison:")
    println("GPU - isunstable: ", result_gpu.isunstable, ", D_trial: ", result_gpu.D_trial)
    println("CPU - isunstable: ", result_cpu.isunstable, ", D_trial: ", result_cpu.D_trial)
    println("Iterations - GPU: ", result_gpu.iterations, ", CPU: ", result_cpu.iterations)
    
    # Compare compositions
    c_sol_diff = maximum(abs.(result_gpu.c_sol - result_cpu.c_sol))
    c_spec_diff = maximum(abs.(result_gpu.c_spec - result_cpu.c_spec))
    
    println("Composition differences - c_sol: ", c_sol_diff, ", c_spec: ", c_spec_diff)
    
    # Compare key results
    isunstable_match = result_gpu.isunstable == result_cpu.isunstable
    D_trial_diff = abs(result_gpu.D_trial - result_cpu.D_trial)
    D_trial_match = D_trial_diff < 1e-8
    compositions_match = c_sol_diff < 1e-8 && c_spec_diff < 1e-8
    
    println("\nTest Summary:")
    println("isunstable matches: ", isunstable_match)
    println("D_trial difference: ", D_trial_diff, " (match: ", D_trial_match, ")")
    println("Compositions match: ", compositions_match)
    println("Overall success: ", isunstable_match && D_trial_match && compositions_match)
    
    return result_gpu, result_cpu, isunstable_match && D_trial_match && compositions_match
end

function test_IG3_gpu(prob; model, verbose=false)
    println("Testing GPU IG3...")
    testCase = prob()
    z_spec = MVector((testCase.T.N)...)
    U_spec = testCase.T.U
    V_spec = testCase.T.V
    T_spec = PreFlash.compute_single_phase_state(U_spec, V_spec, z_spec, model)

    z_spec_gpu = CuArray(z_spec)
    
    stability_result_gpu = StabilityGPU.VT_stabilityAnalysis_gpu(; T_spec, V_spec, z_spec_gpu, model)
    c_sol_cpu = stability_result_gpu.c_sol
    c_sol_gpu = CuArray(c_sol_cpu)

    # Test GPU version
    println("Running GPU version...")
    result_gpu = StabilityGPU.IG3_gpu(c_sol_gpu, T_spec, U_spec, V_spec, z_spec_gpu; model, verbose)
    # Test CPU version
    println("Running CPU version...")
    result_cpu = Stability.IG3(c_sol_cpu, T_spec, U_spec, V_spec, z_spec; model, verbose)
    
    println("\nIG3 Results Comparison:")
    
    # Compare results
    if result_gpu isa CuArray && result_cpu isa Vector
        result_gpu_cpu = Array(result_gpu)
        diff = maximum(abs.(result_gpu_cpu - result_cpu))
        println("Maximum difference: ", diff)
        println("Results match: ", diff < 1e-3)
        return result_cpu, result_gpu, diff < 1e-3
    else
        println("Return types - GPU: ", typeof(result_gpu), ", CPU: ", typeof(result_cpu))
        return result_cpu, result_gpu, false
    end
end

function quick_test_attempt_two_phase_flash_gpu(prob; model)
    println("Testing GPU attempt_two_phase_flash...")
    testCase = prob()
    N_spec = MVector((testCase.T.N)...)
    U_spec = testCase.T.U
    V_spec = testCase.T.V
    T_stab = PreFlash.compute_single_phase_state(U_spec, V_spec, N_spec, model)
    S_one = EOS.S_EOS(T_stab, V_spec, N_spec; model)
    numPhases = 2
    stab = Stability.VT_stabilityAnalysis(; model, T_spec=T_stab, V_spec, z_spec=N_spec)
    x0 = Stability.IG3(stab.c_sol, T_stab, U_spec, V_spec, N_spec; model, verbose=false)
    # initial_guess = x0
    
    ρ_mix = sum(N_spec .* model.Mw) / V_spec
    N_G = x0[1:model.Nc]
    V_G = x0[model.Nc+1]
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
    
    initial_guess = vcat(ρG ./ ρ_mix, α_est, T_stab)
    # Convert to GPU
    initial_guess_gpu = CUDA.CuArray(initial_guess)
    # @show typeof(initial_guess_gpu), typeof(initial_guess)
    N_spec_gpu = CUDA.CuArray(N_spec)
    Scale = vcat(N_spec, V_spec, model.T_c)
    Scale_gpu = CuArray(Scale)
    # Test GPU version
    println("Running GPU version...")
    @time success_gpu, result_gpu = PreFlashGPU.attempt_two_phase_flash_gpu(
        initial_guess_gpu, U_spec, V_spec, N_spec_gpu, model,
        Scale_gpu, T_stab, S_one, numPhases
    )
    @show result_gpu
    # Test CPU version
    println("Running CPU version...")
    @time success_cpu, result_cpu = PreFlash.attempt_two_phase_flash(
        initial_guess, U_spec, V_spec, N_spec, model,
        Scale, T_stab, S_one, numPhases
    )
    
    println("\nFlash Results Comparison:")
    println("Success - GPU: $success_gpu, CPU: $success_cpu, Match: $(success_gpu == success_cpu)")
    
    if success_gpu == success_cpu
        result_gpu_cpu = Array(result_gpu)
        diff = maximum(abs.(result_gpu_cpu - result_cpu))
        println("Result difference: $diff")
        println("Results match: $(diff < 1e-8)")
        return success_gpu, result_gpu, success_cpu, result_cpu, diff < 1e-8
    else
        println("Different success outcomes - cannot compare results directly")
        return success_gpu, result_gpu, success_cpu, result_cpu, false
    end
end

function quick_test_stability_analysis_fallback_gpu(prob; model)
    println("Testing GPU stability_analysis_fallback with CuArray precision...")
    testCase = prob()
    N_spec = MVector((testCase.T.N)...)
    U_spec = testCase.T.U
    V_spec = testCase.T.V
    # Convert to GPU with explicit Float64
    N_spec_gpu = CuArray(N_spec)
    T_stab = PreFlash.compute_single_phase_state(U_spec, V_spec, N_spec, model)
    S_one = EOS.S_EOS(T_stab, V_spec, N_spec; model)
    Scale = vcat(N_spec, V_spec, model.T_c)
    numPhases = 2
    println("Precision check:")
    println("  N_spec_gpu type: ", eltype(N_spec_gpu))
    
    # Test GPU version
    println("Running GPU version...")
    @time success_gpu, result_gpu = PreFlashGPU.stability_analysis_fallback_gpu(
        U_spec, V_spec, N_spec_gpu, model, T_stab, S_one, Scale, numPhases
    );
    
    # Test CPU version
    println("Running CPU version...")
    @time success_cpu, result_cpu = PreFlash.stability_analysis_fallback(
        U_spec, V_spec, N_spec, model, T_stab, S_one, Scale, numPhases
    );
    
    println("\nStability Fallback Results Comparison:")
    println("Success - GPU: $success_gpu, CPU: $success_cpu, Match: $(success_gpu == success_cpu)")
    println("Result precision - GPU: ", eltype(result_gpu), ", CPU: ", eltype(result_cpu))
    
    if success_gpu == success_cpu
        result_gpu_cpu = Array(result_gpu)
        diff = maximum(abs.(result_gpu_cpu - result_cpu))
        
        # Use appropriate tolerance based on precision
        tolerance = eltype(result_gpu) == Float32 ? 1e-6 : 1e-12
        println("Result difference: $diff")
        println("Tolerance: $tolerance")
        println("Results match: $(diff < tolerance)")
        
        return success_gpu, result_gpu, success_cpu, result_cpu, diff < tolerance
    else
        println("Different success outcomes - cannot compare results directly")
        return success_gpu, result_gpu, success_cpu, result_cpu, false
    end
end

function test_solve_rho_Q_from_UVN_gpu(prob; model)
    println("Testing GPU solve_rho_Q_from_UVN with CuArray precision...")
    testCase = prob()
    N_spec = MVector((testCase.T.N)...)
    U_spec = testCase.T.U
    V_spec = testCase.T.V
    # Convert to GPU with explicit Float64
    N_spec_gpu = CuArray(N_spec)
    x_guess = nothing
    x_guess_gpu = nothing
    numPhases = 2
    singlePhaseSure = false
    println("Precision check:")
    println("  N_spec_gpu type: ", eltype(N_spec_gpu))
    
    # Test GPU version
    println("Running GPU version...")
    @time result_gpu = PreFlashGPU.solve_rho_Q_from_UVN_gpu(
        U_spec, V_spec, N_spec_gpu; 
        model, singlePhaseSure, x_guess, numPhases
    )
    
    # Test CPU version
    println("Running CPU version...")
    @time result_cpu = PreFlash.solve_rho_Q_from_UVN(
        U_spec, V_spec, MVector(N_spec); 
        model, singlePhaseSure, x_guess, numPhases
    )
    
    println("\nFlash Results Comparison:")
    # println("Status - GPU: $(result_gpu.status), CPU: $(result_cpu.status), Match: $(result_gpu.status == result_cpu.status)")
    println("Result precision - GPU: ", eltype(result_gpu.flash_result), ", CPU: ", eltype(result_cpu.flash_result))
    
    if result_gpu.success == result_cpu.success
        result_gpu_cpu = Array(result_gpu.flash_result)
        diff = maximum(abs.(result_gpu_cpu - result_cpu.flash_result))
        
        # Use appropriate tolerance based on precision
        tolerance = eltype(result_gpu.flash_result) == Float32 ? 1e-3 : 1e-6
        println("Result difference: $diff")
        println("Tolerance: $tolerance")
        println("Results match: $(diff < tolerance)")
        
        return result_gpu, result_cpu, diff < tolerance
    else
        println("Different status outcomes - cannot compare results directly")
        return result_gpu, result_cpu, false
    end
end

function test_flash_calculation_gpu(prob; model)
    println("Testing GPU flash_calculation (high-level interface)...")
    testCase = prob()
    N_spec = MVector((testCase.T.N)...)
    U_spec = testCase.T.U
    V_spec = testCase.T.V
    x_guess = nothing
    singlePhaseSure = false
    # Convert to GPU
    N_spec_gpu = CuArray(N_spec)
    
    println("Precision check:")
    println("  N_spec_gpu type: ", eltype(N_spec_gpu))
    
    # Test GPU version
    println("Running GPU version...")
    @time status_gpu, result_gpu = PreFlashGPU.flash_calculation_gpu(
        U_spec, V_spec, N_spec_gpu; 
        model, singlePhaseSure, x_guess
    )
    
    # Test CPU version
    println("Running CPU version...")
    @time status_cpu, result_cpu = PreFlash.flash_calculation(
        U_spec, V_spec, MVector(N_spec); 
        model, singlePhaseSure, x_guess
    )
    
    println("\nFlash Calculation Results Comparison:")
    println("Status - GPU: $status_gpu, CPU: $status_cpu, Match: $(status_gpu == status_cpu)")
    println("Result precision - GPU: ", eltype(result_gpu), ", CPU: ", eltype(result_cpu))
    
    if status_gpu == status_cpu
        result_gpu_cpu = Array(result_gpu)
        diff = maximum(abs.(result_gpu_cpu - result_cpu))
        
        tolerance = eltype(result_gpu) == Float32 ? 1e-3 : 1e-6
        println("Result difference: $diff")
        println("Tolerance: $tolerance")
        println("Results match: $(diff < tolerance)")
        
        return (status_gpu, result_gpu), (status_cpu, result_cpu), diff < tolerance
    else
        println("Different status outcomes - cannot compare results directly")
        return (status_gpu, result_gpu), (status_cpu, result_cpu), false
    end
end

using CUDA
using InteractiveUtils   # for code_typed, methods, which
using Logging
# optional: using Cthulhu

"""
gpu_inspect_run(f, args...; rethrow_err=true)

Runs `f(args...)` (synchronized). On error it:
 - prints the exception and backtrace
 - prints a summary of every argument (typeof, eltype if array, size if array)
 - detects whether any argument or nested field contains a Dual-like type
 - prints the selected method (which) and methods(f)
 - attempts `code_typed(f, Tuple{...})` (without `interactive=true` to avoid the MethodError)
 - gives guidance for GPU vs CPU AD issues
"""
function gpu_inspect_run(f, args...; rethrow_err::Bool=true)
    try
        CUDA.@sync f(args...)
        return :ok
    catch err
        bt = catch_backtrace()
        @error "Caught exception" exception = err

        println("\n--- Exception ---")
        showerror(stderr, err)
        println("\n--- Backtrace (most recent call last) ---")
        Base.show_backtrace(stderr, bt)
        println("--- End backtrace ---\n")

        # Helper to detect if a value or element contains a Dual type (heuristic)
        function contains_dual(x)
            # quick check for concrete Dual types:
            t = typeof(x)
            # Heuristic: name contains "Dual" (adjust to your module's Dual type)
            if occursin("Dual", string(t))
                return true
            end
            # If array or container, check element type if possible
            try
                el = eltype(x)
                if occursin("Dual", string(el))
                    return true
                end
            catch
                # eltype may fail for scalars or custom types
            end
            # If iterable and not string, check some elements (avoid huge scans)
            try
                if isa(x, AbstractArray)
                    # check first few elements
                    for i in 1:min(length(x), 8)
                        if contains_dual(x[i])
                            return true
                        end
                    end
                end
            catch
                # ignore indexing errors
            end
            return false
        end

        # Print argument summaries
        println("=== Argument summaries ===")
        for (i, a) in enumerate(args)
            typ = typeof(a)
            s = "arg $i: typeof = $typ"
            try
                if isa(a, AbstractArray)
                    s *= ", eltype = $(eltype(a)), size = $(size(a))"
                end
            catch _; end
            s *= ", contains_dual = $(contains_dual(a))"
            println(s)
        end
        println("=== End summaries ===\n")

        # Show which method would be called
        argtypes = Tuple(map(typeof, args))
        println("=== Selected method info ===")
        try
            m = which(f, argtypes)
            println("which(...) => ", m)
        catch inner
            @warn "which(...) failed" exception = inner
        end
        println("Available methods for the function:")
        try
            display(methods(f))
        catch
            # ignore
        end
        println("=== End method info ===\n")

        # Try code_typed for the specific signature (do NOT pass `interactive=true` here)
        println("=== Attempting code_typed(f, Tuple{...}) ===")
        try
            code_typed(f, argtypes)  # note: interactive kw sometimes unsupported for exceptions
        catch inner
            @warn "code_typed failed for this signature" exception = inner
            println("Hint: try running the above `code_typed` manually with the concrete types in REPL.")
        end
        println("=== End code_typed attempt ===\n")

        # Helpful hints for this specific error
        println("=== Quick diagnosis tips ===")
        println("- The error indicates a `Dual{Float32}` ended up on the GPU.")
        println("- Common causes:")
        println("  1) You used a CPU AD library (e.g., ForwardDiff.Dual) that is not GPU-compatible.")
        println("  2) A broadcast or operation created a Dual on the device (mixing CPU Duals and CuArrays).")
        println("  3) A function returns Duals for scalar ops and you broadcasted it over a CuArray.")
        println("- Fixes to consider:")
        println("  • Move AD to CPU: compute Duals / derivatives on CPU and only push plain Float arrays to GPU.")
        println("  • Use a GPU-capable AD tool / make your Dual type isbits and GPU-friendly (non-trivial).")
        println("  • Avoid broadcasting functions that produce Duals directly on CuArrays; instead map on CPU.")
        println("  • Check all `.eltype` of your CuArrays and CPU arrays before the call.")
        println("  • Insert explicit conversions: `Float32.(x)` or `CUDA.unsafe_convert` only where appropriate.")
        println("=== End tips ===\n")

        # rethrow
        rethrow_err && rethrow(err)
    end
end


function test_a_res_gpu(prob; model, model_gpu, M = 2)
testCase = prob()
N_spec = MVector((testCase.T.N)...)
U_spec = testCase.T.U
V_spec = testCase.T.V
T_spec = PreFlash.compute_single_phase_state(U_spec, V_spec, N_spec, model)
# Convert to GPU
N_spec_gpu = cu(N_spec)
# out = CUDA.zeros(Float64, model_gpu.Nc)
out = zeros(Float64, model_gpu.Nc)
# x = CUDA.zeros(Float64, model_gpu.Nc)
# @show typeof(N_spec)
N = model.Nc
x = CuArray([1.0, 2.0])
dx = CuArray([0.0, 0.0])
y = CUDA.zeros(Float32, N)
dy = CUDA.zeros(Float32, N)

RT = model.R * T_spec
F = z -> EOS.a_ideal(T_spec, V_spec, z; model)
∂F_N = ForwardDiff.gradient(F, N_spec)
# mu1 = EosGPU.da_ideal_dzi(T_spec, V_spec, N_spec, model, 1)
mu = EOS.chem_pot(T_spec, V_spec, N_spec; model)
mu_out = CUDA.zeros(Float32, model.Nc)
x_local = CUDA.zeros(Float32, model.Nc)
mu_out = EosGPU.launch_mu_batch(T_spec, V_spec, N_spec; model = model_gpu, M)
# @show Array(mu_out), mu
return mu_out
# try
# @cuda threads=1 EosGPU.μ_tot(1, T_spec, V_spec, N_spec_gpu, model_gpu, x_local, mu_out)
# # synchronize()
# catch error
#         @error "Caught exception" exception=error

#         println("\n=== Cthulhu-style typed code for the error ===")
#         try
#             code_typed(error; interactive = true)
#         catch inner
#             @warn "code_typed could not introspect this error" exception=inner
#         end

#         rethrow(error)   # optional: crash after introspection
# end
# @show Array(mu_out), mu
# synchronize()
# mu2 = EosGPU.μ_tot(2, T_spec, V_spec, N_spec, model_gpu)
# @show ∂F_N
# @show mu, mu2, mu1
# @show a_res_device
# @show a_res_cpu
end

res =test_a_res_gpu(Problems.prob_1; model=model_1_4, model_gpu =model_1_4_gpu, M = 1_000_000_00);
nothing


kwargs_1_4_gpu = (;
    Mw =  SA[16.04e-3, 34.083e-3],
    T_c = SA[190.4, 373.2],  # Critical temperatures in K
    P_c = SA[46.0e5, 89.4e5],  # Critical pressures in Pa    
    ρ_c = SA[10139.0, 10190.0],
    ω = SA[0.011, 0.081], #0.0810919       # Acentric factors
    # ω=[0.011, 0.008], #0.0810919       # Acentric factors
    δ = SA[0.0 0.083; 0.083 0.0],
    α = SA[
        19.25 5.213e-2 1.197e-5 -1.132e-8
        31.94 1.463e-3 2.432e-5 -1.176e-8
    ],
)



model_1_4_gpu = EosGPU.PengRobinson_gpu(; kwargs_1_4_gpu...);

res = EosGPU.test_kernel(2, model_1_4_gpu);
mu_res = EosGPU.test_mu_res_kernel(2, model_1_4_gpu);


test_flash_calculation_gpu(Problems.prob_5; model=model_5_6)
nothing

test_solve_rho_Q_from_UVN_gpu(Problems.prob_5; model=model_5_6)
nothing

quick_test_stability_analysis_fallback_gpu(Problems.prob_5; model=model_5_6)
nothing

quick_test_attempt_two_phase_flash_gpu(Problems.prob_1; model=model_1_4)
nothing 

x_cpu, x_gpu =test_IG3_gpu(Problems.prob_; model=model_5_6, verbose=false)

x_cpu, x_gpu =test_VT_stabilityAnalysis_gpu(Problems.prob_6; model=model_5_6);


x_cpu, x_gpu =quick_test_generate_all_initial_approximations_gpu(Problems.prob_5; model=model_5_6, rng_seed=12345)

run_flash_calculations(Problems.prob_1, model_1_4)
run_flash_calculations(Problems.prob_2, model_1_4)
run_flash_calculations(Problems.prob_3, model_1_4)
run_flash_calculations(Problems.prob_4, model_1_4)
run_flash_calculations(Problems.prob_5, model_5_6)
run_flash_calculations(Problems.prob_6, model_5_6)


Nbatch = 10000
Nc = 6

# CPU-side data
U_cpu = rand(Float64, Nbatch)
V_cpu = rand(Float64, Nbatch)

# N is an array of Static Vectors
N_cpu = [@SVector rand(Nc) for i in 1:Nbatch]

U_d = CuArray(U_cpu)              # CuArray{Float64,1}
V_d = CuArray(V_cpu)              # CuArray{Float64,1}
N_d = CuArray(N_cpu)              # CuArray{SVector{Nc,Float64},1}

using CUDA
CUDA.versioninfo()
A1 = CuArray(rand(Float32, 10000, 10000))
B1 = CuArray(rand(Float32, 10000, 10000))

@time C1 = A1 * B1; # Runs on GPU using cuBLAS

A = rand(Float32, 10000, 10000);
B = rand(Float32, 10000, 10000);

@time C = A * B;  # Runs on CPU
CUDA.functional()
CUDA.synchronize()

A = CuArray(rand(Float32, 3, 3))
b = CuArray(rand(Float32, 3))
A \ b

using CUDA

dev = CUDA.device()

println("Name: ", CUDA.name(dev))
println("Compute Capability: ", CUDA.capability(dev))
println("SM count: ", CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
println("Warp size: ", CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_WARP_SIZE))
println("Max threads/block: ", CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK))
println("Concurrent kernels: ", CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_CONCURRENT_KERNELS))
println("Async engines: ", CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT))





# simple kernel — writes z .= x + 2*y (GPU device code)
function mykernel!(z, x, y)
    i = (blockIdx().x-1)*blockDim().x + threadIdx().x
    if i <= length(z)
        z[i] = x[i] + 2f0 * y[i]
    end
    return
end

function call_kernel!(z::CuDeviceVector{Float32},
                      x::CuDeviceVector{Float32},
                      y::CuDeviceVector{Float32})
    N = length(z)
    threads = 256
    blocks = cld(N, threads)
    @cuda threads=threads blocks=blocks mykernel!(z, x, y)
    return nothing
end

function forward_and_grad(x::CuArray{Float32},
                          y::CuArray{Float32})
    N = length(x)
    z = CUDA.zeros(Float32, N)

    # Allocate gradient buffers
    dz = CUDA.ones(Float32, N)          # seed/output-adjoint (for reverse mode)
    dx = CUDA.zeros(Float32, N)         # gradient w.r.t x
    dy = CUDA.zeros(Float32, N)         # gradient w.r.t y

    # call Enzyme to differentiate call_kernel! w.r.t x and y
    Enzyme.autodiff(
      Enzyme.Reverse,
      call_kernel!,
      Enzyme.DuplicatedNoNeed(z, dz),  # treat z as output with seed dz
      Enzyme.Duplicated(x, dx),        # differentiate w.r.t x
      Enzyme.Duplicated(y, dy)         # differentiate w.r.t y
    )

    return dx, dy
end

# Example usage
N = 10
x = CUDA.randn(Float32, N)
y = CUDA.randn(Float32, N)
dx, dy = forward_and_grad(x, y)

function gpu_func(x, y)
    z = x .+ 2 .* y       # elementwise operations on CuArray
    w = z .^ 2
    return sum(w)         # can also include other differentiable GPU calls
end

x = cu(rand(Float32, 10))
y = cu(rand(Float32, 10))
dx = CUDA.zeros(Float32, 10)
dy = CUDA.zeros(Float32, 10)
dz_dx = Enzyme.autodiff_deferred(Enzyme.Reverse, Const(gpu_func), Active, Duplicated(x, dx), Duplicated(y, dy))
dz_dy = Enzyme.autodiff_deferred(Enzyme.Reverse, gpu_func, y)

AutoDiffGPU.test()

using ForwardDiff, CUDA
CUDA.allowscalar()
f(x) = sum(x .^ 2)  # simple sum-of-squares function
z = cu([1.0, 2.0, 3.0])  # CuArray input
grad = ForwardDiff.gradient(f, z)  # works with CuArrays
println("Gradient: ", grad)  # should be [2.0, 4.0, 6.0]