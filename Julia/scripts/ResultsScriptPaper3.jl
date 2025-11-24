if false
    include("../src/MultiComponent.jl")
    include("../src/Sols.jl")
    using .MultiComponent
end
# cd("Pipelines/scripts/")
using Pkg
Pkg.activate("./Julia/")
using Revise  # Load Revise AFTER activating environment but BEFORE loading your package

using CO2Transport
using CO2Transport.MultiComponent
# using Clapeyron

using NLsolve
using StaticArrays
using Printf
using Statistics

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
    Sols.flash_calculation(p1.U, p1.V, MVector((p1.N)...); model)
end

run_flash_calculations(Problems.prob_1, model_1_4)
run_flash_calculations(Problems.prob_2, model_1_4)
run_flash_calculations(Problems.prob_3, model_1_4)
run_flash_calculations(Problems.prob_4, model_1_4)
run_flash_calculations(Problems.prob_5, model_5_6)
run_flash_calculations(Problems.prob_6, model_5_6)

using CUDA
CUDA.versioninfo()
A1 = CuArray(rand(Float32, 10000, 10000))
B1 = CuArray(rand(Float32, 10000, 10000))

@time C1 = A1 * B1 # Runs on GPU using cuBLAS

A = rand(Float32, 10000, 10000)
B = rand(Float32, 10000, 10000)

@time C = A * B  # Runs on GPU using cuBLAS
CUDA.functional()
CUDA.synchronize()

A = cu(rand(Float32, 3, 3))
b = cu(rand(Float32, 3))
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
