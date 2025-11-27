module MultiComponent
using Revise
using ExportAll
using ForwardDiff
# using Zygote
using NaNMath
using Plots
using PolygonOps
using StaticArrays
using Clapeyron

include("Problems.jl")
include("Solvers.jl")
include("Scalers.jl")
include("EOS.jl")
include("CubicFuncs.jl")
# include("PTFlash.jl")
include("Flash.jl")
include("Stability.jl")
include("PreFlash.jl")



export Problems, EOS, CubicFuncs, Solvers, Flash, Stability, Scalers, PreFlash

end