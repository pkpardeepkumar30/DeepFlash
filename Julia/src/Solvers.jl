module Solvers
using ExportAll
using NLsolve
using LinearAlgebra

# We are overloading the `assess_convergence` function from NLsolve. This is useful
# when we want to use a custom convergence criterion in our solvers.
function NLsolve.assess_convergence(x::AbstractVector,
    x_previous::AbstractVector,
    f::AbstractVector,
    xtol::Real,
    ftol::Real)
    # Component-wise relative x convergence check
    # @show xtol
    x_rel_diff = abs.(x .- x_previous)
    x_scale = max.(abs.(x), 1.0)  # Prevent too tight tolerance for x≈0
    x_converged = all(x_rel_diff .≤ xtol .* x_scale) #&& norm(f) < 1.0 # Global convergence check
    # Component-wise absolute f convergence check
    f_converged = all(abs.(f) .≤ ftol)

    return x_converged, f_converged
end

# function NLsolve.assess_convergence(x,
#                             x_previous,
#                             f,
#                             xtol,
#                             ftol)
#     x_converged, f_converged = false, false
#     if norm(x-x_previous) <= xtol
#         x_converged = true
#     end
#     if maximum(abs, f) <= ftol
#         f_converged = true
#     end

#     return x_converged, f_converged
# end

@exportAll()

end