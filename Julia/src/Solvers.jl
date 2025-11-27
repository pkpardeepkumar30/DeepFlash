module Solvers
using ExportAll
using NLsolve
using LinearAlgebra
using CUDA, ForwardDiff

function first_order_linesearch(f, grad_f, x, Δx; α=0.5, β=0.8)
    # Initial step size
    α_current = 1.0
    fx = f(x)
    grad_fx = grad_f(x)

    # Armijo condition (sufficient decrease)
    while f(x + α_current * Δx) > fx + α * α_current * dot(grad_fx, Δx)
        α_current *= β
        if α_current < 1e-10  # Prevent infinite loop
            break
        end
    end
    return α_current
end

function armijo_linesearch(f, grad_f, x, Δx; c=0.1, β=0.5, max_iter=100)
    α = 1.0  # Initial step size
    fx = f(x)
    grad_fx = grad_f(x)
    directional_derivative = dot(grad_fx, Δx)
    
    # Ensure Δx is a descent direction (avoid infinite loops)
    if directional_derivative >= 0
        @warn "Δx is not a descent direction!"
        return 1e-10
    end
    # @show "Directional derivative: $directional_derivative"
    for _ in 1:max_iter
        f_new = f(x + α * Δx)
        if f_new <= fx + c * α * directional_derivative
            return α
        end
        α *= β  # Reduce step size
    end
    return α  # Return last α even if max_iter reached
end

function second_order_linesearch(f, grad_f, x, Δx; α=0.5, β=0.8)
    # Initial step size
    α_current = 1.0
    fx = f(x)
    grad_fx = grad_f(x)

    # Armijo condition with first-order approximation
    while f(x + α_current * Δx) > fx + α * α_current * dot(grad_fx, Δx)
        α_current *= β
        if α_current < 1e-10
            break
        end
    end
    return α_current
end

# Convert vector residual to scalar objective function
function objective_from_residuals(r)
    return x -> 0.5 * sum(abs2, r(x))  # F(x) = ½‖r(x)‖²
end

# Compute third directional derivative of scalar function F(x)
function third_deriv_f(F, x, dx)
    ϕ(α) = F(x .+ α .* dx)
    return ForwardDiff.derivative(α -> ForwardDiff.derivative(
               α -> ForwardDiff.derivative(ϕ, α), α), 0.0)
end

function third_order_linesearch(f, x, Δx; α=0.5, β=0.8)
    # Create scalar function from residuals
    F = objective_from_residuals(f)
    
    fx = F(x)
    grad_fx = ForwardDiff.gradient(F, x)
    hess_fx = ForwardDiff.hessian(F, x)
    t3 = third_deriv_f(F, x, Δx)

    α_current = 1.0

    while true
        x_new = x .+ α_current .* Δx
        fx_new = F(x_new)

        approx = fx + α * α_current * dot(grad_fx, Δx) +
                 0.5 * α_current^2 * dot(Δx, hess_fx * Δx) +
                 (1/6) * α_current^3 * t3

        if fx_new <= approx
            break
        end

        α_current *= β
        if α_current < 1e-10
            break
        end
    end

    return α_current
end

function backtracking_linesearch(f, x, fx, J, Δx; c=1e-4, β=0.5, max_ls=10)
    α = 1.0

    φ_x = 0.5 * norm(fx)^2                # current merit value
    g  = J' * fx                          # grad(φ)
    directional_derivative = dot(g, Δx)   # scalar

    @inbounds for k in 1:max_ls
        x_trial = x .+ α .* Δx
        Fx_trial = f(x_trial)
        φ_trial = 0.5 * norm(Fx_trial)^2

        if φ_trial <= φ_x + c * α * directional_derivative
            return α                      # accept step size
        end

        α *= β
    end

    return α
end

function newton_stability(f, grad_f, hess_f, x0; tol=1e-8, maxiter=100, c=1e-4, β=0.5)
    
    x = copy(x0)
    α = 1.0
    
    for iter = 1:maxiter
        fx = f(x)
        J = grad_f(x)
        Δx = -(J + 1e-10*I) \ fx  # Regularized Jacobian (GPU-friendly)
        
        # Backtracking line search on merit function
        α = backtracking_linesearch(f, x, fx, J, Δx; c=c, β=β)
        
        rel_tol = converged(x, Δx; tol)

        if rel_tol
            return x, true, iter
        end

        x .+= α * Δx
    end

    println("Max iterations reached without convergence.")
    return x, false, maxiter
end


# function check_convergence(x, Δx, g; tol)
#     step_tol = maximum(abs.(Δx) ./ max.(abs.(x), 1.0))
#     grad_tol = maximum(abs.(g)  ./ max.(abs.(x), 1.0))

#     return (step_tol ≤ tol) && (grad_tol ≤ sqrt(tol))
# end

function converged(x, Δx; tol)
    all(abs.(Δx) .≤ tol * max.(abs.(x), 1.0))
end

function newton_mixed(f, g, H, x0; tol=1e-8, maxiter=100)
    # @show tol
    
    tol = 1e-10
    # @show chanigng_tol = tol
    x = CuArray(x0)
    for iter in 1:maxiter
        # Compute derivatives on CPU with ForwardDiff
        grad_fx = CuArray(g(Array(x)))
        hess_fx = CuArray(H(Array(x)))

        # Solve Newton step on GPU
        Δx = -(hess_fx + 1e-10*I) \ grad_fx

        x .+= Δx
        x_cpu  = Array(x)
        Δx_cpu = Array(Δx)
        rel_tol = converged(x_cpu, Δx_cpu; tol)
        
        if rel_tol            
            # @show "converged in iter: $iter"
            return x_cpu, true, iter
        end
    end

    return Array(x), false, maxiter
end

function assess_convergence(x::AbstractVector,
    x_previous::AbstractVector,
    f::AbstractVector,
    xtol::Real,
    ftol::Real)
    # Component-wise relative x convergence check
    # @show xtol
    x_rel_diff = abs.(x .- x_previous)
    x_scale = max.(abs.(x), 1.0)  # Prevent too tight tolerance for x≈0
    x_converged = all(x_rel_diff .≤ xtol .* x_scale)

    # Component-wise absolute f convergence check
    f_converged = all(abs.(f) .≤ ftol)

    return x_converged, f_converged
end


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