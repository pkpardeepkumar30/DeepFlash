module AutoDiffGPU
using ForwardDiff
using CUDA
import CUDA: sin, cos, tan, exp, log
# ----- Dual number type (isbits so it works on GPU) -----
mutable struct Dual{T}
    val::T
    dot::T
end

# constructors
Dual(x::T, d::T) where {T} = Dual{T}(x, d)
Dual(x::T) where {T} = Dual{T}(x, zero(T))

# ----- Basic promotion & conversion so mixed ops work -----
Base.promote_rule(::Type{Dual{T}}, ::Type{T}) where {T} = Dual{T}
Base.convert(::Type{Dual{T}}, x::T) where {T} = Dual{T}(x, zero(T))

# ----- Arithmetic on Dual numbers -----
# addition
Base.:+(a::Dual{T}, b::Dual{T}) where {T} = Dual{T}(a.val + b.val, a.dot + b.dot)
Base.:+(a::Dual{T}, b::T) where {T} = Dual{T}(a.val + b, a.dot)
Base.:+(a::T, b::Dual{T}) where {T} = Dual{T}(a + b.val, b.dot)

# subtraction
Base.:-(a::Dual{T}, b::Dual{T}) where {T} = Dual{T}(a.val - b.val, a.dot - b.dot)
Base.:-(a::Dual{T}, b::T) where {T} = Dual{T}(a.val - b, a.dot)
Base.:-(a::T, b::Dual{T}) where {T} = Dual{T}(a - b.val, -b.dot)

# multiplication
Base.:*(a::Dual{T}, b::Dual{T}) where {T} =
    Dual{T}(a.val*b.val, a.dot*b.val + a.val*b.dot)
Base.:*(a::Dual{T}, b::T) where {T} = Dual{T}(a.val * b, a.dot * b)
Base.:*(a::T, b::Dual{T}) where {T} = Dual{T}(a * b.val, a * b.dot)

# division
Base.:/(a::Dual{T}, b::Dual{T}) where {T} =
    Dual{T}(a.val/b.val, (a.dot*b.val - a.val*b.dot) / (b.val*b.val))
Base.:/(a::Dual{T}, b::T) where {T} = Dual{T}(a.val / b, a.dot / b)
Base.:/(a::T, b::Dual{T}) where {T} = Dual{T}(a / b.val, (-a * b.dot) / (b.val*b.val))

# power by small integer (n >= 0)
function Base.:^(a::Dual{T}, n::Integer) where {T}
    # simple repeated multiply (works for small integer exponents)
    if n == 0
        return Dual{T}(one(T), zero(T))
    elseif n == 1
        return a
    else
        res = a
        for i in 2:n
            res = res * a
        end
        return res
    end
end

# math helpers (example: sin, cos, exp) -- add as needed
Base.sin(a::Dual{T}) where {T} = Dual{T}(sin(a.val), cos(a.val) * a.dot)
Base.cos(a::Dual{T}) where {T} = Dual{T}(cos(a.val), -sin(a.val) * a.dot)
Base.exp(a::Dual{T}) where {T} = Dual{T}(exp(a.val), exp(a.val) * a.dot)
Base.log(a::Dual{T}) where {T} = Dual{T}(log(a.val), a.dot / a.val)
# add more functions if you use them inside GPU functions

sin(a::Dual{T}) where {T} = Dual{T}(sin(a.val), cos(a.val) * a.dot)
cos(a::Dual{T}) where {T} = Dual{T}(cos(a.val), -sin(a.val) * a.dot)
exp(a::Dual{T}) where {T} = Dual{T}(exp(a.val), exp(a.val) * a.dot)
log(a::Dual{T}) where {T} = Dual{T}(log(a.val), a.dot / a.val)

Base.asin(a::Dual) = Dual(asin(a.val), a.dot / sqrt(1 - a.val^2))
Base.acos(a::Dual) = Dual(acos(a.val), -a.dot / sqrt(1 - a.val^2))
Base.atan(a::Dual) = Dual(atan(a.val), a.dot / (1 + a.val^2))


Base.sinh(a::Dual) = Dual(sinh(a.val), cosh(a.val) * a.dot)
Base.cosh(a::Dual) = Dual(cosh(a.val), sinh(a.val) * a.dot)
Base.tanh(a::Dual) = Dual(tanh(a.val), a.dot / (cosh(a.val)^2))

Base.exp(a::Dual) = Dual(exp(a.val), exp(a.val) * a.dot)
Base.log(a::Dual) = Dual(log(a.val), a.dot / a.val)

exp(a::Dual) = Dual(exp(a.val), exp(a.val) * a.dot)
log(a::Dual) = Dual(log(a.val), a.dot / a.val)

@inline exp(a::Dual) = begin
    x  = a.val
    dx = a.dot
    ex = exp(x)          # GPU device exp
    Dual(ex, ex * dx)
end

Base.:^(a::Dual, n::Number) =
    Dual(a.val^n, n * a.val^(n-1) * a.dot)

Base.abs(a::Dual) = Dual(abs(a.val), a.val > 0 ? a.dot : -a.dot)
Base.max(a::Dual, b::Dual) =
    a.val > b.val ? a : b
Base.max(a::Dual, b::Number) =
    a.val > b ? a : Dual(b, zero(a.dot))
Base.max(a::Number, b::Dual) =
    a > b.val ? Dual(a, zero(b.dot)) : b
Base.min(a::Dual, b::Dual) =
    a.val < b.val ? a : b
Base.min(a::Dual, b::Number) =
    a.val < b ? a : Dual(b, zero(a.dot))
Base.min(a::Number, b::Dual) =
    a < b.val ? Dual(a, zero(b.dot)) : b
Base.clamp(a::Dual, lo::Number, hi::Number) =
    a.val < lo ? Dual(lo, zero(a.dot)) :
    a.val > hi ? Dual(hi, zero(a.dot)) : a

relu(a::Dual) = a.val > 0 ? a : Dual(zero(a.val), zero(a.dot))
Base.sign(a::Dual) = Dual(sign(a.val), zero(a.dot))


# mark small functions @inline for GPU performance
@generated function Base.sin(a::Dual{T}) where {T}
    quote
        Dual{T}(sin(a.val), cos(a.val) * a.dot)
    end
end

# ----- GPU-compatible function that can accept Dual -----
# Example: g(x) = x^3 + 2x
@generated function g(x)
    # generated so it compiles for Dual{T} or Float types
    quote
        return x^3 + 2f0 * x
    end
end

# ----- Kernel: apply g elementwise using Dual numbers to get derivative -----
function dual_forward_kernel(f, y, dy, x, N)
    # i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= N
        # seed tangent = 1.0 for derivative wrt x[i]
        xd = Dual(x[i], 1f0)
        rd = f(xd)
        y[i] = rd.val
        dy[i] = rd.dot
    end
    return
end

@inline function gradient(f, x, y, dy)
    # res = ForwardDiff.gradient(f, x)

    N = length(x)
    @inbounds for i in 1:N
        xd = Dual.(x, zero(eltype(x)))   # vector of Dual
        xd[i].dot = one(eltype(x))       # seed derivative
        rd = f(xd)
        y[1]   = rd.val
        dy[i]  = rd.dot
    end
end


# ----- Wrapper to allocate arrays and launch kernel -----
function forward_ad_on_gpu(f, x)
    N = length(x)
    y = CUDA.zeros(Float32, N)
    ∂y = CUDA.zeros(Float32, N)
    # @show eltype(x)
    
    # threads = 128
    # blocks = cld(N, threads)
    @cuda threads=1 blocks=1 gradient(f, x, y, ∂y)

    return y, ∂y
end

 f(x) = max(x[1], x[2]) ^ 2 + 3f0 * x[1] + sin(x[2])
# ----- Test run -----
function test()
    x = cu(Float32[1.0, 2.0])
    x = [1.0, 2.0]
    y = zeros(Float32, 2)
    dy = zeros(Float32, 2)
    # f(x) = max(x[1], x[2]) ^ 2 + 3f0 * x[1] + sin(x[2])
    f(x) = x[1] + x[2]
    #  gradient(f, x, y, dy)
    res = ForwardDiff.gradient(f, x)
    println("Gradient via ForwardDiff: ", Array(res))
    # y, dy = forward_ad_on_gpu(x -> f(x), x)
    println("x = ", Array(x))
    println("y = ", Array(y))   # expected [1+2=3, 8+4=12, 27+6=33]
    println("dy = ", Array(dy)) # derivative g'(x)=3x^2 + 2 => [5, 14, 29]
end



end