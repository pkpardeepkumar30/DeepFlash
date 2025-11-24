module CubicFuncs

using ..EOS

function cubic_eos_coefficients(P, T, z; model)
    # Compute reduced pressure (˜P) and reduced temperature (˜T)
    RT = model.R * T
    R = model.R
    # @show T, V
    N = sum(z)
    x = z ./ N
    b_mix = EOS.b(x; model)
    a_mix = EOS.a(T, x; model)

    tilde_P = (P * b_mix^2) / a_mix
    tilde_T = (R * T * b_mix) / a_mix
    # rho_tilde = bN/ V
    # Cubic equation coefficients in terms of ˜ρ
    c3 = tilde_P + tilde_T - 1
    c2 = -3 * tilde_P - 2 * tilde_T + 1
    c1 = tilde_P - tilde_T
    c0 = tilde_P

    return [1.0, c2 / c3, c1 / c3, c0 / c3], a_mix, b_mix
end

function solve_cubic_positive_roots(a, b, c)
    Q = (a^2 - 3 * b) / 9
    R = (2 * a^3 - 9 * a * b + 27 * c) / 54
    M = R^2 - Q^3
    single_root = M > 0
    if single_root
        # Single real roots
        S = -sign(R) * (abs(R) + sqrt(M))^(1 / 3)
        if S == 0
            T = 0
        else
            T = Q / S
        end
        return S + T - a / 3
    else
        # Three real roots
        theta = acos(R / sqrt(Q^3))
        r1 = -(2 * sqrt(Q) * cos(theta / 3)) - a / 3
        r2 = -(2 * sqrt(Q) * cos((theta + 2 * pi) / 3)) - a / 3
        r3 = -(2 * sqrt(Q) * cos((theta - 2 * pi) / 3)) - a / 3
        return (r1, r2, r3)
    end
end

function pick_root(roots, P, T, z; model, b_mix, phase = :unknown)
    r_ϵ = b_mix # minimum_allowable_root
    max_r = maximum(roots)
    min_r = minimum((x) -> x > r_ϵ ? x : Inf, roots)
    N = sum(z)
    V = NaN
    V_max = b_mix * N / max_r
    V_min = b_mix * N / min_r
    if min_r == max_r
        r = min_r
    elseif phase == :vapor
        r = min_r
    elseif phase == :liquid
        r = max_r
    else
        function Gibbs(V)
            A_EOS(T, V, z; model) + P * V
        end
        if Gibbs(V_min) < Gibbs(V_max)
            r = min_r
        else
            r = max_r
        end
    end
    return r
end

function get_PR_volume(P, T, z; model, phase = :unknown)
    coeffs, a_mix, b_mix = cubic_eos_coefficients(P, T, z; model)
    roots = solve_cubic_positive_roots(coeffs[2], coeffs[3], coeffs[4])

    r_vap = pick_root(roots, P, T, z; model, b_mix, phase = :vapor)
    r_liq = pick_root(roots, P, T, z; model, b_mix, phase = :liquid)
    r_un = pick_root(roots, P, T, z; model, b_mix, phase = :unknown)

    V(r) = b_mix * sum(z) / r
    V_vap = V(r_vap)
    V_liq = V(r_liq)
    V_un = V(r_un)
    V_un, V_vap, V_liq
end

end