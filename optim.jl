"""
    solve_tr(g, H, Δ)

Solve the trust region problem of minimizing ``μ(p) = g^T p + p^T H p/2``
subject to ``|p| <= Δ``.  Uses the eigensolver-based approach
of Gander, Golub, and Von Matt; see also the 2017 paper of Adachi,
Iwata, Nakatsukasa, and Takeda.
"""
function solve_tr(g, H, Δ)
    n = length(g)

    # Check interior case
    try
        F = cholesky(H)
        p = -(F\g)
        if norm(p) <= Δ
            return p, false
        end
    catch e
        # Hit this case if Cholesky errors (not pos def)
    end

    # Compute the relevant eigensolve
    w = g/Δ
    M = [H    -I ;
         -w*w' H ]
    λs, V = eigen(M)

    # The right most eigenvalue (always sorted to the end in Julia) is real,
    # and corresponds to the desired λ
    p = sortperm(λs, by=real)
    λs = λs[p]
    V = V[:,p]
    λ = -real(λs[1])
    v = real(V[:,1])
    y2 = v[1:n]
    y1 = v[n+1:end]

    # Check if we are in the hard case (to some tolerance)
    gap = real(λs[2])-real(λs[1])
    if norm(y1) <= 1e-8/sqrt(gap)
        # Hard case -- we punt a little and assume only one null vector
        #  Compute min-norm solution plus a multiple of the null vector.
        v = y2/norm(y2)
        q = -(H+norm(H)/n^2*v*v')\g
        return q + v*sqrt(Δ^2-q'*q), true
    else
        # Standard case -- extract solution from eigenvector
        return -sign(g'*y2) * Δ * y1/norm(y1), true
    end
end


"""
    tr_newton(x0, fsetup, f, ∇f, Hf;
              Δ=1.0, nsteps=100, rtol=1e-6, Δmax=Inf,
              monitor=(x, rnorm, Δ)->nothing)

Run a trust region Newton iteration for optimizing a function f.
We split the evaluation of f into a setup phase (fsetup) and actual
evaluation of f or the gradient and Hessian.  The functions
f, ∇f, and Hf take the return value of fsetup as output.
maximizing an acquisition
function α for the surrogate `s`.  The name of the acquisition
function and its first and second derivatives are given by `α_key`,
`∇α_key`, and `Hα_key`.
"""
function tr_newton(x0, fsetup, f, ∇f, Hf;
                   Δ=1.0, nsteps=100, rtol=1e-6, Δmax=Inf,
                   monitor=(x, rnorm, Δ)->nothing)

    # Compute an intial step
    x = copy(x0)
    sx = fsetup(x)
    ϕx = f(sx)
    gx = ∇f(sx)
    Hx = Hermitian(Hf(sx))
    p, hit_constraint = solve_tr(gx, Hx, Δ)

    for k = 1:nsteps

        # Compute gain ratio for new point and decide to accept or reject
        xnew = x + p
        snew = fsetup(xnew)
        ϕnew = f(snew)
        μdiff = -( gx'*p + (p'*Hx*p)/2 )
        ρ = (ϕx - ϕnew)/μdiff

        # Adjust radius
        if ρ < 0.25
            Δ /= 4.0
        elseif ρ > 0.75 && hit_constraint
            Δ = min(2*Δ, Δmax)
        end

        # Accept if enough gain (and check convergence)
        if ρ > 0.1
            x[:] = xnew
            sx = snew
            ϕx = f(snew)
            gx = ∇f(snew)
            monitor(x, norm(gx), Δ)
            if norm(gx) < rtol
                return x, sx
            end
            Hx = Hermitian(Hf(snew))
        end

        # Otherwise, solve the trust region subproblem for new step
        p, hit_constraint = solve_tr(gx, Hx, Δ)

    end
    return x, sx
end


"""
    tr_sr1(x0, fsetup, f, ∇f, H;
           Δ=1.0, nsteps=100, rtol=1e-6, Δmax=Inf,
           monitor=(x, rnorm, Δ)->nothing)

Run a trust region Newton iteration for maximizing an acquisition
function α for the surrogate `s`.  The name of the acquisition
function and its first and second derivatives are given by `α_key`,
`∇α_key`, and `Hα_key`.
"""
function tr_SR1(x0, fsetup, f, ∇f, H;
                Δ=1.0, nsteps=100, rtol=1e-6, Δmax=Inf,
                monitor=(x, rnorm, Δ)->nothing)

    η = 0.1   # Acceptable gain parameter
    r = 1e-8   # Breakdown threshold

    # Compute an intial step
    x = copy(x0)
    sx = fsetup(x)
    ϕx = f(sx)
    gx = ∇f(sx)

    for k = 1:nsteps

        s, hit_constraint = solve_tr(gx, H, Δ)
        xnew = x + s
        sxnew = fsetup(xnew)
        ϕnew = f(sxnew)
        gnew = ∇f(sxnew)
        y = gnew - gx
        μdiff = -( gx'*s + (s'*H*s)/2 )
        ρ = (ϕx - ϕnew)/μdiff

        # Adjust trust region radius
        if ρ > 0.75
            if norm(s) >= 0.8*Δ
                Δ = min(Δmax, 2*Δ)
            end
        elseif ρ < 0.1
            Δ = Δ/2
        end

        # Update the Hessian approximation
        v = y-H*s
        if abs(s'*v) >= r*norm(s)*norm(v)
            H += (v*v')/(v'*s)
        end

        # Accept step if sufficient improvement
        if ρ > η
            x[:] = xnew
            sx = sxnew
            ϕx = ϕnew
            gx = gnew
        end

        # Monitor
        monitor(x, norm(gx), Δ)

        # Convergence check
        if norm(gx) < rtol
            return x, sx
        end
    end
    return x, sx
end
