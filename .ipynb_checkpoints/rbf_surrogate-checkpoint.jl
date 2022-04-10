# ------------------------------------------------------------------
# Radial basis function collection
# ------------------------------------------------------------------


struct RBFfun
    θ       # Hyperparameter vector
    ψ       # Radial basis function
    Dρ_ψ    # Derivative of the RBF wrt ρ
    Dρρ_ψ   # Second derivative
    ∇θ_ψ    # Gradient with respect to hypers
end


(rbf :: RBFfun)(ρ) = rbf.ψ(ρ)


function kernel_transformθ(kfun, θ, g, dg; kwargs...)
    base_rbf = kfun(g(θ); kwargs...)
    ∇θ_ψ(ρ) = dg(θ)' * base_rbf.∇θ_ψ(ρ)
    RBFfun(θ, base_rbf.ψ, base_rbf.Dρ_ψ, base_rbf.Dρρ_ψ, ∇θ_ψ)
end


function kernel_scale(kfun, θ; kwargs...)
    s = θ[1]
    base_rbf = kfun(θ[2:end]; kwargs...)
    ψ(ρ)     = s * base_rbf.ψ(ρ)
    Dρ_ψ(ρ)  = s * base_rbf.Dρ_ψ(ρ)
    Dρρ_ψ(ρ) = s * base_rbf.Dρρ_ψ(ρ)
    ∇θ_ψ(ρ)  = vcat([base_rbf.ψ(ρ)], s * base_rbf.∇θ_ψ(ρ))
    RBFfun(θ, ψ, Dρ_ψ, Dρρ_ψ, ∇θ_ψ)
end


function kernel_matern12(θ=[1.0])
    l = θ[1]
    c = 1.0/l
    ψ(ρ) = exp(-c*ρ)
    Dρ_ψ(ρ) = -c*exp(-c*ρ)
    Dρρ_ψ(ρ) = c*c*exp(-c*ρ)
    ∇θ_ψ(ρ) = [exp(-c*ρ)*ρ/l^2]
    return RBFfun(θ, ψ, Dρ_ψ, Dρρ_ψ, ∇θ_ψ)
end

using ForwardDiff
# k(ρ, θ)
function kernel_generic(k, θ=[1.0])
    ψ(ρ) = k(ρ, θ)
    Dρ_ψ(ρ) = ForwardDiff.derivative(ψ, ρ)
    Dρρ_ψ(ρ) = ForwardDiff.derivative(Dρ_ψ, ρ)
    ∇θ_ψ(ρ) = ForwardDiff.gradient(θ->k(ρ, θ), θ)
    return RBFfun(θ, ψ, Dρ_ψ, Dρρ_ψ, ∇θ_ψ)
end

function kernel_matern52_generic(θ=[1.0])
    function k(ρ, θ)  # 1+s+s^2/3
        l = θ[1]
        c = sqrt(5.0)/l
        s = c*ρ
        (1+s*(1+s/3))*exp(-s)
    end
    kernel_generic(k, θ)
end


function kernel_matern32(θ=[1.0])
    l = θ[1]
    c = sqrt(3.0)/l

    function ψ(ρ)
        s = c*ρ
        (1+s)*exp(-s)
    end

    function Dρ_ψ(ρ)
        s = c*ρ
        -s*exp(-s)*c
    end

    function Dρρ_ψ(ρ)
        s = c*ρ
        (-1+s)*exp(-s)*c*c
    end

    function ∇θ_ψ(ρ)
        s = c*ρ
        [-s*exp(-s)*(-s/l)]
    end

    return RBFfun(θ, ψ, Dρ_ψ, Dρρ_ψ, ∇θ_ψ)
end


function kernel_matern52(θ=[1.0])
    l = θ[1]
    c = sqrt(5.0)/l

    function ψ(ρ)  # 1+s+s^2/3
        s = c*ρ
        (1+s*(1+s/3))*exp(-s)
    end

    function Dρ_ψ(ρ)  # (1+2s/3)-(1+s+s^2/3) = -s/3-s^2/3
        s = c*ρ
        -s*(1+s)*exp(-s)*c/3
    end

    function Dρρ_ψ(ρ)  # (-1/3-2s/3) - (-s/3-s^2/3) = -1/3-s/3+s^2/3
        s = c*ρ
        (-1+s*(-1+s))*exp(-s)*c*c/3
    end

    function ∇θ_ψ(ρ)
        s = c*ρ
        [-s*(1+s)*exp(-s)*(-s/l/3)]
    end

    return RBFfun(θ, ψ, Dρ_ψ, Dρρ_ψ, ∇θ_ψ)
end


# Squared exponential kernel
function kernel_SE(θ=[1.0])
    l = θ[1]
    c = 1.0/l

    function ψ(ρ)
        s = c*ρ
        exp(-s*s/2)
    end

    function Dρ_ψ(ρ)
        s = c*ρ
        -s*exp(-s*s/2)*c
    end

    function Dρρ_ψ(ρ)
        s = c*ρ
        (-1+s*s)*exp(-s*s/2)*c*c
    end

    function ∇θ_ψ(ρ)
        s = c*ρ
        [-s*exp(-s*s/2)*(-s/l)]
    end

    return RBFfun(θ, ψ, Dρ_ψ, Dρρ_ψ, ∇θ_ψ)
end


# Inverse multiquadric kernel
function kernel_invmq(θ=[1.0])
    l = θ[1]
    c = 1.0/l

    function ψ(ρ)
        s = c*ρ
        1/sqrt(1+s*s)
    end

    function Dρ_ψ(ρ)
        s = c*ρ
        -c*s/sqrt(1+s*s)^3
    end

    function Dρρ_ψ(ρ)
        s = c*ρ
        ss = s*s
        c*c*(2*ss-1)/sqrt(1+ss)^5
    end

    function ∇θ_ψ(ρ)
        s = c*ρ
        ss = s*s
        [ss/l/sqrt(1+ss)^3]
    end

    return RBFfun(θ, ψ, Dρ_ψ, Dρρ_ψ, ∇θ_ψ)
end


# Simple multiquadric kernel -- Conditionally positive definite of order 1
# More general case looks like (-1)^m (1+(r/l)^2)^(β/2), with order
# m = ceil(β/2) -- this is the β=1 case.
function kernel_mq(θ=[1.0])
    l = θ[1]
    c = 1.0/l

    function ψ(ρ)
        s = c*ρ
        sqrt(1+s*s)
    end

    function Dρ_ψ(ρ)
        s = c*ρ
        c*s/sqrt(1+s*s)
    end

    function Dρρ_ψ(ρ)
        s = c*ρ
        c*c/sqrt(1+s*s)^3
    end

    function ∇θ_ψ(ρ)
        s = c*ρ
        ss = s*s
        [-ss/sqrt(1+ss)/l]
    end

    return RBFfun(θ, ψ, Dρ_ψ, Dρρ_ψ, ∇θ_ψ)
end


# β should not be an even integer, conditionally pos def with order ceil(β/2)
# Most common case is the cubic spline (β = 3)
function kernel_poly(θ=[]; β=3)
    m = ceil(β/2)
    s = (-1)^m
    ψ(ρ) = s*ρ^β
    Dρ_ψ(ρ) = s*β*ρ^(β-1)
    Dρρ_ψ(ρ) = s*β*(β-1)*ρ^(β-2)
    ∇θ_ψ(ρ) = []
    return RBFfun([], ψ, Dρ_ψ, Dρρ_ψ, ∇θ_ψ)
end


# Conditionally positive definite with order k+1
# Most common case is the thin plate splines (k = 1)
function kernel_polylog(θ=[]; k=1)
    m = k+1
    s = (-1)^m
    kk = 2*k
    ψ(ρ) = s*ρ^(kk-1)*xlogx(ρ)
    Dρ_ψ(ρ) = s*ρ^(kk-2)*( kk * xlogx(ρ) + ρ )
    Dρρ_ψ(ρ) = s*ρ^(kk-3)*( kk*(kk-1) * xlogx(ρ) + (2*kk-1)*ρ )
    ∇θ_ψ(ρ) = []
    return RBFfun([], ψ, Dρ_ψ, Dρρ_ψ, ∇θ_ψ)
end


function eval_k(rbf :: RBFfun, r)
    rbf(norm(r))
end


function eval_∇k(rbf :: RBFfun, r)
    ρ = norm(r)
    if ρ == 0
        return 0*r
    end
    u = r/ρ
    rbf.Dρ_ψ(ρ)*u
end


function eval_Hk(rbf :: RBFfun, r)
    ρ = norm(r)
    if ρ > 0
        u = r/ρ
        Dψr = rbf.Dρ_ψ(ρ)/ρ
        D2ψ = rbf.Dρρ_ψ(ρ)
        return (D2ψ-Dψr)*u*u' + Dψr*I
    else
#         return (rbf.θ[1]^-1) * Matrix(I, length(r), length(r))
        # Should this negative sign be in front?
        return rbf.Dρρ_ψ(ρ) * Matrix(I, length(r), length(r)) 
    end
end


function eval_KXX(rbf :: RBFfun, X)
    d, N = size(X)
    KXX = zeros(N, N)
    ψ0 = rbf(0)
    for j = 1:N
        KXX[j,j] = ψ0
        for i = j+1:N
            Kij = rbf(norm(X[:,i]-X[:,j]))
            KXX[i,j] = Kij
            KXX[j,i] = Kij
        end
    end
    KXX + 1e-8*I
end


"""
(TODO) Might want to change the naming to something that suggests 
this evaluates the covariance between pairs and not across all
pairs.
"""
function eval_Dk(rbf :: RBFfun, r; D)
    K = eval_k(rbf, r)
    ∇K = eval_∇k(rbf, r)
    HK = eval_Hk(rbf, r)
    return [K   ∇K'
            -∇K -HK] 
end


"""
"""
function eval_DKxX(rbf :: RBFfun, x, X; D)
    M, N = size(X)
    KxX = eval_Dk(rbf, x-X[:,1], D=D)
    for j = 2:N
        KxX = hcat(
            KxX,
            eval_Dk(rbf, x-X[:,j], D=D)
        )
    end
    KxX
end

"""
    eval_DKXX(rbf, X, D=D)

Constructs a covariance matrix between observations and gradients
where the block entries are the covariances between observations
and gradient observations

Kij = [k(xi,xj) ∇K(xi,xj)
       ∇k(xi,xj) Hk(xi,xj)]
KXX = [K11 ... K1N
       .   ...  .
       .   ...  .
       KN1 ... KNN]
"""
function eval_DKXX(rbf :: RBFfun, X; D)
    M, N = size(X)
    nd1 = N*(D+1)
    K = zeros(nd1, nd1)
    r0 = zeros(M)
    ψ0 = eval_Dk(rbf, r0, D=D)
    s(i) = (i-1)*(D+1)+1
    e(i) = s(i)+D
    
    for i = 1:N
        # Starting indices
        si, ei = s(i), e(i)
        K[si:ei, si:ei] = ψ0
        # Reduce computations by leveraging symmetric structure of
        # covariance matrix
        for j = i+1:N
            # Row remains stationary as columns (j=i+1) vary as a function
            # of the row index (i)
            sj, ej = s(j), e(j)
            Kij = eval_Dk(rbf, X[:,i]-X[:,j], D=D)
            K[si:ei, sj:ej] = Kij
            K[sj:ej, si:ei] = Kij'
        end
    end
    
    K
end


"""
(DEPRECATED FOR USE IN ROLLOUT)
    eval_KXXD(rbf, X, D=D)

Constructs a covariance matrix between observations and gradients
where the first N rows correspond to non-gradient observations and
subsequent rows are gradient covariances.

KXX = [K(X,X) ∇K(X,X)
       ∇K(X,X) HK(X,X)]

TODO: Needs to be corrected. The covariance matrix constructed isn't correct.
"""
function eval_KXXD(rbf :: RBFfun, X; D)
    M, N = size(X)
    nd1 = N*(D+1)
    K = zeros(nd1, nd1)
    r0 = zeros(M)
    K[1:N,1:N] = eval_KXX(rbf, X)
    Hk0 = eval_Hk(rbf, r0)
    ∇k0 = eval_∇k(rbf, r0)
    e(j) = N+(D*(j-1))+1
    h(i) = (i-1)*D + (N-i)+1 + i
    
    for i = 1:N
        ei, hi = e(i), h(i)
        K[ei:ei+D-1, ei:ei+D-1] = Hk0
        # Do i == j computations here as well then change iterator j=1:N to
        # j=i+1:N and use the symmetry of the matrix to update covariance K
        for j = 1:N
            # Compute the covariance between f(xi) and ∂f(xj)
            ej = e(j)
            rij = X[:,i]-X[:,j]
            kij = eval_∇k(rbf, rij)
            Hkij = eval_Hk(rbf, rij)
            K[i, ej:ej+D-1] = kij'
            K[ej:ej+D-1, i] = kij
            K[hi:hi+D-1, ej:ej+D-1] = Hkij
        end
    end
    
    K
end 


function eval_DΘ_KXX(rbf :: RBFfun, X, δθ)
    d, N = size(X)
    δKXX = zeros(N, N)
    δψ0 = rbf.∇θ_ψ(0)' * δθ
    for j = 1:N
        δKXX[j,j] = δψ0
        for i = j+1:N
            δKij = rbf.∇θ_ψ(norm(X[:,i]-X[:,j]))' * δθ
            δKXX[i,j] = δKij
            δKXX[j,i] = δKij
        end
    end
    δKXX
end


function eval_KXY(rbf :: RBFfun, X, Y)
    d, M = size(X)
    d, N = size(Y)
    KXY = zeros(M, N)
    for j = 1:N
        for i = 1:M
            KXY[i,j] = rbf(norm(X[:,i]-Y[:,j]))
        end
    end
    KXY
end


function eval_KxX(rbf :: RBFfun, x, X)
    d, N = size(X)
    KxX = zeros(N)
    for i = 1:N
        KxX[i] = rbf(norm(x-X[:,i]))
    end
    KxX
end


function eval_∇KxX(rbf :: RBFfun, x, X)
    d, N = size(X)
    ∇KxX = zeros(d, N)
    for j = 1:N
        r = x-X[:,j]
        ρ = norm(r)
        if ρ > 0.
            ∇KxX[:,j] = rbf.Dρ_ψ(ρ) * r/ρ
        end
    end
    ∇KxX
end


function eval_δKXX(rbf :: RBFfun, X, δX)
    d, N = size(X)
    δKXX = zeros(N, N)
    for j = 1:N
        for i = j+1:N
            δKij = eval_∇k(rbf, X[:,i]-X[:,j])'*(δX[:,i]-δX[:,j])
            δKXX[i,j] = δKij
            δKXX[j,i] = δKij
        end
    end
    δKXX
end


function eval_δKXY(rbf :: RBFfun, X, Y, δX, δY)
    d, N = size(X)
    d, M = size(Y)
    δKXY = zeros(N, M)
    for i = 1:N
        for j = 1:M
            δKXY[i,j] = eval_∇k(rbf, X[:,i]-Y[:,j])'*(δX[:,i]-δY[:,j])
        end
    end
    δKXY
end


function eval_δKxX(rbf :: RBFfun, x, X, δX)
    d, N = size(X)
    δKxX = zeros(N)
    for j = 1:N
        δKxX[j] = eval_∇k(rbf, x-X[:,j])'*(-δX[:,j])
    end
    δKxX
end


function eval_δ∇KxX(rbf :: RBFfun, x, X, δX)
    d, N = size(X)
    δ∇KxX = zeros(d, N)
    for j = 1:N
        δ∇KxX[:,j] = eval_Hk(rbf, x-X[:,j])*(-δX[:,j])
    end
    δ∇KxX
end


# ------------------------------------------------------------------
# Operations on GP/RBF surrogates
# ------------------------------------------------------------------

struct RBFsurrogate
    ψ :: RBFfun
    X :: Matrix{Float64}
    K :: Matrix{Float64}
    fK :: Cholesky
    y :: Vector{Float64}
    c :: Vector{Float64}
end


"""
RBF surrogate with derivative observations intertwined in
y, such that y = [y0 ∇y0 ... yn ∇yn], where ∇yi is a
d-dimensional vector of gradient observations.
"""
struct DRBFsurrogate
    ψ :: RBFfun
    X :: Matrix{Float64}
    K :: Matrix{Float64}
    fK :: Cholesky
    y :: Vector{Float64}
    c :: Vector{Float64}
    D :: Int8
end


function fit_surrogate(ψ :: RBFfun, X, f)
    d, N = size(X)
    K = eval_KXX(ψ, X)
    fK = cholesky(K)
    y = [f(X[:,j]) for j=1:N]
    c = fK\y
    RBFsurrogate(ψ, X, K, fK, y, c)
end

function fit_surrogate(ψ :: RBFfun, X, f)
    d, N = size(X)
    K = eval_KXX(ψ, X)
    fK = cholesky(Hermitian(K))
    y = f
    c = fK\y
    RBFsurrogate(ψ, X, K, fK, y, c)
end

function fit_surrogate(ψ :: RBFfun, X, sur::RBFsurrogate, f)
    d, N = size(X)
    K = eval_KXX(ψ, X)
    fK = cholesky(K)
    y = reshape([sur.y... f], size(K, 1))
    c = fK\y
    RBFsurrogate(ψ, X, K, fK, y, c)
end


"""
Naive update of our surrogate without using smart
matrix updating. This could be prohibitively
expensive
"""
function update_surrogate(s :: RBFsurrogate, Xnew, ynew)
    X = hcat(s.X, Xnew)
    y = vcat(s.y, ynew)
    KXY = eval_KXY(s.ψ, s.X, Xnew)
    K = [s.K  KXY
           KXY' eval_KXX(s.ψ, Xnew)]
    fK = cholesky(K)
    c = fK\y
    RBFsurrogate(s.ψ, X, K, fK, y, c)
end


function fit_dsurrogate(ψ :: RBFfun, X, f, ∇f; D)
    d, N = size(X)
    y = Array{Float64, 1}()
    K = eval_DKXX(ψ, X, D=D)
    fK = cholesky(K)
    for j = 1:N
        push!(y, [f(X[:,j]), ∇f(X[:,j])...]...)
    end
    c = fK\y
    DRBFsurrogate(ψ, X, K, fK, y, c, D)
end


function log_likelihood(s :: RBFsurrogate)
    n = size(s.X)[2]
    -s.y'*s.c/2 - sum(log.(diag(s.fK.L))) - n*log(2π)/2
end


function δlog_likelihood(s :: RBFsurrogate, δθ)
    δK = eval_DΘ_KXX(s.ψ, s.X, δθ)
    (s.c'*δK*s.c - tr(s.fK\δK))/2
end


function ∇log_likelihood(s :: RBFsurrogate)
    nθ = length(s.ψ.θ)
    δθ = zeros(nθ)
    ∇L = zeros(nθ)
    for j = 1:nθ
        δθ[:] .= 0.0
        δθ[j] = 1.0
        ∇L[j] = δlog_likelihood(s, δθ)
    end
    ∇L
end

"""
NOTE TO SELF: Variance optimized. Assuming that the scale factor
is optimized
"""
function log_likelihood_v(s :: RBFsurrogate)
    n = size(s.X)[2]
    α = s.y'*s.c/n
    -n/2*(1.0 + log(α) + log(2π)) - sum(log.(diag(s.fK.L)))
end


function δlog_likelihood_v(s :: RBFsurrogate, δθ)
    n = size(s.X)[2]
    c = s.c
    y = s.y
    δK = eval_DΘ_KXX(s.ψ, s.X, δθ)
    n/2*(c'*δK*c)/(c'*y) - tr(s.fK\δK)/2
end


function ∇log_likelihood_v(s :: RBFsurrogate)
    nθ = length(s.ψ.θ)
    δθ = zeros(nθ)
    ∇L = zeros(nθ)
    for j = 1:nθ
        δθ[:] .= 0.0
        δθ[j] = 1.0
        ∇L[j] = δlog_likelihood_v(s, δθ)
    end
    ∇L
end


function optimize_hypers(θ, kernel_constructor, X, f;
                         Δ=1.0, nsteps=100, rtol=1e-6, Δmax=Inf,
                         monitor=(x, rnorm, Δ)->nothing)
    θ = copy(θ)
    d = length(θ)
    Lref = log_likelihood(fit_surrogate(kernel_constructor(θ), X, f))
    gsetup(θ) = fit_surrogate(kernel_constructor(θ), X, f)
    g(s) = log_likelihood(s)/Lref
    ∇g(s) = ∇log_likelihood(s)/Lref

    return tr_SR1(θ, gsetup, g, ∇g, Matrix{Float64}(I,d,d),
                  Δ=Δ, nsteps=nsteps, rtol=rtol, Δmax=Δmax,
                  monitor=monitor)
end

function optimize_hypers(θ, kernel_constructor, X, sur::RBFsurrogate, f;
                         Δ=1.0, nsteps=100, rtol=1e-6, Δmax=Inf,
                         monitor=(x, rnorm, Δ)->nothing)
    θ = copy(θ)
    d = length(θ)
    Lref = log_likelihood(fit_surrogate(kernel_constructor(θ), X, sur, f))
    gsetup(θ) = fit_surrogate(kernel_constructor(θ), X, sur, f)
    g(s) = log_likelihood(s)/Lref
    ∇g(s) = ∇log_likelihood(s)/Lref

    return tr_SR1(θ, gsetup, g, ∇g, Matrix{Float64}(I,d,d),
                  Δ=Δ, nsteps=nsteps, rtol=rtol, Δmax=Δmax,
                  monitor=monitor)
end


function optimize_hypers_v(θ, kernel_constructor, X, f;
                           Δ=1.0, nsteps=100, rtol=1e-6, Δmax=Inf,
                           monitor=(x, rnorm, Δ)->nothing)
    θ = copy(θ)
    d = length(θ)

    Lref = log_likelihood_v(fit_surrogate(kernel_constructor(θ), X, f))
    gsetup(θ) = fit_surrogate(kernel_constructor(θ), X, f)
    g(s) = log_likelihood_v(s)/Lref
    ∇g(s) = ∇log_likelihood_v(s)/Lref

    θ0, s = tr_SR1(θ, gsetup, g, ∇g, Matrix{Float64}(I,d,d),
                   Δ=Δ, nsteps=nsteps, rtol=rtol, Δmax=Δmax,
                   monitor=monitor)
    α = s.c'*s.y/(size(s.X)[2])
    θ = vcat([α], θ0)
    rbf = kernel_scale(kernel_constructor, θ)
    θ, fit_surrogate(rbf, X, f)
end


function eval(s :: RBFsurrogate, x, ymin)
    sx = LazyStruct()
    set(sx, :s, s)
    set(sx, :x, x)
    set(sx, :ymin, ymin)

    d, N = size(s.X)

    sx.kx  = () -> eval_KxX(s.ψ, x, s.X)
    sx.∇kx = () -> eval_∇KxX(s.ψ, x, s.X)

    sx.μ  = () -> sx.kx' * s.c
    sx.∇μ = () -> sx.∇kx * s.c
    sx.Hμ = function()
        H = zeros(d, d)
        for j = 1:N
            H += s.c[j] * eval_Hk(s.ψ, x-s.X[:,j])
        end
        H
    end

    sx.w  = () -> s.fK\sx.kx
    sx.Dw = () -> s.fK\(sx.∇kx')
    sx.∇w = () -> sx.Dw'
    sx.σ  = () -> sqrt(s.ψ(0)-sx.kx'*sx.w)
    sx.∇σ = () -> -(sx.∇kx * sx.w)/sx.σ
    sx.Hσ = function()
        H = -sx.∇σ * sx.∇σ' - sx.∇kx * sx.Dw
        w = sx.w
        for j = 1:N
            H -= w[j]*eval_Hk(s.ψ, x-s.X[:,j])
        end
        H /= sx.σ
        H
    end

    sx.z =  () -> (ymin-sx.μ)/sx.σ
    sx.∇z = () -> (-sx.∇μ-sx.z*sx.∇σ)/sx.σ
    sx.Hz = () -> Hermitian((-sx.Hμ + (sx.∇μ*sx.∇σ' + sx.∇σ*sx.∇μ')/sx.σ -
        sx.z*(sx.Hσ-2/sx.σ*sx.∇σ*sx.∇σ'))/sx.σ)

    sx.Φz = () -> normcdf(sx.z)
    sx.ϕz = () -> normpdf(sx.z)
    sx.g  = () -> sx.z * sx.Φz + sx.ϕz

    sx.EI  = () -> sx.σ*sx.g
    sx.∇EI = () -> sx.g*sx.∇σ + sx.σ*sx.Φz*sx.∇z
    sx.HEI = () -> Hermitian(sx.Hσ*sx.g +
        sx.Φz*(sx.∇σ*sx.∇z' + sx.∇z*sx.∇σ' + sx.σ*sx.Hz) +
        sx.σ*sx.ϕz*sx.∇z*sx.∇z')

    # Optimizing expected improvement is tricky in regions where EI is
    # exponentially small -- we have to have a reasonable starting
    # point to get going.  For negative z values, we rewrite g(z) = G(-z)
    # in terms of the Mills ratio R(z) = Q(z)/ϕ(z) where Q(z) is the
    # complementary CDF.  Then G(z) = H(z) ϕ(z) where H(z) = 1-zR(z).
    # For sufficiently large R, the Mills ratio can be computed by a
    # generalized continued fraction due to Laplace:
    #   R(z) = 1/z+ 1/z+ 2/z+ 3/z+ ...
    # We rewrite this as
    #   R(z) = W(z)/(z W(z)+1) where W(z) = z + 2/z+ 3/z+ ...
    # Using this definition, we have
    #   H(z) = 1/(1+z W(z))
    #   log G(z) = -log(w+zW(z)) + normlogpdf(z)
    #   [log G(z)]' = -Q(z)/G(z) = -W(z)
    #   [log G(z)]'' = 1 + zW(z) - W(z)^2
    # The continued fraction doesn't converge super-fast, but that is
    # almost surely fine for what we're doing here.  If needed, we could
    # do a similar manipulation to get an optimized rational approximation
    # to W from Cody's 1969 rational approximations to erfc.  Or we could
    # use a less accurate approximation -- the point of getting the tails
    # right is really to give us enough inormation to climb out of the flat
    # regions for EI.

    sx.WQint = function()
        z = -sx.z
        u = z
        for k = 500:-1:2
            u = k/(z+u)
        end
        z + u
    end

    sx.logEI = function()
        z = sx.z
        if z < -1.0
            W = sx.WQint
            log(sx.σ) - log(1-z*W) + normlogpdf(z)
        else
            log(sx.σ) + log(sx.g)
        end
    end

    sx.∇logEI = function()
        z = sx.z
        if z < -1.0
            sx.∇σ/sx.σ + sx.WQint*sx.∇z
        else
            sx.∇σ/sx.σ + sx.Φz/sx.g*sx.∇z
        end
    end

    sx.HlogEI = function()
        z = sx.z
        if z < -1.0
            W = sx.WQint
            HlogG = 1.0-(z+W)*W
            Hermitian( (sx.Hσ - sx.∇σ*sx.∇σ'/sx.σ)/sx.σ +
                HlogG*sx.∇z*sx.∇z' + W*sx.Hz)
        else
            W = sx.Φz/sx.g
            HlogG = (sx.ϕz-sx.Φz*sx.Φz/sx.g)/sx.g
            Hermitian( (sx.Hσ - sx.∇σ*sx.∇σ'/sx.σ)/sx.σ +
                HlogG*sx.∇z*sx.∇z' + W*sx.Hz)
        end
    end

    sx
end


eval(s :: RBFsurrogate, x) = eval(s, x, minimum(s.y))
(s :: RBFsurrogate)(x) = eval(s, x)


# ------------------------------------------------------------------
# Operations on GP/RBF surrogate derivatives wrt node positions
# ------------------------------------------------------------------

struct δRBFsurrogate
    s :: RBFsurrogate
    X :: Matrix{Float64}
    K :: Matrix{Float64}
    y :: Vector{Float64}
    c :: Vector{Float64}
end


function fit_δsurrogate(s, δX, ∇f)
    d, N = size(s.X)
    δK = eval_δKXX(s.ψ, s.X, δX)
    δy = [∇f(s.X[:,j])'*δX[:,j] for j=1:N]
    δc = s.fK\(δy-δK*s.c)
    δRBFsurrogate(s, δX, δK, δy, δc)
end


"""
Updates perturbed surrogate given the previously
updated surrogate
"""
function update_δsurrogate(us :: RBFsurrogate, δs :: δRBFsurrogate, Xnew, ∇ynew)
    d, N = size(us.X)
    δX = hcat(δs.X, Xnew)
    δK = eval_δKXX(us.ψ, us.X, δX)
    δy = vcat(δs.y, ∇ynew'*δX[:,end])
    δc = us.fK\(δy-δK*us.c)
    δRBFsurrogate(us, δX, δK, δy, δc)
end


function fit_surrogates(ψ :: RBFfun, X, f, ∇f)
    d, N = size(X)
    δX = rand(d, N)
    s = fit_surrogate(ψ, X, f)
    δs = fit_δsurrogate(s, δX, ∇f)
    return s, δs
end


function eval(δs :: δRBFsurrogate, sx, δymin)
    δsx = LazyStruct()
    set(δsx, :sx, sx)
    set(δsx, :δymin, δymin)

    s = δs.s
    x = sx.x
    d, N = size(s.X)

    δsx.kx  = () -> eval_δKxX(s.ψ, x, s.X, δs.X)
    δsx.∇kx = () -> eval_δ∇KxX(s.ψ, x, s.X, δs.X)

    δsx.μ  = () -> δsx.kx'*s.c + sx.kx'*δs.c
    δsx.∇μ = () -> δsx.∇kx*s.c + sx.∇kx*δs.c

    δsx.σ  = () -> (-2*δsx.kx'*sx.w + sx.w'*(δs.K*sx.w)) / (2*sx.σ)
    δsx.∇σ = () -> (-δsx.∇kx*sx.w - sx.∇w*δsx.kx + sx.∇w*(δs.K*sx.w)-δsx.σ*sx.∇σ)/sx.σ

    δsx.z  = () -> (δymin-δsx.μ-sx.z*δsx.σ)/sx.σ
    δsx.∇z = () -> (-δsx.∇μ-sx.∇z*δsx.σ-sx.z*δsx.∇σ)/sx.σ - δsx.z/sx.σ*sx.∇σ

    δsx.EI  = () -> sx.g*δsx.σ + sx.σ*sx.Φz*δsx.z
    δsx.∇EI = () -> δsx.∇σ*sx.g + sx.Φz*(δsx.z*sx.∇σ + δsx.σ*sx.∇z + sx.σ*δsx.∇z) + sx.ϕz*δsx.z*sx.∇z

    δsx
end


function eval(δs :: δRBFsurrogate, sx)
    ymin, j_ymin = findmin(δs.s.y)
    δymin = δs.y[j_ymin]
    eval(δs, sx, δymin)
end


(δs :: δRBFsurrogate)(sx) = eval(δs, sx)


# ------------------------------------------------------------------

"""
    tr_newton(s, x0, α_key, ∇α_key, Hα_key;
              Δ=1.0, nsteps=100, rtol=1e-6, Δmax=Inf,
              monitor=(x, rnorm, Δ)->nothing)

Run a trust region Newton iteration for maximizing an acquisition
function α for the surrogate `s`.  The name of the acquisition
function and its first and second derivatives are given by `α_key`,
`∇α_key`, and `Hα_key`.
"""
function tr_newton(s :: RBFsurrogate, x0,
                   α_key :: Symbol, ∇α_key :: Symbol, Hα_key :: Symbol;
                   Δ=1.0, nsteps=100, rtol=1e-6, Δmax=Inf,
                   monitor=(x, rnorm, Δ)->nothing)
    ymin = minimum(s.y)
    fsetup(x) = eval(s, x, ymin)
    f(sx) = -Base.getproperty(sx, α_key)
    ∇f(sx) = -Base.getproperty(sx, ∇α_key)
    Hf(sx) = -Base.getproperty(sx, Hα_key)
    return tr_newton(x0, fsetup, f, ∇f, Hf;
                     Δ=Δ, nsteps=nsteps, rtol=rtol, Δmax=Δmax, monitor=monitor)
end


function tr_SR1(s :: RBFsurrogate, x0,
                α_key :: Symbol, ∇α_key :: Symbol;
                Δ=1.0, nsteps=100, rtol=1e-6, Δmax=Inf,
                monitor=(x, rnorm, Δ)->nothing)
    d = length(x0)
    ymin = minimum(s.y)
    fsetup(x) = eval(s, x, ymin)
    f(sx) = -Base.getproperty(sx, α_key)
    ∇f(sx) = -Base.getproperty(sx, ∇α_key)
    return tr_SR1(x0, fsetup, f, ∇f, Matrix{Float64}(I,d,d);
                  Δ=Δ, nsteps=nsteps, rtol=rtol, Δmax=Δmax, monitor=monitor)
end


function tr_newton_EI(s :: RBFsurrogate, x0;
                      Δ=1.0, nsteps=100, rtol=1e-6, Δmax=Inf,
                      monitor=(x, rnorm, Δ)->nothing)
    tr_newton(s, x0, :EI, :∇EI, :HEI;
              Δ=Δ, nsteps=nsteps, rtol=rtol, Δmax=Δmax, monitor=monitor)
end


function tr_newton_EIh(s :: RBFsurrogate, x0;
                       Δ=1.0, nsteps=100, rtol=1e-6, Δmax=Inf,
                       monitor=(x, rnorm, Δ)->nothing)
    tr_newton(s, x0, :logEI, :∇logEI, :HlogEI;
              Δ=Δ, nsteps=nsteps, rtol=rtol, Δmax=Δmax, monitor=monitor)
end


function tr_SR1_EI(s :: RBFsurrogate, x0;
                      Δ=1.0, nsteps=100, rtol=1e-6, Δmax=Inf,
                      monitor=(x, rnorm, Δ)->nothing)
    tr_SR1(s, x0, :EI, :∇EI;
           Δ=Δ, nsteps=nsteps, rtol=rtol, Δmax=Δmax, monitor=monitor)
end
