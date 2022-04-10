# https://www.sfu.ca/~ssurjano/optimization.html
# https://en.wikipedia.org/wiki/Test_functions_for_optimization

struct TestFunction
    dim
    bounds
    xopt
    f
    ∇f
end


(f :: TestFunction)(x) = f.f(x)


function tplot(f :: TestFunction)
    if f.dim == 1
        xx = range(f.bounds[1,1], f.bounds[1,2], length=100)
        plot(xx, (x) -> f([x]))
        scatter!([xy[1] for xy in f.xopt], [f(xy) for xy in f.xopt], label="xopt")
    elseif f.dim == 2
        xx = range(f.bounds[1,1], f.bounds[1,2], length=100)
        yy = range(f.bounds[2,1], f.bounds[2,2], length=100)
        plot(xx, yy, (x,y) -> f([x,y]), st=:contour)
        # scatter!([xy[1] for xy in f.xopt], [xy[2] for xy in f.xopt])
        scatter!([f.xopt[1]], [f.xopt[2]], label="xopt")
    else
        error("Can only plot 1- or 2-dimensional TestFunctions")
    end
end


function TestRosenbrock()
    f(xy) = (1-xy[1])^2 + 100*(xy[2]-xy[1]^2)^2
    ∇f(xy) = [-2*(1-xy[1]) - 400*xy[1]*(xy[2]-xy[1]^2), 200*(xy[2]-xy[1]^2)]
    return TestFunction(2, [-5.0 10.0 ; -5.0 10.0 ], [1.0, 1.0], f, ∇f)
end


function TestRastrigin(n)
    f(x) = 10*n + sum(x.^2 - 10*cos.(2π*x))
    ∇f(x) = 2*x + 20π*sin.(2π*x)
    bounds = zeros(n, 2)
    bounds[:,1] .= -5.12
    bounds[:,2] .=  5.12
    xopt = (zeros(n),)
    return TestFunction(n, bounds, xopt, f, ∇f)
end


function TestAckley(d; a=20.0, b=0.2, c=2π)
    
    function f(x)
        nx = norm(x)
        cx = sum(cos.(c*x))
        -a*exp(-b/sqrt(d)*nx) - exp(cx/d) + a + exp(1) - 18.4
    end
    
    function ∇f(x)
        nx = norm(x)
        if nx == 0.0
            return zeros(d)
        else
            cx = sum(cos.(c*x))
            dnx = x/nx
            dcx = -c*sin.(c*x)
            (a*b)/sqrt(d)*exp(-b/sqrt(d)*norm(x))*dnx - exp(cx/d)/d*dcx
        end
    end

    bounds = zeros(d,2)
    bounds[:,1] .= -32.768
    bounds[:,2] .=  32.768
    xopt = (zeros(d),)

    TestFunction(d, bounds, xopt, f, ∇f)
end


function TestSixHump()

    function f(xy)
        x = xy[1]
        y = xy[2]
        xterm = (4.0-2.1*x^2+x^4/3)*x^2
        yterm = (-4.0+4.0*y^2)*y^2
        xterm + x*y + yterm
    end

    function ∇f(xy)
        x = xy[1]
        y = xy[2]
        dxterm = (-4.2*x+4.0*x^3/3)*x^2 + (4.0-2.1*x^2+x^4/3)*2.0*x
        dyterm = (8.0*y)*y^2 + (-4.0+4.0*y^2)*2.0*y
        [dxterm + y, dyterm + x]
    end

    # There's a symmetric optimum
    xopt = ([0.0898, -0.7126], [-0.0898, 0.7126])

    TestFunction(2, [-3.0 3.0 ; -2.0 2.0], xopt, f, ∇f)
end


function TestBranin(; a=1.0, b=5.1/(4*π^2), c=5/π, r=6.0, s=10.0, t=1.0/(8π))
    f(x) = a*(x[2]-b*x[1]^2+c*x[1]-r)^2 + s*(1-t)*cos(x[1]) + s
    ∇f(x) = [2*a*(x[2]-b*x[1]^2+c*x[1]-r)*(-2*b*x[1]+c) - s*(1-t)*sin(x[1]),
             2*a*(x[2]-b*x[1]^2+c*x[1]-r)]
    bounds = [-5.0 10.0 ; 0.0 15.0]
    xopt = ([-π, 12.275], [π, 2.275], [9.42478, 2.475])
    TestFunction(2, bounds, xopt, f, ∇f)
end


function TestGramacyLee()
    f(x) = sin(10π*x[1])/(2*x[1]) + (x[1]-1.0)^4
    ∇f(x) = [5π*cos(10π*x[1])/x[1] - sin(10π*x[1])/(2*x[1]^2) + 4*(x[1]-1.0)^3]
    bounds = zeros(1, 2)
    bounds[1,1] = 0.5
    bounds[1,2] = 2.5
    xopt=([0.548],)
    TestFunction(1, bounds, xopt, f, ∇f)
end

negative(t::TestFunction) = TestFunction(t.dim, t.bounds, t.xopt, x -> -t.f(x), x -> -t.∇f(x))