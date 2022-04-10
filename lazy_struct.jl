"""
The `LazyStruct` table is used for functional-style lazy evaluation.  We set
entries to a function that can produce the correct value; each such function is
invoked the first time it is needed, and aftwerward we use a cached value.

For example:
```
s = LazyStruct()
s.x = () -> 2        # Function that returns 2
s.y = () -> s.x * 3  # Function that returns the result of s.x (=2) * 3
println(s.y)         # Calls s.y to get a result; that in turn calls s.x
println(s.y)         # This time, use the cached result (6)
```
"""
struct LazyStruct
    thunks
    values
    LazyStruct() = new(Dict(), Dict())
    LazyStruct(s :: LazyStruct) = new(copy(s.thunks), copy(s.values))
end

function Base.setproperty!(s :: LazyStruct, v :: Symbol, f)
    thunks = getfield(s, :thunks)
    thunks[v] = f
    delete!(getfield(s, :values), v)
end

function Base.getproperty(s :: LazyStruct, v :: Symbol)
    values = getfield(s, :values)
    thunks = getfield(s, :thunks)
    if haskey(values, v)
        return values[v]
    elseif haskey(thunks, v)
        values[v] = thunks[v]()
        return values[v]
    end
    getfield(s, v)
end

function set(s :: LazyStruct, k :: Symbol, v)
    getfield(s, :values)[k] = v
end
