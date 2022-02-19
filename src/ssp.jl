function default_args(::Type{SSPResampling}, intent::Symbol)
    (order = :none, randomisation = :none);
end

function _resample!(res::SSPResampling,
                    ind::AbstractVector{<: Integer},
                    w::AbstractVector{<: AbstractFloat},
                    rng::AbstractRNG)
    FT = float_type(res);

    # Cheap Julia convert of https://github.com/nchopin/particles/blob/master/particles/resampling.py
    n = length(res.ind)
    @assert n == length(w)
    @assert n == length(ind)
    # Aliases for temporary vectors
    nr_children = res.ind; xi = res.aux.w; order = res.o.order
    xi .= n .* w
    nr_children .= floor.(xi)
    xi .-= nr_children
    rand!(rng, res.u)
    #i, j = 1, 2
    i, j = order[1], order[2]
    for k in 1:(n - 1)
        @inbounds delta_i = min(xi[j], one(FT) - xi[i])  # increase i, decr j
        @inbounds delta_j = min(xi[i], one(FT) - xi[j])  # the opposite
        sum_delta = delta_i + delta_j
        # prob we increase xi[i], decrease xi[j]
        pj = (sum_delta > zero(FT)) ? delta_i / sum_delta : zero(FT)
        # sum_delta = 0. => xi[i] = xi[j] = 0.
        if @inbounds res.u[k] < pj  # swap i, j, so that we always inc i
            j, i = i, j
            delta_i = delta_j
        end
        @inbounds if xi[j] < one(FT) - xi[i]
            @inbounds xi[i] += delta_i
            #j = k + 2
            j = (k == n-1) ? n+1 : order[k+2]
        else
            @inbounds xi[j] -= delta_i
            @inbounds nr_children[i] += 1
            #i = k + 2
            i = (k == n-1) ? n+1 : order[k+2]
        end
    end
    allocated = sum(nr_children)
    # due to round-off error accumulation, we may be missing one particle
    if allocated == n - 1
        last_ij = (j == n + 1) ? i : j
        @inbounds nr_children[last_ij] += 1
        allocated += 1
    end
    @assert allocated == n
    k = 1
    for i = 1:n
        @inbounds nr_children[i] == 0 && continue
        @inbounds end_k = k + nr_children[i] - 1
        for j = k:end_k
            @inbounds ind[j] = i
        end
        k = end_k + 1
    end
    nothing
end
