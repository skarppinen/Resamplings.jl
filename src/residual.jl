function default_args(::Type{ResidualResampling}, intent::Symbol)
    (order = :none, randomisation = :none);
end

function _resample!(res::ResidualResampling,
                    ind::AbstractVector{<: Integer},
                    w::AbstractVector{<: AbstractFloat},
                    rng::AbstractRNG)
    n = length(ind)
    copyto!(res.aux.w, w);
    res.aux.w .*= n
    k = 0
    for i = 1:n
        @inbounds r_i = floor(res.aux.w[i])
        @inbounds res.aux.w[i] -= r_i
        r_int = Int(r_i)
        if r_int > 0
            @assert k + r_int <= n
            @inbounds ind[(k + 1):(k + r_int)] .= i
            k += r_int
        end
    end
    res.aux.w ./= sum(res.aux.w)
    r = n - k
    if r > 0
        generate_ordered_uniforms!(MultinomialResampling, view(res.u, 1:r), rng);
        _ascending_inv_cdf_lookup!(view(ind, (k + 1):n), res.aux.w, res.u);
    end
    nothing
end
