function default_args(::Type{SSSResampling}, intent::Symbol)
    (order = :none, randomisation = intent == :conditional ? :circular : :none);
end

function _resample!(res::SSSResampling,
                    ind::AbstractVector{<: Integer},
                    w::AbstractVector{<: AbstractFloat},
                    rng::AbstractRNG)
    n = length(res.u)
    c = _sss_condition(w)
    if c * n <= 1.0
        # Condition is met, do single-event stuff
        _sss_resample!(ind, w, rng, res.u, res.aux.w, c)
    else
        # Standard systematic.
        generate_ordered_uniforms!(SystematicResampling, res.u, rng);
        _ascending_inv_cdf_lookup!(ind, w, res.u, res.o.order);
    end
    nothing
end

@inline function _sss_condition(w, wm = inv(length(w)))
    mapreduce(w_i -> abs(w_i - wm), +, w) / 2.0;
end

function _sss_resample!(ind::AbstractVector{<: Integer},
                        w::AbstractVector{<: AbstractFloat},
                        rng::AbstractRNG,
                        w_small::AbstractVector{<: AbstractFloat},
                        w_large::AbstractVector{<: AbstractFloat},
                        c::AbstractFloat)
    FT = eltype(w_small);
    ind .= eachindex(ind)
    # Whether there is a resample event:
    if rand(rng) >= c * length(w)
        return nothing;
    end
    # Form weights:
    inv_c = one(FT) / c;
    wm = 1.0 / length(w);
    for i in eachindex(w)
        @inbounds if w[i] > wm
            w_large[i] = (w[i] - wm) * inv_c
            w_small[i] = zero(FT)
        else
            w_small[i] = (wm - w[i]) * inv_c
            w_large[i] = zero(FT)
        end
    end
    k = _inv_cdf_lookup(w_small, rand(rng))
    i = _inv_cdf_lookup(w_large, rand(rng))
    @inbounds ind[k] = i
    nothing;
end

function _conditional_resample!(res::SSSResampling{Randomisation{:circular}}, ind, w, k, i, rng)
    FT = float_type(res);
    n = length(w)
    c = _sss_condition(w)
    if c*n <= one(FT)
        _sss_conditional_resample!(res, ind, w, k, i, rng, c)
    else
        _systematic_conditional_resample!(res, ind, w, k, i, rng)
    end
    nothing
end

function _sss_conditional_resample!(res, ind, w, k, i, rng, c::FT) where {FT}
    w_large = res.u; w_small = res.aux.w

    n = length(w)
    @inbounds wi = w[i]; nw = n*wi
    alpha_i = nw - one(FT)
    r = c * n
    r_prime = (r + alpha_i)/nw
    # Theoretically true, but not numerically:
    #@assert zero(FT) <= r_prime <= one(FT)

    res.ind .= eachindex(res.ind)

    if rand(rng, FT) >= r_prime
        # No resampling event, only apply appropriate circular shift
        offset = k - i
    else
        # Event
        if alpha_i >= zero(FT)
            # Big weight, probability of "duplicate":
            r_pp = 2*alpha_i/(r+alpha_i)
            # Normalising factors for the rest large weights
            #large_norm = n*(one(FT) - r_pp)/(r - alpha_i)
            large_norm = n/(r+alpha_i)
            # ... and small weights
            small_norm = n/r
        else
            # This won't be used anyway
            r_pp = one(FT)
            # Normalising factor for large weights
            large_norm = n/r
            # ...and small weights
            small_norm = n/(r + alpha_i)
        end
        _sss_positive_negative_weights!(w_large, w_small, w, i, r_pp, small_norm, large_norm)
        ell = _inv_cdf_lookup(w_small, rand(rng, FT))
        m = _inv_cdf_lookup(w_large, rand(rng, FT))
        res.ind[ell] = m
        if m == i
            # Duplicated, so res.ind[i] = i and res.ind[ell] = i. Randomize
            if rand(rng, FT) < 0.5
                offset = k-ell
            else
                offset = k-i
            end
        else
            # Not duplicated, so res.ind[i] = i...
            offset = k-i
        end
    end
    circshift!(ind, res.ind, offset)
    @assert ind[k] == i
    nothing
end

function _sss_positive_negative_weights!(w_large, w_small, w, i, r_pp::FT, small_norm, large_norm) where {FT}
    # Mean weight
    wm = one(FT)/length(w)
    for j in eachindex(w)
        @inbounds if w[j] > wm
            w_large[j] = (j == i) ? r_pp : (w[j] - wm)*large_norm
            w_small[j] = zero(FT)
        else
            w_small[j] = (j == i) ? zero(FT) : (wm - w[j])*small_norm
            w_large[j] = zero(FT)
        end
    end
    nothing
end
