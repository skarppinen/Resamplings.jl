function default_args(::Type{KillingResampling}, intent::Symbol)
    (randomisation = intent == :conditional ? :circular : :none,
     order = :none,
     max_div = true);
end

function _resample!(res::KillingResampling{<: Randomisation, <: Order{:none}},
                    ind::AbstractVector{<: Integer},
                    w::AbstractVector{<: AbstractFloat},
                    rng::AbstractRNG)
    n = length(res.u);
    @assert length(ind) == n && length(w) == n;
    w_max = res.aux.max_div ? maximum(w) : one(eltype(res.u));
    r = 0;
    rand!(rng, res.u);
    for i = eachindex(w)
        @inbounds if res.u[i] <= w[i] / w_max
            ind[i] = i;
        else
            ind[i] = 0;
            r += 1;
        end
    end
    if r > 0
        generate_ordered_uniforms!(MultinomialResampling, view(res.u, 1:r), rng);
        vi = view(res.ind, 1:r);
        _ascending_inv_cdf_lookup!(vi, w, res.u);
        shuffle!(rng, vi);
        k = 0;
        for i = eachindex(w)
            @inbounds if ind[i] == 0
                ind[i] = vi[k += 1];
            end
        end
    end
    w_max;
end

function _conditional_resample!(res::KillingResampling{Randomisation{:circular}, <: Order{:none}},
                                ind::AbstractVector{<: Integer},
                                w::AbstractVector{<: AbstractFloat},
                                k::Integer,
                                i::Integer,
                                rng::AbstractRNG)
    # First unconditional killing:
    FT = float_type(res);
    w_max = _resample!(res, ind, w, rng);
    w_max_inv = one(FT) / w_max;

    # Then draw circular shift offset S:
    n = length(res.u)
    c = one(FT)
    for j = eachindex(w)
        j == i && continue
        @inbounds c_ = (one(FT) .- w[j] * w_max_inv) / n;
        @inbounds res.u[j] = c_
        c -= c_
    end
    #@assert c >= 0
    c = max(zero(FT), c) # Might not be for numerical reasons
    @inbounds res.u[i] = c
    S = _inv_cdf_lookup(res.u, rand(rng, FT));
    circshift!(res.ind, ind, k - S);
    # Finalise by setting the desired index
    copyto!(ind, res.ind);
    ind[k] = i;
    nothing
end
