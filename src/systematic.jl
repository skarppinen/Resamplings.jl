function default_args(::Type{SystematicResampling}, intent::Symbol)
    (randomisation = intent == :conditional ? :circular : :none,
     order = :partition);
end

function generate_ordered_uniforms!(::Type{<: SystematicResampling},
                                    u::AbstractVector{T},
                                    rng::AbstractRNG) where {T <: AbstractFloat}
    u_ = rand(rng, T);
    _fill_u_systematic!(u, u_);
    nothing;
end

function _fill_u_systematic!(u, u_)
    n = length(u)
    for i = eachindex(u)
        @inbounds u[i] = (i - 1 + u_) / n;
    end
    nothing;
end

@inline function _conditional_resample!(res::SystematicResampling{Randomisation{:circular}},
                                        ind::AbstractVector{<: Integer},
                                        w::AbstractVector{<: AbstractFloat},
                                        k::Integer,
                                        i::Integer,
                                        rng::AbstractRNG)
    _systematic_conditional_resample!(res, ind, w, k, i, rng);
    nothing
end


function _systematic_conditional_resample!(res::Resampling,
                                           ind::AbstractVector{<: Integer},
                                           w::AbstractVector{<: AbstractFloat},
                                           k::Integer,
                                           i::Integer,
                                           rng::AbstractRNG)
    FT = float_type(res);
    n = length(w)
    # Randomise 'U' of Chopin & Singh (2015)
    @inbounds nw = n*w[i]; _nw_ = floor(nw); r = nw - _nw_
    @assert nw > 0
    p = r*(_nw_ + one(FT))/nw
    # Draw u_ assuming w[i] is the first weight
    if rand(rng, FT) < p
        u_ = rand(rng, FT)*r
        ni = Int(_nw_) + 1
    else
        u_ = r + rand(rng, FT)*(one(FT) - r)
        ni = Int(_nw_)
    end
    _fill_u_systematic!(res.u, u_)

    # Shift 'ordered' indices so that w[i] = w[ind[1]]
    s = findfirst(j -> j==i, res.o.order)
    circshift!(ind, res.o.order, 1-s)
    # Do lookup with that order
    _ascending_inv_cdf_lookup!(res.ind, w, res.u, ind)

    # These should always hold:
    #@assert res.ind[1] == i
    #@assert ni == findfirst( j -> j!=i, res.ind) - 1

    # Pick a cyclic permutation, again as in the paper
    c_ = rand(rng, Base.OneTo(ni))
    # ... but now shift so that ind[k] = i ...
    circshift!(ind, res.ind, k -1 - c_ + 1)
    # Make sure that this is always true (which might not be due to numerical reasons)

    ind[k] = i
    nothing
end
