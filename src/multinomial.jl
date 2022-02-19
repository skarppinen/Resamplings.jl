function default_args(::Type{MultinomialResampling}, intent::Symbol)
    (randomisation = intent == :conditional ? :shuffle : :none,
     order = :none);
end

function generate_ordered_uniforms!(::Type{<: MultinomialResampling},
                                      u::AbstractVector{T},
                                      rng::AbstractRNG = Random.GLOBAL_RNG) where {T <: AbstractFloat}
    randexp!(rng, u);
    cumsum!(u, u);
    u ./= (u[end] + randexp(T));
    nothing;
end

function _resample!(res::MultinomialResampling{R, <: Order{:none}},
                    ind::AbstractVector{<: Integer},
                    w::AbstractVector{<: AbstractFloat},
                    rng::AbstractRNG = Random.GLOBAL_RNG) where R
    generate_ordered_uniforms!(typeof(res), res.u, rng);
    _ascending_inv_cdf_lookup!(ind, w, res.u, res.o);
end


function _conditional_resample!(res::MultinomialResampling{Randomisation{:shuffle}, <: Order{:none}},
                                ind::AbstractVector{<: Integer},
                                w::AbstractVector{<: AbstractFloat},
                                k::Integer,
                                i::Integer,
                                rng::AbstractRNG)

    _resample!(res, ind, w, rng);
    _randomise!(res, ind, rng);
    ind[k] = i;
    nothing;
end
