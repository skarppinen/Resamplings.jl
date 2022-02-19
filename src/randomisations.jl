## Randomisation handling.
@inline function _randomise!(r::Resampling{S, Randomisation{:none}},
                             ind::AbstractVector{<: Integer},
                             rng::AbstractRNG) where S
    nothing;
end
@inline function _randomise!(res::Resampling{S, Randomisation{:circular}},
                             ind::AbstractVector{<: Integer},
                             rng::AbstractRNG) where S
    FT = float_type(res);
    n = length(res.o.order);
    r = floor(rand(rng, FT) * n);
    copyto!(res.ind, ind);
    circshift!(ind, res.ind, r);
    nothing
end
@inline function _randomise!(res::Resampling{S, Randomisation{:shuffle}},
                             ind::AbstractVector{<: Integer},
                             rng::AbstractRNG) where S
    shuffle!(rng, ind);
    nothing;
end
