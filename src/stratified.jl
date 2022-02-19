function default_args(::Type{StratifiedResampling}, intent::Symbol)
    (randomisation = :none,
     order = :none);
end

function generate_ordered_uniforms!(::Type{<: StratifiedResampling},
                                    u::AbstractVector{<: AbstractFloat},
                                    rng::AbstractRNG)
    n = length(u);
    rand!(rng, u);
    for i = eachindex(u)
        @inbounds u[i] = (i - 1.0 + u[i]) / n;
    end
    nothing;
end
