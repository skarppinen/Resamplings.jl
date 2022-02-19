function _resample!(res::Union{StratifiedResampling, SystematicResampling},
                    ind::AbstractVector{<: Integer},
                    w::AbstractVector{<: AbstractFloat},
                    rng::AbstractRNG = Random.GLOBAL_RNG)
    generate_ordered_uniforms!(typeof(res), res.u, rng);
    _ascending_inv_cdf_lookup!(ind, w, res.u, res.o);
end

"""
Inverse CDF lookup with uniform r.v's in `u` assumed to be ascending in value.
"""
function _ascending_inv_cdf_lookup!(ind::AbstractVector{<: Integer},
                                    p::AbstractVector{<: AbstractFloat},
                                    u::AbstractVector{<: AbstractFloat},
                                    order::AbstractVector{<: Integer} = Base.OneTo(length(p)))
    n = length(ind);
    m = length(p);
    @assert m >= 1;
    K = 1; @inbounds S = p[order[1]];
    for j in Base.OneTo(n)
        @inbounds U = u[j];
        # Find K such that F(K) >= u.
        # K is not reset by assumption of ascending u's.
        while K < m && U > S
            K = K + 1;
            @inbounds S = S + p[order[K]];
        end
        @inbounds ind[j] = order[K];
    end
    nothing;
end

"""
Inverse CDF lookup for one uniform r.v.
"""
function _inv_cdf_lookup(p::AbstractVector{<: AbstractFloat},
                         U::AbstractFloat)
    m = length(p);
    @assert m >= 1
    K = 1; @inbounds S = p[1]
    while K < m && U > S
        K = K + 1       # Note that K is not reset!
        @inbounds S = S + p[K] # S is the partial sum up to K
    end
    return K;
end

"""
Sample one index from 1:length(x) proportional on the weights in `x`.
It is assumed that the weights are normalised to 1.0 and `x` is nonempty.
"""
@inline function _wsample_one(x::AbstractVector{<: AbstractFloat}, rng::AbstractRNG = Random.GLOBAL_RNG)
  u = rand(rng);
  s = zero(eltype(x));
  for i in eachindex(x)
    s += @inbounds x[i];
    if u <= s
      return i;
    end
  end
  length(x);
end


## Misc. functions
function Base.show(io::IO, ::MIME"text/plain", res::Resampling{T, Randomisation{R}, <: Order{O}}) where {T, R, O}
    print(io, string(_prettify_name(T), "Resampling(N = ",
                     length(res.ind), ", Randomisation{:$R}, Order{:$O})"));
end

function Base.show(io::IO, ::MIME"text/plain",
                   res::Union{MultinomialResampling{Randomisation{R}}, KillingResampling{Randomisation{R}}}) where R
    print(io, string(_prettify_name(resampling_type(res)), "Resampling(N = ",
                     length(res.ind), ", Randomisation{:$R})"));
end

"""
Check whether a particular resampling implements conditional resampling with
the specified randomisation and order.

Example: has_conditional(:systematic, :circular, :partition)
"""
function has_conditional(resampling::Symbol, randomisation::Symbol, order::Symbol)
    @assert resampling in _RESAMPLING_CHOICES "invalid resampling, valid are $_RESAMPLING_CHOICES";
    @assert randomisation in _RANDOMISATION_CHOICES "invalid randomisation, valid are $_RANDOMISATION_CHOICES";
    @assert order in _ORDER_CHOICES "invalid order, valid are $_ORDER_CHOICES";
    resampling_type = Resampling{resampling, Randomisation{randomisation}, <: Order{order}};
    hasmethod(_conditional_resample!, Tuple{resampling_type,
                                            AbstractVector{<: Integer},
                                            AbstractVector{<: AbstractFloat},
                                            Integer, Integer, AbstractRNG});
end

function has_conditional(::Resampling{S, Randomisation{R}, <: Order{O}}) where {S, R, O}
    has_conditional(S, R, O);
end

function _prettify_name(resampling::Symbol)
    res_str = string(resampling);
    res_name = if length(res_str) <= 4
        uppercase(res_str);
    else
        uppercasefirst(res_str);
    end
    res_name;
end

"""
Prints the constructors of all valid conditional resamplings.
"""
function list_conditional_resamplings()
    out = Vector{String}(undef, 0);
    msg = "";
    for resampling in _RESAMPLING_CHOICES
        res_name = _prettify_name(resampling);
        for randomisation in _RANDOMISATION_CHOICES
            for order in _ORDER_CHOICES
                !has_conditional(resampling, randomisation, order) && continue;
                if !(resampling in (:multinomial, :killing))
                    msg *= string(res_name, "Resampling", "(N; randomisation = :$randomisation, order = :$order)\n");
                else
                    msg *= string(res_name, "Resampling", "(N; randomisation = :$randomisation)\n");
                end
            end
        end
        if msg != ""
            msg = rstrip(msg, '\n'); # Remove last linebreak.
            msg = string(res_name, " resampling:\n", msg);
            push!(out, msg);
            msg = "";
        end
    end

    if !isempty(out)
        for i in 1:(length(out) - 1)
            out[i] *= "\n";
        end
        println("Valid constructors for conditional resampling:");
        println();
        for o in out
            println(o);
        end
    end
end
