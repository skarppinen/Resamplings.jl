module Resamplings

export Resampling, Randomisation, Order, resample!, conditional_resample!,
       MultinomialResampling, KillingResampling, StratifiedResampling, SystematicResampling,
       SSSResampling, ResidualResampling, SSPResampling, has_conditional, list_conditional_resamplings

using Random: Random, GLOBAL_RNG, AbstractRNG, randexp!, randexp, shuffle!, rand!;

## - Randomisation type -
const _RANDOMISATION_CHOICES = (:none, :shuffle, :circular);
# R: Symbol type for name of randomisation.
struct Randomisation{R}
    function Randomisation(s::Symbol)
         if !(s in _RANDOMISATION_CHOICES)
             msg = "Randomisation `$s` not supported. Supported randomisations are `$_RANDOMISATION_CHOICES`.";
             throw(ArgumentError(msg));
         end
         new{s}();
    end
end

## - Order type -
const _ORDER_CHOICES = (:none, :sort, :partition);

# O: Symbol type of name of ordering.
# IVec: Vector type for indices.
struct Order{O, IVec <: AbstractVector{<: Integer}}
    order::IVec
    function Order(s::Symbol, n::Integer; IT::DataType = Int)
        if !(s in _ORDER_CHOICES)
            msg = "Order `$s` not supported. Supported orderings are `$_ORDER_CHOICES`.";
            throw(ArgumentError(msg));
        end
        if n <= 1
            msg = "argument `n` should be greater than one.";
            throw(ArgumentError(msg));
        end
        if s == :none
            order = Base.OneTo(IT(n));
        else
            order = IT.(collect(1:n));
        end
        new{s, typeof(order)}(order);
    end
end

## - Backend object for resamplings -
const _RESAMPLING_CHOICES = (:multinomial, :killing, :systematic, :stratified,
                             :residual, :sss, :ssp);

# S: Symbol type for name of resampling.
# O: An ordering object that determines how ordering is handled.
# R: A randomisation object that determines how indices are randomised (after sampling them in order)
# FVec: Type for float vector used.
# IVec: Type for integer vector used.
# A: Some kind of named tuple of additional values, or nothing.
struct Resampling{S,
                  R <: Randomisation,
                  O <: Order,
                  FVec <: AbstractVector{<: AbstractFloat},
                  IVec <: AbstractVector{<: Integer},
                  A <: Union{<: NamedTuple, Nothing}}
    r::R # Object that determines how randomisation is done.
    o::O # Object that determines order of indices.
    u::FVec # Some kind of vector of floats for keeping uniform r.v's.
    ind::IVec # Some kind of vector of indices.
    aux::A # A field for auxiliary data or parameters used by some of the resamplings.
    function Resampling{S}(N::Integer,
                           randomisation::Symbol,
                           order::Symbol,
                           FT::DataType,
                           IT::DataType,
                           aux::Any) where S
        if N <= 1
            msg = "`N` must be >= 2.";
            throw(ArgumentError(msg));
        end
        if !(S in _RESAMPLING_CHOICES)
            msg = "Resampling `$S` not supported. Supported resamplings are `$_RESAMPLING_CHOICES`.";
            throw(ArgumentError(msg));
        end
        r = Randomisation(randomisation);
        o = Order(order, N, IT = IT);
        u = zeros(FT, N); ind = zeros(IT, N);
        new{S, typeof(r), typeof(o), typeof(u), typeof(ind), typeof(aux)}(
            r, o, u, ind, aux
        );
    end
end
@inline float_type(::Resampling{S, R, O, FVec}) where {S, R, O, FVec} = eltype(FVec);
@inline resampling_type(::Resampling{S}) where S = S;

## - Aliases and specific implementations of resamplings -
const MultinomialResampling = Resampling{:multinomial};
const KillingResampling = Resampling{:killing};
const ResidualResampling = Resampling{:residual};
const SSPResampling = Resampling{:ssp};
const SSSResampling = Resampling{:sss};
const StratifiedResampling = Resampling{:stratified};
const SystematicResampling = Resampling{:systematic};
include("multinomial.jl");
include("systematic.jl");
include("stratified.jl");
include("killing.jl");
include("residual.jl");
include("sss.jl");
include("ssp.jl");

## - Some common functionality -
include("randomisations.jl");
include("orderings.jl");
include("common.jl");
include("constructors.jl");


@inline function _check_args_resample(res::Resampling, ind::AbstractVector{<: Integer},
                                      w::AbstractVector{<: AbstractFloat})
    n = length(res.ind);
    @assert n == length(ind) "length of `ind` does not match with number of particles used to construct resampling.";
    @assert n == length(w) "length of `ind` and `w` must match."
    nothing;
end

@inline function _check_args_conditional_resample(res::Resampling, ind::AbstractVector{<: Integer},
                                                  w::AbstractVector{<: AbstractFloat},
                                                  k::Integer, i::Integer)
    _check_args_resample(res, ind, w);
    n = length(res.ind);
    @assert 1 <= k <= n "`k` should be between 1 and `length(ind)`."
    @assert 1 <= i <= n "`i` should be between 1 and `length(ind)`."
    @assert w[i] > 0.0 "w[i] == 0.0, where i = $i. cannot proceed with conditional resampling";
    nothing;
end

## User facing API for calling resamplings. (main functions of the package)
function resample!(res::Resampling,
                   ind::AbstractVector{<: Integer},
                   w::AbstractVector{<: AbstractFloat},
                   rng::AbstractRNG = Random.GLOBAL_RNG)
    _check_args_resample(res, ind, w);
    _set_resample_order!(res.o, w);
    _resample!(res, ind, w, rng);
    _randomise!(res, ind, rng);
end

function conditional_resample!(res::Resampling,
                               ind::AbstractVector{<: Integer},
                               w::AbstractVector{<: AbstractFloat},
                               k::Integer,
                               i::Integer,
                               rng::AbstractRNG = Random.GLOBAL_RNG)
    _check_args_conditional_resample(res, ind, w, k, i);
    _set_resample_order!(res.o, w);
    _conditional_resample!(res, ind, w, k, i, rng);
end



end
