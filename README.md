# Resamplings.jl

Resamplings.jl is a Julia package implementing resampling algorithms intended to be used with (conditional) particle filters.
The package aims to provide reasonably fast and easy to use functionality for resampling within performance-critical particle
filtering code.
The implementations of the resamplings are based on 
[[Chopin, Singh, Soto and Vihola; 2022]](https://arxiv.org/abs/2203.10037) and 
[[Karppinen, Singh and Vihola; 2022]](https://arxiv.org/abs/2205.13898).

Currently, the package provides the following resampling algorithms:

* multinomial

* stratified

* killing

* systematic

* residual

* SSS

* SSP

The behaviour of each resampling may additionally be altered with additional options (see "Constructing Resampling objects" below).

All resamplings support _unconditional resampling_ that draws indices
$A^{(1:N)}$ given (normalised, i.e summing to unity) weights $w^{(1:N)}$.
Multinomial, systematic, killing and SSS resamplings also support _conditional resampling_
that draws indices $A^{(-k)} \mid A^k = i$ with (normalised) weights $w^{(1:N)}$,
where $A^{(-k)}$ stands for the indices $A^{(1:N)}$ excluding the $k$th.

## Installation

To install Resamplings.jl, just run the following commands in the Julia REPL:

```
import Pkg
Pkg.add(url = "https://github.com/skarppinen/Resamplings.jl.git")
```

## API

Resamplings.jl exports two main in place functions:

* `resample!(res, ind, w, rng)` does unconditional resampling in place to `ind` given normalised weights `w`.

* `conditional_resample!(res, ind, w, k, i, rng)` does conditional resampling in place to `ind` given `ind[k] = i` and normalised weights `w`.

These functions do not modify `w`. After calling `conditional_resample!`, the condition `ind[k] = i` holds.
The types of the arguments should be as follows:

* The argument `res` should be a subtype of `Resampling`. (see below)

* `ind` should be a subtype of `AbstractVector{<: Integer}`.

* `w` should be a subtype of `AbstractVector{<: AbstractFloat}`.

* `k` and `i` should be subtypes of `Integer`.

* `rng` should be a subtype of `AbstractRNG` from the package `Random`. `rng` defaults to `Random.GLOBAL_RNG`.

Furthermore, `resample!` and `conditional_resample!` assume that:

* (_not checked!_) `w` is normalised.

* (_checked_) the vectors `ind` and `w` are both of length `N >= 2`. `N` must match with the number of particles
used to construct the resampling (see "Constructing Resampling objects" below).
An `AssertionError` is raised if either of these conditions does not hold.

* (_checked_) $i, k \in \{1:N\}$. An `AssertionError` is raised if either of these does not hold.

Additionally, `conditional_resample!` assumes (and checks) that `w[i]` is strictly positive.
Attempting to call `conditional_resample!` for resamplings not implementing conditional resampling raises a `MethodError`.
The call `list_conditional_resamplings()` may be used to print constructors for resamplings that implement conditional resampling (see also below).
The function `has_conditional` can be called on a Resampling object to check whether it can be used for conditional resampling.

## Constructing Resampling objects

Resamplings may be constructed with the following kind of syntax:
```
res_mult = Resampling{:multinomial}(10);
res_strat = Resampling{:stratified}(128);
```
where the numbers refer to the numbers of particles used.

Resamplings.jl also provides the following aliases to refer to each resampling:

* `MultinomialResampling === Resampling{:multinomial}`

* `StratifiedResampling === Resampling{:stratified}`

* `KillingResampling === Resampling{:killing}`

* `SystematicResampling === Resampling{:systematic}`

* `ResidualResampling === Resampling{:residual}`

* `SSSResampling === Resampling{:sss}`

* `SSPResampling === Resampling{:ssp}`

That is, for example, to construct an object for systematic resampling, the constructor `SystematicResampling(N)`
may be used instead of `Resampling{:systematic}(N)`.

## Additional arguments for the constructors of resamplings

The user facing constructor for each resampling is of the form:
```
Resampling{S}(N; randomisation, order, intent)
```
where `S` is a `Symbol` corresponding to a particular resampling (see above).

The arguments are:

* `N`: The number of particles used.

* `randomisation`: A `Symbol` specifying the type of randomisation applied to the indices
which are sampled internally in ascending order.
May be `:default` (default), `:none`, `:shuffle` or `:circular`.
`:shuffle` shuffles the indices randomly and `:circular` applies a random circular shift.
`:default` uses the argument `intent` (see below) to choose a sensible default for the resampling being constructed.

* `order`: specifies an order for the weights `w`. May be `:default` (default), `:none`, `:sort` or `:partition`.
The default is `:default`, which uses `intent` to choose a sensible default for the resampling being constructed.
This argument is available only in the constructors excluding multinomial and killing resampling.

* `intent`: May be `:unconditional` or `:conditional`, default is `:unconditional`.
Specifies how `:default` in arguments `randomisation` and `order` should be resolved (if either is set to `:default`).
The default values produced depend on the resampling being constructed.
`:unconditional` uses a sensible default from the perspective of unconditional resampling, and
`:conditional` from the perspective of conditional resampling. Furthermore, setting `intent = :conditional`
ensures that the output object can implement conditional resampling.
If values for `randomisation` and `order` are passed such that this can not be guaranteed, an `ArgumentError` is thrown.

## Authors

* Santeri Karppinen (skarppinen@iki.fi)

* Matti Vihola

University of Jyväskylä, Finland, Department of Mathematics and Statistics

## License

MIT
