include("helpers.jl");
using Random

## Check that all resample! operations work with
# all order and randomisation settings.
@testset "Check that call to `resample!` works for all resamplings with any order and randomisation + weights are not modified in `resample!`" begin
    N = 16;
    ind = zeros(Int, N);
    w_orig = rand(N); w_orig .= w_orig ./ sum(w_orig);
    w = copy(w_orig);
    @assert sum(w) ≈ 1.0

    for order in [:none, :sort, :partition],
        randomisation in [:none, :shuffle, :circular]

        for constr in [SSSResampling,
                       ResidualResampling, SSPResampling,
                       StratifiedResampling, SystematicResampling]
            res = constr(N, order = order, randomisation = randomisation);
            @test resample!(res, ind, w) == nothing
            @test w == w_orig;

        end
    end

    for randomisation in [:none, :shuffle, :circular]

        for constr in [MultinomialResampling, KillingResampling]
            res = constr(N, randomisation = randomisation);
            @test resample!(res, ind, w) == nothing
            @test w == w_orig;

        end
    end
end

@testset "Check that calls to default constructors (such as `MultinomialResampling(10)`) work for all resamplings" begin
    N = 16;
    ind = zeros(Int, N);
    w_orig = rand(N); w_orig .= w_orig ./ sum(w_orig);
    w = copy(w_orig);
    @assert sum(w) ≈ 1.0

    for constr in [MultinomialResampling, KillingResampling, SSSResampling,
                   SSPResampling, ResidualResampling,
                   StratifiedResampling, SystematicResampling]
        res = constr(N);
        @test resample!(res, ind, w) == nothing;
        @test w == w_orig;
    end
end

@testset "Check that `resample!` detects invalid arguments as promised" begin
    N = 16;
    ind = zeros(Int, N);
    w = rand(N + 1); w .= w ./ sum(w);
    res = MultinomialResampling(N);
    @test_throws AssertionError resample!(res, ind, w)

    N = 16;
    ind = zeros(Int, N + 1);
    w = rand(N); w .= w ./ sum(w);
    res = MultinomialResampling(N);
    @test_throws AssertionError resample!(res, ind, w)

    N = 16;
    ind = zeros(Int, N);
    w = rand(N); w .= w ./ sum(w);
    res = MultinomialResampling(N + 1);
    @test_throws AssertionError resample!(res, ind, w)
end

@testset "Check that `resample!` does not allocate for any order or randomisation (NOTE: currently excluding order = :sort)" begin
    N = 16;
    ind = zeros(Int, N);
    w_orig = rand(N); w_orig .= w_orig ./ sum(w_orig);
    w = copy(w_orig);
    @assert sum(w) ≈ 1.0

    for order in [:none, :partition],
        randomisation in [:none, :shuffle, :circular]

        for constr in [SSSResampling,
                       ResidualResampling, SSPResampling,
                       StratifiedResampling, SystematicResampling]
            res = constr(N, order = order, randomisation = randomisation);
            allocs = @allocated resample!(res, ind, w);
            @test allocs == 0
        end
    end

    for randomisation in [:none, :shuffle, :circular]
        for constr in [MultinomialResampling, KillingResampling]
            res = constr(N, randomisation = randomisation);
            allocs = @allocated resample!(res, ind, w);
            @test allocs == 0
        end
    end
end

println("Running symmetry tests... this will take a while")

## Test symmetry for unconditional resamplings, that is P(A^{i} = j) = w^{j},
# where w contains the normalised weights.
# Only cases with randomisation != :none checked, otherwise symmetry does not
# necessarily hold.
for constr in ["multinomial", "killing"]
    for randomisation in [:shuffle, :circular]
        Random.seed!(28012022);
        cutoff = 4.0;
        N = 16;
        m = 50_000_000;
        ind = zeros(Int, N);
        w = collect(1.0:1.0:N); w .= w ./ sum(w);
        @assert sum(w) ≈ 1.0
        set_name = string("Symmetricity of $constr resampling, ",
                           "randomisation = :$randomisation.");

        @testset "$set_name" begin
            res = Resampling{Symbol(constr)}(N; randomisation = randomisation);
            est_prob_mat = empirical_resampling_probs(w, m, res, ref_cur = 0);
            sigma = sqrt.(w .* (1.0 .- w) ./ m);
            should_be_N01 = (est_prob_mat .- w) ./ sigma;
            @test sum(abs.(should_be_N01) .> cutoff) <= 1
        end
    end
end

for constr in ["sss", "residual", "ssp", "stratified", "systematic"]

    for randomisation in [:shuffle, :circular],
        order in [:none, :sort, :partition]

        Random.seed!(28012022);
        cutoff = 4.0;
        N = 16;
        m = 50_000_000;
        ind = zeros(Int, N);
        w = collect(1.0:1.0:N); w .= w ./ sum(w);
        @assert sum(w) ≈ 1.0
        set_name = string("Symmetricity of $constr resampling ",
                           "(randomisation = :$randomisation", ", ",
                           "order = :$order)");

        @testset "$set_name" begin
            res = Resampling{Symbol(constr)}(N; randomisation = randomisation, order = order);
            est_prob_mat = empirical_resampling_probs(w, m, res, ref_cur = 0);
            sigma = sqrt.(w .* (1.0 .- w) ./ m);
            should_be_N01 = (est_prob_mat .- w) ./ sigma;
            @test sum(abs.(should_be_N01) .> cutoff) <= 1
        end
    end
end
