include("helpers.jl");
using Random

@testset "Function `has_conditional` works" begin
    @test has_conditional(:multinomial, :shuffle, :none) == true;
    @test has_conditional(:multinomial, :circular, :none) == false;
    @test has_conditional(:multinomial, :none, :none) == false;
    @test has_conditional(:killing, :none, :none) == false;
    @test has_conditional(:killing, :circular, :none) == true;
    @test has_conditional(:killing, :shuffle, :none) == false;

    for randomisation in Resamplings._RANDOMISATION_CHOICES, order in Resamplings._ORDER_CHOICES
        @test has_conditional(:ssp, randomisation, order) == false;
    end
    for randomisation in Resamplings._RANDOMISATION_CHOICES, order in Resamplings._ORDER_CHOICES
        if randomisation == :circular
            @test has_conditional(:systematic, randomisation, order) == true;
        else
            @test has_conditional(:systematic, randomisation, order) == false;
        end
    end

end

@testset "Calls to `conditional_resample!` work for implemented conditional resamplings" begin
    for resampling in setdiff(Resamplings._RESAMPLING_CHOICES, (:multinomial, :killing)),
        randomisation in Resamplings._RANDOMISATION_CHOICES,
        order in Resamplings._ORDER_CHOICES
        !has_conditional(resampling, randomisation, order) && (continue;)

        N = 16;
        ind = zeros(Int, N);
        w = rand(N); w .= w ./ sum(w);
        k = 3;
        i = 4;
        #ind[k] = i;
        res = Resampling{resampling}(N; randomisation = randomisation, order = order);
        @test conditional_resample!(res, ind, w, k, i) == nothing;
    end

    for resampling in (:multinomial, :killing),
        randomisation in Resamplings._RANDOMISATION_CHOICES

        !has_conditional(resampling, randomisation, :none) && (continue;)

        N = 16;
        ind = zeros(Int, N);
        w = rand(N); w .= w ./ sum(w);
        k = 3;
        i = 4;
        #ind[k] = i;
        res = Resampling{resampling}(N; randomisation = randomisation);
        @test conditional_resample!(res, ind, w, k, i) == nothing;
    end
end

@testset "Check that `conditional_resample!` detects invalid arguments as promised" begin
    N = 16;
    ind = zeros(Int, N);
    w = rand(N + 1); w .= w ./ sum(w);
    res = MultinomialResampling(N);
    @test_throws AssertionError conditional_resample!(res, ind, w, 1, 1)

    N = 16;
    ind = zeros(Int, N + 1);
    w = rand(N); w .= w ./ sum(w);
    res = MultinomialResampling(N);
    @test_throws AssertionError conditional_resample!(res, ind, w, 1, 1)

    N = 16;
    ind = zeros(Int, N);
    w = rand(N); w .= w ./ sum(w);
    res = MultinomialResampling(N + 1);
    @test_throws AssertionError conditional_resample!(res, ind, w, 1, 1)

    N = 16;
    ind = zeros(Int, N);
    w = rand(N); w .= w ./ sum(w);
    res = MultinomialResampling(N);
    @test_throws AssertionError conditional_resample!(res, ind, w, 0, 1)
    @test_throws AssertionError conditional_resample!(res, ind, w, N + 1, 1)
    @test_throws AssertionError conditional_resample!(res, ind, w, 1, 0)
    @test_throws AssertionError conditional_resample!(res, ind, w, 1, N + 1)
    @test_throws AssertionError conditional_resample!(res, ind, w, N + 1, N + 1)
    @test_throws AssertionError conditional_resample!(res, ind, w, -1, -1)

    #@test_throws AssertionError conditional_resample!(res, ind, w, 3, 3);
end


@testset "Check that `conditional_resample!` does not allocate for implemented conditional resamplings (NOTE: excluding order = :sort)" begin
    N = 16;
    ind = zeros(Int, N);
    w = rand(N); w .= w ./ sum(w);
    @assert sum(w) ≈ 1.0
    k = 3;
    i = 4;
    #ind[k] = i;

    for resampling in setdiff(Resamplings._RESAMPLING_CHOICES, (:multinomial, :killing)),
        randomisation in Resamplings._RANDOMISATION_CHOICES,
        order in Resamplings._ORDER_CHOICES
            !has_conditional(resampling, randomisation, order) && (continue;)
            order == :sort && (continue;)

            res = Resampling{resampling}(N, order = order, randomisation = randomisation);
            allocs = @allocated conditional_resample!(res, ind, w, k, i);
            @test allocs == 0
    end

    for resampling in (:multinomial, :killing),
        randomisation in Resamplings._RANDOMISATION_CHOICES

            !has_conditional(resampling, randomisation, :none) && (continue;)

            res = Resampling{resampling}(N, randomisation = randomisation);
            allocs = @allocated conditional_resample!(res, ind, w, k, i);
            @test allocs == 0
    end
end

let
    N = 16;
    w_orig = rand(N);
    w_orig .= w_orig ./ sum(w_orig);
    w = copy(w_orig);
    @assert sum(w) ≈ 1.0;
    ind = zeros(Int, N);

    for resampling in setdiff(Resamplings._RESAMPLING_CHOICES, (:multinomial, :killing)),
        randomisation in Resamplings._RANDOMISATION_CHOICES,
        order in Resamplings._ORDER_CHOICES

        !has_conditional(resampling, randomisation, :none) && (continue;)
        res = Resampling{resampling}(N; order = order, randomisation = randomisation);
        call = Resamplings._prettify_name(resampling) * "Resampling(N; " * "randomisation = :$randomisation, order = :$order)";
        set_name = "100 random tests that `conditional_resample!` with $call preserves conditioning and does not modify weights";
        @testset "$set_name" begin
            for i in 1:100
                ref_prev = Resamplings._wsample_one(w);
                ref_cur = rand(1:N);
                conditional_resample!(res, ind, w, ref_cur, ref_prev);
                @test w == w_orig;
                @test ind[ref_cur] == ref_prev;
            end
        end
    end

    for resampling in (:multinomial, :killing),
        randomisation in Resamplings._RANDOMISATION_CHOICES

        !has_conditional(resampling, randomisation, :none) && (continue;)
        res = Resampling{resampling}(N; randomisation = randomisation);
        call = Resamplings._prettify_name(resampling) * "Resampling(N; " * "randomisation = :$randomisation)";
        set_name = "100 random tests that `conditional_resample!` with $call preserves conditioning and does not modify weights";
        @testset "$set_name" begin
            for i in 1:100
                ref_prev = Resamplings._wsample_one(w);
                ref_cur = rand(1:N);
                ind[ref_cur] = ref_prev;
                conditional_resample!(res, ind, w, ref_cur, ref_prev);
                @test w == w_orig;
                @test ind[ref_cur] == ref_prev;
            end
        end
    end
end

## Symmetricity tests for conditional resamplings.
for resampling in setdiff(Resamplings._RESAMPLING_CHOICES, (:multinomial, :killing)),
    randomisation in Resamplings._RANDOMISATION_CHOICES,
    order in Resamplings._ORDER_CHOICES

    !has_conditional(resampling, randomisation, order) && (continue;)

    Random.seed!(28012022);
    cutoff = 4.0;
    N = 16;
    m = 50_000_000;
    ind = zeros(Int, N);
    w = collect(1.0:1.0:N); w .= w ./ sum(w);
    @assert sum(w) ≈ 1.0
    set_name = string("Symmetricity of conditional $(string(resampling)) resampling ",
                      "(randomisation = :$randomisation", ", ",
                      "order = :$order)");

    @testset "$set_name" begin
        res = Resampling{resampling}(N; randomisation = randomisation, order = order);
        est_prob_mat = empirical_resampling_probs(w, m, res, ref_cur = 8);
        sigma = sqrt.(w .* (1.0 .- w) ./ m);
        should_be_N01 = (est_prob_mat .- w) ./ sigma;
        @test sum(abs.(should_be_N01) .> cutoff) == 0
    end
end

for resampling in (:multinomial, :killing),
    randomisation in Resamplings._RANDOMISATION_CHOICES

    !has_conditional(resampling, randomisation, :none) && (continue;)

    Random.seed!(28012022);
    cutoff = 4.0;
    N = 16;
    m = 50_000_000;
    ind = zeros(Int, N);
    w = collect(1.0:1.0:N); w .= w ./ sum(w);
    @assert sum(w) ≈ 1.0
    set_name = string("Symmetricity of conditional $(string(resampling)) resampling",
                      ", randomisation = :$randomisation");

    @testset "$set_name" begin
        res = Resampling{resampling}(N; randomisation = randomisation);
        est_prob_mat = empirical_resampling_probs(w, m, res, ref_cur = 8);
        sigma = sqrt.(w .* (1.0 .- w) ./ m);
        should_be_N01 = (est_prob_mat .- w) ./ sigma;
        @test sum(abs.(should_be_N01) .> cutoff) == 0
    end
end
