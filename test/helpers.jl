"""
Function computes empirical probabilities of resampling the integerss 1:N with
probabilities given in `w` based on `m` draws.
`ref_cur` is the assumed conditioning index. Setting `ref_cur` < 0 does computations for
the corresponding unconditional resampling.

The output is a matrix, whose element (i, j) gives the empirical probability
of drawing integer i at index j.
If the resampling is symmetric, that is, P(A^{i} = j) = `w[j]` holds,
then each column should have approximately same empirical probabilities.
Here, A^{(i)} stands for the integer at the ith index.
"""
function empirical_resampling_probs(w::AbstractVector{<: Real}, m::Integer,
                                    resampling; ref_cur::Int = 1)
    @assert sum(w) ≈ 1.0 "`w` should be normalised.";
    N = length(w);
    ind = zeros(Int, N);
    conditional = ref_cur > 0;

    # For each index in `ind` construct a vector for summing
    # how many times each value in 1:N was found at that index after
    # resampling. results[3][8] accumulates how many times in index
    # 3 we found the number 8.
    counts = map(1:N) do n
        zeros(Int, N);
    end

    for i in Base.OneTo(m)
        # Do resampling.
        if conditional
            ref_prev = Resamplings._wsample_one(w);
            conditional_resample!(resampling, ind, w, ref_cur, ref_prev);
            @assert ind[ref_cur] == ref_prev "something wrong, ind[ref_cur] != ref_prev"
        else
            resample!(resampling, ind, w);
        end

        # Update counts.
        for j in eachindex(ind)
            @inbounds counts[j][ind[j]] += 1;
        end
    end

    # Return matrix of empirical probabilities.
    map(counts) do count_vec
        count_vec ./ m
    end |> x -> hcat(x...)
end

function empirical_unbiasedness_test(w::AbstractVector{<: Real}, m::Integer,
                                     resampling; ref_cur::Int = 1)
    @assert sum(w) ≈ 1.0 "`w` should be normalised.";
    N = length(w);
    ind = zeros(Int, N);
    conditional = ref_cur > 0;
    counts = zeros(Int, N); # Temporary: index i tells how many i found in a single round.
    mean_counts = zeros(N); # Output: index i tells mean of how many i was sampled over iterations.

    for i in Base.OneTo(m)
        # Do resampling.
        if conditional
            ref_prev = Resamplings._wsample_one(w);
            ind[ref_cur] = ref_prev;
            conditional_resample!(resampling, ind, w, ref_cur, ref_prev);
            @assert ind[ref_cur] == ref_prev "something wrong, ind[ref_cur] != ref_prev"
        else
            resample!(resampling, ind, w);
        end

        # Compute counts of each index this iteration.
        counts .= 0;
        for j in eachindex(ind)
            @inbounds counts[ind[j]] += 1;
        end

        # Update mean counts ().
        invi = inv(i);
        mean_counts .= invi .* counts + (1.0 - invi) .* mean_counts;
        #for j in eachindex(mean_counts)
        #    @inbounds mean_counts[j] = invm * counts[j]  + (1.0 - invm) * mean_counts[j];
        #end

    end
    mean_counts;
end
