@inline function _set_resample_order!(o::Order{:none},
                                      w::AbstractVector{<: AbstractFloat})
    nothing;
end
@inline function _set_resample_order!(o::Order{:sort},
                                      w::AbstractVector{<: AbstractFloat})
     sortperm!(o.order, w);
     nothing;
end
@inline function _set_resample_order!(o::Order{:partition},
                                      w::AbstractVector{<: AbstractFloat})
    _partition_order!(o.order, w);
    nothing;
end

@inline function _ascending_inv_cdf_lookup!(ind::AbstractVector{<: Integer},
                                            p::AbstractVector{<: AbstractFloat},
                                            u::AbstractVector{<: AbstractFloat},
                                            o::Order)
    _ascending_inv_cdf_lookup!(ind, p, u, o.order);
end


# This finds order of partition so that first are those
# less than the pivot (default: average) and then those greather than the pivot
# (Like the partition scheme you do in quicksort)
function _partition_order!(ind::AbstractVector{<: Integer},
                           w::AbstractVector{<: AbstractFloat},
                           pivot::AbstractFloat = inv(length(w)))
    # Sanity checks
    n = length(w)
    n <= 1 && return nothing
    @assert n == length(ind)

    # Set order
    ind .= 1:n

    # Indices "where we are at" in rearrangement.
    # Start from the very left and very right end
    i_lower = 0; i_upper = n + 1

    while true
        # Find "next" index i_lower with > pivot:
        while i_lower < min(i_upper, n)
            i_lower += 1
            @inbounds w[ind[i_lower]] > pivot && break
        end
        # Find "previous" index i_upper with < pivot:
        while i_upper > i_lower
            i_upper -= 1
            @inbounds w[ind[i_upper]] < pivot && break
        end
        # If the indices met, we are done
        i_upper == i_lower && break
        # Otherwise, swap the elements
        @inbounds swapindices!(ind, i_lower, i_upper)
    end
    nothing
end

Base.@propagate_inbounds function swapindices!(x::AbstractVector, i::Integer, j::Integer)
    x[i], x[j] = x[j], x[i];
    nothing;
end
