using LinearAlgebra
using Printf
using Random

# Move the selected axis to the front, then group the other axes into columns.
function unfold(tensor, mode)
    axes_order = (mode, (axis for axis in 1:ndims(tensor) if axis != mode)...)
    matrix = reshape(permutedims(tensor, axes_order), size(tensor, mode), :)
    return matrix, axes_order
end

function multiply_along(tensor, transform, mode)
    matrix, axes_order = unfold(tensor, mode)
    transformed = transform * matrix
    other_sizes = map(axis -> size(tensor, axis), Base.tail(axes_order))
    folded = reshape(transformed, (size(transform, 1), other_sizes...))
    return permutedims(folded, invperm(collect(axes_order)))
end

function leading_basis(matrix, rank, rng; randomized)
    if !randomized
        return svd(matrix).U[:, 1:rank]
    end

    # Sample a subspace slightly larger than the desired rank.
    width = min(rank + 4, minimum(size(matrix)))
    sketch = matrix * randn(rng, size(matrix, 2), width)
    basis = Matrix(qr(sketch).Q)[:, 1:width]
    reduced = basis' * matrix
    return basis * svd(reduced).U[:, 1:rank]
end

function compress(tensor, ranks, rng; randomized=false)
    length(ranks) == ndims(tensor) ||
        throw(ArgumentError("Provide one rank per tensor dimension."))

    factors = Matrix{Float64}[]
    core = tensor
    for (mode, rank) in enumerate(ranks)
        matrix, _ = unfold(core, mode)
        1 <= rank <= minimum(size(matrix)) ||
            throw(ArgumentError("Rank $rank is out of range for mode $mode."))
        basis = leading_basis(matrix, rank, rng; randomized)
        push!(factors, basis)
        core = multiply_along(core, basis', mode)
    end
    return core, factors
end

function reconstruct(core, factors)
    tensor = core
    for (mode, basis) in enumerate(factors)
        tensor = multiply_along(tensor, basis, mode)
    end
    return tensor
end

function demo()
    rng = MersenneTwister(42)
    dimensions = (40, 32, 24)
    ranks = (6, 5, 4)

    core = randn(rng, ranks...)
    factors = [
        Matrix(qr(randn(rng, dimension, rank)).Q)[:, 1:rank]
        for (dimension, rank) in zip(dimensions, ranks)
    ]
    clean = reconstruct(core, factors)
    noise = randn(rng, dimensions...)
    observed = clean + (0.02 * norm(clean) / norm(noise)) * noise

    println("Input shape: ", size(observed), "  Target ranks: ", ranks)
    for (name, randomized) in (("SVD", false), ("Randomized SVD", true))
        estimate, bases = compress(observed, ranks, rng; randomized)
        approximation = reconstruct(estimate, bases)
        error = norm(observed - approximation) / norm(observed)
        stored = length(estimate) + sum(length, bases)
        @printf("%-15s relative error: %.4f  entries: %d / %d\n",
            name, error, stored, length(observed))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    demo()
end
