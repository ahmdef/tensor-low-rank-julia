# Low-rank tensor approximation in Julia

How much of a three-dimensional array can we keep with a much smaller representation? This demo uses a Tucker decomposition to approximate a noisy synthetic tensor, then compares two ways of choosing its factor matrices.

## Run the demo

From the repository root, run:

```sh
julia demo.jl
```

Tested with Julia 1.12.6. The demo uses only Julia standard libraries.

## What the demo does

1. Generate a tensor of size `40 × 32 × 24` with an underlying Tucker rank of `(6, 5, 4)`, then add noise.
2. Compress one axis at a time. The SVD method takes the leading singular vectors of each unfolding. The randomized method estimates a subspace with a Gaussian sketch before applying SVD in that smaller space.
3. Reconstruct the tensor from the core and factor matrices. Measure the relative Frobenius error against the noisy input.

## Why Tucker compression works

When a few patterns explain most of the variation along each axis, a tensor can be approximated using a small basis for each axis. Tucker decomposition stores those bases as factor matrices. A smaller core stores the weights that combine the basis vectors. Multiplying the core along each axis by its factor matrix reconstructs an approximation at the original size.

In this demo, the `40 × 32 × 24` tensor has 30,720 entries. Using ranks `(6, 5, 4)` gives a `6 × 5 × 4` core and three factor matrices of sizes `40 × 6`, `32 × 5`, and `24 × 4`. The core and factors together store:

```text
(6 × 5 × 4) + (40 × 6) + (32 × 5) + (24 × 4) = 616 entries
```

That's about **2% of the entries in the original tensor**.

## Example result

| Method | Relative reconstruction error | Stored scalar entries |
| --- | ---: | ---: |
| SVD | 1.98% | 616 |
| Randomized SVD | 3.38% | 616 |

At the same storage budget, the SVD approximation was closer to the noisy input in this seeded run.

## Coursework and credit

This demo is based on a first-year scientific computing project at ENSEEIHT. The school provided the assignment, the Pluto notebook scaffold, and supporting materials. I contributed to parts of the implementation as a member of a two-person team.

The [Julia script](demo.jl) is a fresh, standalone demo using generated data. It gives readers a way to explore the tensor approximation ideas from the coursework.
