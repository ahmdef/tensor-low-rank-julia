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

## Example result

| Method | Relative reconstruction error | Stored scalar entries |
| --- | ---: | ---: |
| SVD | 1.98% | 616 |
| Randomized SVD | 3.38% | 616 |

The input has 30,720 entries. At these target ranks, the core and three factor matrices use about 2% of that count. The errors shown are from one seeded example.

## Coursework and credit

This demo is based on a first-year scientific computing project at ENSEEIHT. The school provided the assignment, the Pluto notebook scaffold, and supporting materials. I contributed to parts of the implementation as a member of a two-person team.

The [Julia script](demo.jl) is a fresh, standalone demo using generated data. It gives readers a way to explore the tensor approximation ideas from the coursework.
