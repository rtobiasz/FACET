#pragma once

// Fixed-point scale for deterministic gradient accumulation in the renderer
// backward pass. Shared by the CUDA kernels and by the host wrapper that
// converts the accumulators back to float, so the two can never drift apart.
//
// Why fixed point: the per-primitive gradients are written with `atomicAdd` at
// addresses indexed by the primitive, so many rays from many warps hit the same
// address at once and the summation order is whatever the hardware produces.
// Float addition is not associative, so the total differed between otherwise
// identical runs. Integer addition is associative, so an integer atomicAdd
// gives the same total in any order.
// float32 and the only rounding is the conversion to an integer.
// resolves 2^-55 = 2.8e-17. Against the gradients measured on kodim18 (500k
// primitives, 0D conditioning) that is 41 bits of precision for dL_dRGB at
// 9.2e-08 and 26 bits for the smallest, dL_dk at 2.0e-09 -- at or above what
// float32's 24-bit mantissa carried before. Against the much larger gradients
// in tests/test_native_renderer.py, where dL_dRGB reaches about 0.5, it leaves
// a factor of 500 of headroom.
//
// A single scale for every buffer is deliberate: per-buffer scales tuned to one
// image looked tighter but silently overflowed as soon as a different
// configuration produced larger gradients. The guard in the host wrapper fails
// loudly if this one is ever outgrown.
#define FACET_FIXED_POINT_SCALE 36028797018963968.0f          // 2^55
#define FACET_FIXED_POINT_INVERSE_SCALE 2.7755575615628914e-17  // 2^-55

// An accumulator past this is close enough to the int64 limit that the totals
// are about to wrap; the host wrapper raises instead of returning wrong values.
#define FACET_FIXED_POINT_GUARD_LIMIT 4611686018427387904LL   // 2^62
