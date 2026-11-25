#pragma once

#include "DeterministicAccumulationScales.h"

// Add one contribution to a fixed-point gradient accumulator.
// zeroes and converts back to the caller's float tensor after the launch, so
// the renderer's public interface stays float32 in and float32 out. `index`
// counts 64-bit slots and matches the element index the float buffer used.
__device__ __forceinline__ void facet_atomic_add_fixed(
	float *base,
	long long index,
	float value,
	float scale
) {
	long long quantum = __float2ll_rn(value * scale);
	// Contributions that quantise to zero are dropped; they are below the
	// resolution set by the scale, and adding zero is a no-op anyway.
	if (quantum != 0LL)
		atomicAdd(((unsigned long long *)base) + index, (unsigned long long)quantum);
}
