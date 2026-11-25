#include "Header.cuh"
#include "optix_device.h"

// *** *** *** *** ***

extern "C" __constant__ SLaunchParams optixLaunchParams;

// *** *** *** *** ***

// Optional hit order for lifted 2D models (build flag ID_ORDER_DELTA, off by
// default). A FACET 2D model is composited in (t, id) order while all its
// primitives share one plane. After lifting to 3D, neighbours sit at slightly
// different depths and plain t order scrambles that composite order. With
// FACET_ID_ORDER_DELTA the sort key is t * (1 + delta * id / 2^19): within a
// relative depth band of about delta the training id order wins, beyond it
// depth order (occlusion) wins. Only the order changes; every hit keeps its
// true t. The default build sorts hits by (t, id).
// FACET_ID_ORDER_SPAN (default 2^19) normalises the id; it must exceed every
// primitive id: the resumed trace below starts at key / (1 + delta), which
// misses later hits with id > SPAN and a slightly smaller t (a model with
// 1.5M primitives lost layers once lifted surfels left the shared plane).
#ifdef FACET_ID_ORDER_DELTA
#ifndef FACET_ID_ORDER_SPAN
#define FACET_ID_ORDER_SPAN 524288.0f
#endif
// FACET_ID_ORDER_SOURCE (opt-in, with FACET_ID_ORDER_DELTA): the band is measured from the source camera,
// which sits at the world origin of a lifted model, instead of from the ray origin. The key is
// t |v| + delta |O + t v| id / 2^19, the length along the ray plus the offset scaled by the hit point's distance
// from the source camera. For rays from the source camera (O = 0) it orders hits exactly as the default key.
// For other cameras and for mirror or glass rays, the tolerance is a property of the surface point, not of
// the ray length, so the same two layers keep the same order whatever ray hits them.
static __device__ __forceinline__ float facet_order_key(float t, int id, float3 O, float3 v) {
#ifdef FACET_ID_ORDER_SOURCE
	float px = O.x + t * v.x, py = O.y + t * v.y, pz = O.z + t * v.z;
	return t * sqrtf(v.x * v.x + v.y * v.y + v.z * v.z)
	     + FACET_ID_ORDER_DELTA * sqrtf(px * px + py * py + pz * pz) * ((float)id * (1.0f / FACET_ID_ORDER_SPAN));
#else
	return t * (1.0f + FACET_ID_ORDER_DELTA * ((float)id * (1.0f / FACET_ID_ORDER_SPAN)));
#endif
}
// Smallest t a hit can have when its key is at least key (used to resume the trace after a full buffer).
static __device__ __forceinline__ float facet_order_t_lower(float key, float3 O, float3 v) {
#ifdef FACET_ID_ORDER_SOURCE
	// key <= t |v| + delta (|O| + t |v|), since id / 2^19 < 1.
	float lv = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z), lo = sqrtf(O.x * O.x + O.y * O.y + O.z * O.z);
	return fmaxf(0.0f, (key - FACET_ID_ORDER_DELTA * lo) / (lv * (1.0f + FACET_ID_ORDER_DELTA)));
#else
	return key / (1.0f + FACET_ID_ORDER_DELTA);
#endif
}
static __device__ __forceinline__ bool facet_hit_before(float ta, int ia, float tb, int ib, float3 O, float3 v) {
	float ka = facet_order_key(ta, ia, O, v), kb = facet_order_key(tb, ib, O, v);
	return (ka < kb) || ((ka == kb) && (ia < ib));
}
#endif

// *** *** *** *** ***

// The bounded, ordered set of ray-proxy intersections that the renderer
// composites front to back.
struct SRayPayload {
	int number_of_hits;
	int number_of_candidates;
	float cursor_t;
	int cursor_id;
	float2 data[HIT_BUFFER_SIZE];
};

// *** *** *** *** ***

extern "C" __global__ void __raygen__test() {
	uint3 launch_index = optixGetLaunchIndex();
	uint3 launch_dimensions = optixGetLaunchDimensions();

	int pixel_ind = launch_index.x;
	int number_of_rays = launch_dimensions.x;

	float3 O = optixLaunchParams.O[pixel_ind];
	float3 v = optixLaunchParams.v[pixel_ind];
	float t_min = optixLaunchParams.t_min[pixel_ind];

	float *t = optixLaunchParams.t;
	int *indices = optixLaunchParams.indices;

	// *********************************************************************************************

	SRayPayload rp;

	unsigned long long rp_addr = ((unsigned long long)&rp);
	unsigned rp_addr_lo = rp_addr;
	unsigned rp_addr_hi = rp_addr >> 32;

	// *********************************************************************************************

	rp.number_of_hits = 0;
	rp.number_of_candidates = 0;
	rp.cursor_t = t_min;
	rp.cursor_id = -1;

	optixTrace(
		optixLaunchParams.AS,
		O,
		v,
		t_min,
		INFINITY,
		0.0f,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
		0,
		1,
		0,

		rp_addr_lo,
		rp_addr_hi
	);

	// *********************************************************************************************

	for (int i = 0; i < HIT_BUFFER_SIZE; ++i) {
		int element_index = (i * number_of_rays) + pixel_ind;

		if (i < rp.number_of_hits) {
			float2 tmp = rp.data[i];

			t[element_index] = tmp.x;
			indices[element_index] = __float_as_int(tmp.y);
		} else {
			t[element_index] = INFINITY;
			indices[element_index] = -1;
		}
	}
	if (optixLaunchParams.candidate_counts != nullptr)
		optixLaunchParams.candidate_counts[pixel_ind] = rp.number_of_candidates;
}

// *** *** *** *** ***

extern "C" __global__ void __raygen__() {
	uint3 launch_index = optixGetLaunchIndex();
	uint3 launch_dimensions = optixGetLaunchDimensions();

	int pixel_ind = launch_index.x;
	int ray_ind = optixLaunchParams.ray_indices[pixel_ind];
	int number_of_rays = launch_dimensions.x;

	float3 O = optixLaunchParams.O[ray_ind];
	float3 v = optixLaunchParams.v[ray_ind];
	float t_min = optixLaunchParams.t_min[ray_ind];

	float *t = optixLaunchParams.t;
	int *indices = optixLaunchParams.indices;

	// *********************************************************************************************

	SRayPayload rp;

	unsigned long long rp_addr = ((unsigned long long)&rp);
	unsigned rp_addr_lo = rp_addr;
	unsigned rp_addr_hi = rp_addr >> 32;

	// *********************************************************************************************

	rp.number_of_hits = 0;
	rp.number_of_candidates = 0;
	rp.cursor_id = optixLaunchParams.cursor_ids ? optixLaunchParams.cursor_ids[ray_ind] : -1;
	// OptiX did not report exact-boundary ties when restarted at cursor_t in the
	// coplanar regression. The host-side kernels therefore pass one ULP below
	// cursor_t; reconstruct the logical lexicographic cursor here.
	rp.cursor_t = (rp.cursor_id >= 0) ? nextafterf(t_min, INFINITY) : t_min;

	float t_trace = t_min;
#ifdef FACET_ID_ORDER_DELTA
	// Any hit whose key follows the cursor's key has t >= key / (1 + delta).
	if (rp.cursor_id >= 0)
		t_trace = fminf(t_min, facet_order_t_lower(facet_order_key(rp.cursor_t, rp.cursor_id, O, v), O, v)
		                           * (1.0f - 1e-6f));
#endif

	optixTrace(
		optixLaunchParams.AS,
		O,
		v,
		t_trace,
		INFINITY,
		0.0f,
		OptixVisibilityMask(255),
		OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
		0,
		1,
		0,

		rp_addr_lo,
		rp_addr_hi
	);

	// *********************************************************************************************

	for (int i = 0; i < HIT_BUFFER_SIZE; ++i) {
		int element_index = (i * number_of_rays) + pixel_ind;

		if (i < rp.number_of_hits) {
			float2 tmp = rp.data[i];

			t[element_index] = tmp.x;
			indices[element_index] = __float_as_int(tmp.y);
		} else {
			t[element_index] = INFINITY;
			indices[element_index] = -1;
		}
	}
}

// *** *** *** *** ***

// Collect every relevant primitive hit and keep the payload sorted by its key.
extern "C" __global__ void __anyhit__() {
	SRayPayload *rp;

	unsigned long long rp_addr_lo = optixGetPayload_0();
	unsigned long long rp_addr_hi = optixGetPayload_1();
	*((unsigned long long *)&rp) = rp_addr_lo + (rp_addr_hi << 32);

	// *********************************************************************************************

	float t_hit = optixGetRayTmax();
	
	float3 O = optixGetObjectRayOrigin();
	float3 v = optixGetObjectRayDirection();
	
	unsigned Gauss_ind = optixGetInstanceIndex();
#ifdef FACET_ID_ORDER_DELTA
	float3 Ow = optixGetWorldRayOrigin(), vw = optixGetWorldRayDirection();
	if ((rp->cursor_id < 0) ? (t_hit < rp->cursor_t)
	                        : !facet_hit_before(rp->cursor_t, rp->cursor_id, t_hit, (int)Gauss_ind, Ow, vw)) {
		optixIgnoreIntersection();
		return;
	}
#else
	if ((t_hit < rp->cursor_t) ||
		((t_hit == rp->cursor_t) && ((int)Gauss_ind <= rp->cursor_id))) {
		optixIgnoreIntersection();
		return;
	}
#endif

	rp->number_of_candidates += 1;

	int number_of_hits = rp->number_of_hits;
	float2 *data = rp->data;

	float *t = optixLaunchParams.t;
	int *indices = optixLaunchParams.indices;

	// *** *** *** *** ***

	float2 tmp1 = make_float2(t_hit, __int_as_float(Gauss_ind));
	float2 tmp2;

	for (int i = 0; i < number_of_hits; ++i) {
		tmp2 = data[i];

		int tmp1_index = __float_as_int(tmp1.y);
		int tmp2_index = __float_as_int(tmp2.y);
#ifdef FACET_ID_ORDER_DELTA
		if (facet_hit_before(tmp1.x, tmp1_index, tmp2.x, tmp2_index, Ow, vw)) {
#else
		if ((tmp1.x < tmp2.x) ||
			((tmp1.x == tmp2.x) && (tmp1_index < tmp2_index))) {
#endif
			data[i] = tmp1;
			tmp1 = tmp2;
		}
	}

	if (number_of_hits < HIT_BUFFER_SIZE) {
		data[number_of_hits] = tmp1;
		rp->number_of_hits = number_of_hits + 1;
	}

	optixIgnoreIntersection();
}
