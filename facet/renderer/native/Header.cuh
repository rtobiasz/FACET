#pragma once

#include "constants.h"

#include "cuda_fp16.h"
#include "cuda_bf16.h"

#include "optix_host.h"
#include "optix_stack_size.h"
#include "optix_stubs.h"

// *************************************************************************************************

struct SbtRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

// *************************************************************************************************

struct SLaunchParams {
	float3 *O;
	float3 *v;
	float *t_min;
	int *cursor_ids;
	float *t;
	int *indices;
	int *candidate_counts;
	OptixTraversableHandle AS;
	int *ray_indices;
};