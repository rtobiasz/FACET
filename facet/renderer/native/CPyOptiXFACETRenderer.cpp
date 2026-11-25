#ifdef _WIN32
	#define NOMINMAX
#endif

#include "constants.h"
#include "CPyOptiXFACETRenderer.h"
#include "DeterministicAccumulationScales.h"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

// *** *** *** *** ***

// Descriptor row stride in half2 for the CUDA kernels: 0 for a broadcast zero
// descriptor (stride 0), otherwise 48 for the fixed 96D ABI. A contiguous
// N x 4, 8 or 16 descriptor is accepted by the training forward and backward
// of COMPACT_CONDITIONING builds (stride 2, 4 or 8; --fast_descriptor) and
// rejected everywhere else, where it would be read as 96 columns.
static int conditioning_stride_of(const torch::Tensor &conditioning_variable, bool allow_compact) {
	if (conditioning_variable.strides()[0] == 0) return 0;
	if (conditioning_variable.dim() == 2) {
		const int64_t width = conditioning_variable.size(1);
		if ((width == 4) || (width == 8) || (width == 16)) {
#ifdef FACET_COMPACT_CONDITIONING
			if (allow_compact && conditioning_variable.is_contiguous()) return (int)(width / 2);
#endif
			throw std::runtime_error(
				"compact N x " + std::to_string(width) + " descriptor needs a renderer built with "
				"COMPACT_CONDITIONING=1 and is accepted by forward_training/backward only");
		}
	}
	return 48;
}

// *** *** *** *** ***

CPyOptiXFACETRenderer::CPyOptiXFACETRenderer(
	int number_of_sides, float chi_square_squared_radius, int max_batch_size
) :
	renderer(number_of_sides, chi_square_squared_radius, max_batch_size)
{
}

// *** *** *** *** ***

void CPyOptiXFACETRenderer::SetGeometry(torch::Tensor &m, torch::Tensor &s, torch::Tensor &q, torch::Tensor &opacity, torch::Tensor &kappa) {
	float *m_ptr = (float *)m.data_ptr();
	float *s_ptr = (float *)s.data_ptr();
	float *q_ptr = (float *)q.data_ptr();
	float *opacity_ptr = (float *)opacity.data_ptr();
	float *kappa_ptr = (float *)kappa.data_ptr();
	c10::IntArrayRef sizes = m.sizes();
	renderer.SetGeometry(
		m_ptr, s_ptr, q_ptr, opacity_ptr, kappa_ptr,
		sizes[0]
	);
}

// *** *** *** *** ***

std::tuple<torch::Tensor, torch::Tensor> CPyOptiXFACETRenderer::Render(
	torch::Tensor &O, torch::Tensor &v, torch::Tensor &t_min,
	int size
) {
	float *O_ptr = (float *)O.data_ptr();
	float *v_ptr = (float *)v.data_ptr();
	float *t_min_ptr = (float *)t_min.data_ptr();

	const int64_t size1[] = {HIT_BUFFER_SIZE, size};
	torch::TensorOptions options;

	options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	torch::Tensor t = torch::empty(size1, options);

	options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	torch::Tensor indices = torch::empty(size1, options);

	float *t_ptr = (float *)t.data_ptr();
	int *indices_ptr = (int *)indices.data_ptr();

	renderer.Render(
		O_ptr, v_ptr, t_min_ptr,
		size,
		t_ptr, indices_ptr, nullptr
	);

	return std::make_tuple(t, indices);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> CPyOptiXFACETRenderer::TraceHits(
	torch::Tensor &O, torch::Tensor &v, torch::Tensor &t_min,
	int size
) {
	const int64_t hit_sizes[] = {HIT_BUFFER_SIZE, size};
	const int64_t count_sizes[] = {size};
	torch::TensorOptions float_options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
	torch::TensorOptions int_options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	torch::Tensor t = torch::empty(hit_sizes, float_options);
	torch::Tensor indices = torch::empty(hit_sizes, int_options);
	torch::Tensor candidate_counts = torch::empty(count_sizes, int_options);

	renderer.Render(
		(float *)O.data_ptr(), (float *)v.data_ptr(), (float *)t_min.data_ptr(),
		size,
		(float *)t.data_ptr(), (int *)indices.data_ptr(),
		(int *)candidate_counts.data_ptr()
	);
	return std::make_tuple(t, indices, candidate_counts);
}

// *** *** *** *** ***

void CPyOptiXFACETRenderer::GetMedianDepth_base(
	torch::Tensor &O,
	torch::Tensor &v,

	torch::Tensor &m,
	torch::Tensor &s,
	torch::Tensor &q,
	torch::Tensor &RGBA,
	torch::Tensor &kappa,

	float T_threshold,

	torch::Tensor &depth_and_index
) {
	float3 *O_ptr = (float3 *)O.data_ptr();
	float3 *v_ptr = (float3 *)v.data_ptr();

	float3 *m_ptr = (float3 *)m.data_ptr();
	float2 *s_ptr = (float2 *)s.data_ptr();
	float4 *q_ptr = (float4 *)q.data_ptr();
	float4 *RGBA_ptr = (float4 *)RGBA.data_ptr();
	float *kappa_ptr = (float *)kappa.data_ptr();

	float2 *depth_and_index_ptr = (float2 *)depth_and_index.data_ptr();

	c10::IntArrayRef size1 = O.sizes();

	renderer.GetMedianDepth_base(
		O_ptr,
		v_ptr,

		size1[0], // number of rays

		m_ptr,
		s_ptr,
		q_ptr,
		RGBA_ptr,
		kappa_ptr,

		T_threshold,

		depth_and_index_ptr
	);
}

// *** *** *** *** ***

void CPyOptiXFACETRenderer::Forward_training_base(
	torch::Tensor &O,
	torch::Tensor &v,
	torch::Tensor &bitmap,

	float bg_color_R, float bg_color_G, float bg_color_B,

	torch::Tensor &m,
	torch::Tensor &s,
	torch::Tensor &q,
	torch::Tensor &RGBA,
	torch::Tensor &kappa,

	float T_threshold,

	torch::Tensor &depth_reg_accums,
	float reg_a,
	float reg_b,

	torch::Tensor &depth_and_index,
	torch::Tensor &surface_normal,
	torch::Tensor &normal_reg_accums
) {
	float3 *O_ptr = (float3 *)O.data_ptr();
	float3 *v_ptr = (float3 *)v.data_ptr();
	float3 *bitmap_ptr = (float3 *)bitmap.data_ptr();

	float3 *m_ptr = (float3 *)m.data_ptr();
	float2 *s_ptr = (float2 *)s.data_ptr();
	float4 *q_ptr = (float4 *)q.data_ptr();
	float4 *RGBA_ptr = (float4 *)RGBA.data_ptr();
	float *kappa_ptr = (float *)kappa.data_ptr();

	float4 *depth_reg_accums_ptr = (float4 *)depth_reg_accums.data_ptr();

	float2 *depth_and_index_ptr = (float2 *)depth_and_index.data_ptr();
	float3 *surface_normal_ptr = (float3 *)surface_normal.data_ptr();
	float4 *normal_reg_accums_ptr = (float4 *)normal_reg_accums.data_ptr();

	c10::IntArrayRef size1 = O.sizes();

	renderer.Forward_training_base(
		O_ptr,
		v_ptr,
		bitmap_ptr,

		size1[0], // number of rays

		make_float3(bg_color_R, bg_color_G, bg_color_B),

		m_ptr,
		s_ptr,
		q_ptr,
		RGBA_ptr,
		kappa_ptr,

		T_threshold,

		depth_reg_accums_ptr,
		reg_a,
		reg_b,

		depth_and_index_ptr,
		surface_normal_ptr,
		normal_reg_accums_ptr
	);
}

// *** *** *** *** ***

void CPyOptiXFACETRenderer::Forward_inference_base(
	torch::Tensor &O,
	torch::Tensor &v,
	torch::Tensor &bitmap,

	float bg_color_R, float bg_color_G, float bg_color_B,

	torch::Tensor &m,
	torch::Tensor &s,
	torch::Tensor &q,
	torch::Tensor &RGBA,
	torch::Tensor &kappa,

	float T_threshold
) {
	float3 *O_ptr = (float3 *)O.data_ptr();
	float3 *v_ptr = (float3 *)v.data_ptr();
	float3 *bitmap_ptr = (float3 *)bitmap.data_ptr();

	float3 *m_ptr = (float3 *)m.data_ptr();
	float2 *s_ptr = (float2 *)s.data_ptr();
	float4 *q_ptr = (float4 *)q.data_ptr();
	float4 *RGBA_ptr = (float4 *)RGBA.data_ptr();
	float *kappa_ptr = (float *)kappa.data_ptr();

	c10::IntArrayRef size1 = O.sizes();

	renderer.Forward_inference_base(
		O_ptr,
		v_ptr,
		bitmap_ptr,
		nullptr,
		nullptr,
		nullptr,

		size1[0], // number of rays

		make_float3(bg_color_R, bg_color_G, bg_color_B),

		m_ptr,
		s_ptr,
		q_ptr,
		RGBA_ptr,
		kappa_ptr,

		T_threshold
	);
}

void CPyOptiXFACETRenderer::Forward_inference_base_with_geometry(
	torch::Tensor &O,
	torch::Tensor &v,
	torch::Tensor &bitmap,
	torch::Tensor &normal,
	torch::Tensor &alpha,
	torch::Tensor &expected_depth_numerator,

	float bg_color_R, float bg_color_G, float bg_color_B,

	torch::Tensor &m,
	torch::Tensor &s,
	torch::Tensor &q,
	torch::Tensor &RGBA,
	torch::Tensor &kappa,

	float T_threshold
) {
	c10::IntArrayRef size1 = O.sizes();
	renderer.Forward_inference_base(
		(float3 *)O.data_ptr(),
		(float3 *)v.data_ptr(),
		(float3 *)bitmap.data_ptr(),
		(float3 *)normal.data_ptr(),
		(float *)alpha.data_ptr(),
		(float *)expected_depth_numerator.data_ptr(),

		size1[0],
		make_float3(bg_color_R, bg_color_G, bg_color_B),

		(float3 *)m.data_ptr(),
		(float2 *)s.data_ptr(),
		(float4 *)q.data_ptr(),
		(float4 *)RGBA.data_ptr(),
		(float *)kappa.data_ptr(),

		T_threshold
	);
}

// *** *** *** *** ***

// *** *** *** *** ***

namespace {

// Deterministic gradient accumulation.
//
// The backward kernels used float `atomicAdd` at addresses indexed by the
// primitive, so many rays from many warps reached the same address at once and
// the summation order was whatever the hardware produced. Float addition is not
// associative, so identical runs disagreed in the last mantissa bits, and the
// optimisation amplified that difference by a factor of four million over a
// thousand iterations. The kernels now accumulate into 64-bit fixed point,
// where addition is associative and the order stops mattering.
//
// That representation stays inside the renderer. The scratch buffers below are
// allocated, zeroed and converted here, so callers still hand over and read
// back float32 tensors and nothing outside this file needs to know the scale.

torch::Tensor fixed_point_scratch(const torch::Tensor &gradient) {
	return torch::zeros(gradient.sizes(), gradient.options().dtype(torch::kInt64));
}

bool fixed_point_guard_enabled() {
	static const bool enabled = []() {
		const char *value = std::getenv("FACET_FIXED_POINT_GUARD");
		return (value == nullptr) || (std::string(value) != "0");
	}();
	return enabled;
}

// Convert one accumulator back into the caller's float tensor. The add is an
// accumulate rather than an overwrite, matching what the kernels did when they
// wrote into the caller's buffer directly.
void fixed_point_resolve(
	torch::Tensor &gradient, const torch::Tensor &scratch, const char *name
) {
	if (scratch.numel() == 0) return;
	if (fixed_point_guard_enabled()) {
		// One synchronisation per buffer per backward. Costs a few per cent of
		// training time and is the only thing standing between a scale that has
		// been outgrown and silently wrong gradients.
		const int64_t largest = scratch.abs().max().item<int64_t>();
		if (largest > FACET_FIXED_POINT_GUARD_LIMIT)
			throw std::runtime_error(
				std::string("deterministic accumulation overflowed for ") + name +
				"; lower FACET_FIXED_POINT_SCALE in "
				"source/renderer/DeterministicAccumulationScales.h and rebuild");
	}
	gradient.add_(scratch.to(torch::kFloat32), FACET_FIXED_POINT_INVERSE_SCALE);
}

}  // namespace

// *** *** *** *** ***

void CPyOptiXFACETRenderer::Backward_base(
	torch::Tensor &O,
	torch::Tensor &v,

	float bg_color_R, float bg_color_G, float bg_color_B,

	torch::Tensor &m,
	torch::Tensor &s,
	torch::Tensor &q,
	torch::Tensor &RGBA,
	torch::Tensor &kappa,

	torch::Tensor &I,
	torch::Tensor &dL_dI,
	torch::Tensor &dL_dRGB,
	torch::Tensor &dL_dA,
	torch::Tensor &dL_d_kappa,
	torch::Tensor &dL_dm,
	torch::Tensor &dL_ds,
	torch::Tensor &dL_dq,

	float T_threshold,

	torch::Tensor &depth_reg_accums,
	torch::Tensor &depth_normal_reg_prefix_sums,
	float lambda_depth,
	float reg_a,
	float reg_b,

	torch::Tensor &depth_and_index,
	torch::Tensor &surface_normal,
	torch::Tensor &normal_reg_accums,
	float lambda_normal
) {
	float3 *O_ptr = (float3 *)O.data_ptr();
	float3 *v_ptr = (float3 *)v.data_ptr();

	float3 *m_ptr = (float3 *)m.data_ptr();
	float2 *s_ptr = (float2 *)s.data_ptr();
	float4 *q_ptr = (float4 *)q.data_ptr();
	float4 *RGBA_ptr = (float4 *)RGBA.data_ptr();
	float *kappa_ptr = (float *)kappa.data_ptr();

	float3 *I_ptr = (float3 *)I.data_ptr();
	float3 *dL_dI_ptr = (float3 *)dL_dI.data_ptr();
	// Fixed-point scratch; see fixed_point_resolve above.
	torch::Tensor dL_dRGB_fixed = fixed_point_scratch(dL_dRGB);
	torch::Tensor dL_dA_fixed = fixed_point_scratch(dL_dA);
	torch::Tensor dL_d_kappa_fixed = fixed_point_scratch(dL_d_kappa);
	torch::Tensor dL_dm_fixed = fixed_point_scratch(dL_dm);
	torch::Tensor dL_ds_fixed = fixed_point_scratch(dL_ds);
	torch::Tensor dL_dq_fixed = fixed_point_scratch(dL_dq);
	float *dL_dRGB_ptr = (float *)dL_dRGB_fixed.data_ptr();
	float *dL_dA_ptr = (float *)dL_dA_fixed.data_ptr();
	float *dL_d_kappa_ptr = (float *)dL_d_kappa_fixed.data_ptr();
	float *dL_dm_ptr = (float *)dL_dm_fixed.data_ptr();
	float *dL_ds_ptr = (float *)dL_ds_fixed.data_ptr();
	float *dL_dq_ptr = (float *)dL_dq_fixed.data_ptr();

	float4 *depth_reg_accums_ptr = (float4 *)depth_reg_accums.data_ptr();
	float4 *depth_normal_reg_prefix_sums_ptr = (float4 *)depth_normal_reg_prefix_sums.data_ptr();

	float2 *depth_and_index_ptr = (float2 *)depth_and_index.data_ptr();
	float3 *surface_normal_ptr = (float3 *)surface_normal.data_ptr();
	float4 *normal_reg_accums_ptr = (float4 *)normal_reg_accums.data_ptr();

	c10::IntArrayRef size1 = O.sizes();

	renderer.Backward_base(
		O_ptr,
		v_ptr,

		size1[0], // number of rays

		make_float3(bg_color_R, bg_color_G, bg_color_B),

		m_ptr,
		s_ptr,
		q_ptr,
		RGBA_ptr,
		kappa_ptr,

		I_ptr,
		dL_dI_ptr,
		dL_dRGB_ptr,
		dL_dA_ptr,
		dL_d_kappa_ptr,
		dL_dm_ptr,
		dL_ds_ptr,
		dL_dq_ptr,

		T_threshold,

		depth_reg_accums_ptr,
		depth_normal_reg_prefix_sums_ptr,
		lambda_depth,
		reg_a,
		reg_b,

		depth_and_index_ptr,
		surface_normal_ptr,
		normal_reg_accums_ptr,
		lambda_normal
	);

	// Convert the fixed-point accumulators back into the caller's tensors.
	fixed_point_resolve(dL_dRGB, dL_dRGB_fixed, "dL_dRGB");
	fixed_point_resolve(dL_dA, dL_dA_fixed, "dL_dA");
	fixed_point_resolve(dL_d_kappa, dL_d_kappa_fixed, "dL_d_kappa");
	fixed_point_resolve(dL_dm, dL_dm_fixed, "dL_dm");
	fixed_point_resolve(dL_ds, dL_ds_fixed, "dL_ds");
	fixed_point_resolve(dL_dq, dL_dq_fixed, "dL_dq");
}

// *** *** *** *** ***

void CPyOptiXFACETRenderer::GetMedianDepth(
	torch::Tensor &conditioning_variable,
	torch::Tensor &features_P_hit_object,

	torch::Tensor &A1,
	torch::Tensor &b1,
	torch::Tensor &A2,
	torch::Tensor &b2,
	torch::Tensor &A3,
	torch::Tensor &b3,

	torch::Tensor &O,
	torch::Tensor &v,

	torch::Tensor &m,
	torch::Tensor &s,
	torch::Tensor &q,
	torch::Tensor &RGBA,
	torch::Tensor &kappa,

	float T_threshold,

	torch::Tensor &depth_and_index
) {
	half2 *conditioning_variable_ptr = (half2 *)conditioning_variable.data_ptr();
	int conditioning_stride = conditioning_stride_of(conditioning_variable, false);
	float *features_P_hit_object_ptr = (float *)features_P_hit_object.data_ptr();

	half *A1_ptr = (half *)A1.data_ptr();
	float *b1_ptr = (float *)b1.data_ptr();
	half *A2_ptr = (half *)A2.data_ptr();
	float *b2_ptr = (float *)b2.data_ptr();
	half *A3_ptr = (half *)A3.data_ptr();
	float *b3_ptr = (float *)b3.data_ptr();

	float3 *O_ptr = (float3 *)O.data_ptr();
	float3 *v_ptr = (float3 *)v.data_ptr();

	float3 *m_ptr = (float3 *)m.data_ptr();
	float2 *s_ptr = (float2 *)s.data_ptr();
	float4 *q_ptr = (float4 *)q.data_ptr();
	float4 *RGBA_ptr = (float4 *)RGBA.data_ptr();
	float *kappa_ptr = (float *)kappa.data_ptr();

	float2 *depth_and_index_ptr = (float2 *)depth_and_index.data_ptr();

	c10::IntArrayRef size1 = O.sizes();

	renderer.GetMedianDepth(
		conditioning_variable_ptr,
		conditioning_stride,
		features_P_hit_object_ptr,

		A1_ptr,
		b1_ptr,
		A2_ptr,
		b2_ptr,
		A3_ptr,
		b3_ptr,

		O_ptr,
		v_ptr,

		size1[0], // number of rays

		m_ptr,
		s_ptr,
		q_ptr,
		RGBA_ptr,
		kappa_ptr,

		T_threshold,

		depth_and_index_ptr
	);
}

// *** *** *** *** ***

void CPyOptiXFACETRenderer::Forward_training(
	torch::Tensor &conditioning_variable,
	torch::Tensor &features_P_hit_object,

	torch::Tensor &A1,
	torch::Tensor &b1,
	torch::Tensor &A2,
	torch::Tensor &b2,
	torch::Tensor &A3,
	torch::Tensor &b3,

	torch::Tensor &O,
	torch::Tensor &v,
	torch::Tensor &bitmap,

	float bg_color_R, float bg_color_G, float bg_color_B,

	torch::Tensor &m,
	torch::Tensor &s,
	torch::Tensor &q,
	torch::Tensor &RGBA,
	torch::Tensor &kappa,

	float T_threshold,

	torch::Tensor &depth_reg_accums,
	float reg_a,
	float reg_b,

	torch::Tensor &depth_and_index,
	torch::Tensor &surface_normal,
	torch::Tensor &normal_reg_accums
) {
	half2 *conditioning_variable_ptr = (half2 *)conditioning_variable.data_ptr();
	int conditioning_stride = conditioning_stride_of(conditioning_variable, true);
	float *features_P_hit_object_ptr = (float *)features_P_hit_object.data_ptr();

	half *A1_ptr = (half *)A1.data_ptr();
	float *b1_ptr = (float *)b1.data_ptr();
	half *A2_ptr = (half *)A2.data_ptr();
	float *b2_ptr = (float *)b2.data_ptr();
	half *A3_ptr = (half *)A3.data_ptr();
	float *b3_ptr = (float *)b3.data_ptr();

	float3 *O_ptr = (float3 *)O.data_ptr();
	float3 *v_ptr = (float3 *)v.data_ptr();
	float3 *bitmap_ptr = (float3 *)bitmap.data_ptr();

	float3 *m_ptr = (float3 *)m.data_ptr();
	float2 *s_ptr = (float2 *)s.data_ptr();
	float4 *q_ptr = (float4 *)q.data_ptr();
	float4 *RGBA_ptr = (float4 *)RGBA.data_ptr();
	float *kappa_ptr = (float *)kappa.data_ptr();

	float4 *depth_reg_accums_ptr = (float4 *)depth_reg_accums.data_ptr();

	float2 *depth_and_index_ptr = (float2 *)depth_and_index.data_ptr();
	float3 *surface_normal_ptr = (float3 *)surface_normal.data_ptr();
	float4 *normal_reg_accums_ptr = (float4 *)normal_reg_accums.data_ptr();

	c10::IntArrayRef size1 = O.sizes();

	renderer.Forward_training(
		conditioning_variable_ptr,
		conditioning_stride,
		features_P_hit_object_ptr,

		A1_ptr,
		b1_ptr,
		A2_ptr,
		b2_ptr,
		A3_ptr,
		b3_ptr,

		O_ptr,
		v_ptr,
		bitmap_ptr,

		size1[0], // number of rays

		make_float3(bg_color_R, bg_color_G, bg_color_B),

		m_ptr,
		s_ptr,
		q_ptr,
		RGBA_ptr,
		kappa_ptr,

		T_threshold,

		depth_reg_accums_ptr,
		reg_a,
		reg_b,

		depth_and_index_ptr,
		surface_normal_ptr,
		normal_reg_accums_ptr
	);
}

// *** *** *** *** ***

void CPyOptiXFACETRenderer::Forward_inference(
	torch::Tensor &conditioning_variable,
	torch::Tensor &features_P_hit_object,

	torch::Tensor &A1,
	torch::Tensor &b1,
	torch::Tensor &A2,
	torch::Tensor &b2,
	torch::Tensor &A3,
	torch::Tensor &b3,

	torch::Tensor &O,
	torch::Tensor &v,
	torch::Tensor &bitmap,

	float bg_color_R, float bg_color_G, float bg_color_B,

	torch::Tensor &m,
	torch::Tensor &s,
	torch::Tensor &q,
	torch::Tensor &RGBA,
	torch::Tensor &kappa,

	float T_threshold
) {
	half2 *conditioning_variable_ptr = (half2 *)conditioning_variable.data_ptr();
	int conditioning_stride = conditioning_stride_of(conditioning_variable, false);
	float *features_P_hit_object_ptr = (float *)features_P_hit_object.data_ptr();

	half *A1_ptr = (half *)A1.data_ptr();
	float *b1_ptr = (float *)b1.data_ptr();
	half *A2_ptr = (half *)A2.data_ptr();
	float *b2_ptr = (float *)b2.data_ptr();
	half *A3_ptr = (half *)A3.data_ptr();
	float *b3_ptr = (float *)b3.data_ptr();

	float3 *O_ptr = (float3 *)O.data_ptr();
	float3 *v_ptr = (float3 *)v.data_ptr();
	float3 *bitmap_ptr = (float3 *)bitmap.data_ptr();

	float3 *m_ptr = (float3 *)m.data_ptr();
	float2 *s_ptr = (float2 *)s.data_ptr();
	float4 *q_ptr = (float4 *)q.data_ptr();
	float4 *RGBA_ptr = (float4 *)RGBA.data_ptr();
	float *kappa_ptr = (float *)kappa.data_ptr();

	c10::IntArrayRef size1 = O.sizes();

	renderer.Forward_inference(
		conditioning_variable_ptr,
		conditioning_stride,
		features_P_hit_object_ptr,

		A1_ptr,
		b1_ptr,
		A2_ptr,
		b2_ptr,
		A3_ptr,
		b3_ptr,

		O_ptr,
		v_ptr,
		bitmap_ptr,
		nullptr,
		nullptr,
		nullptr,

		size1[0], // number of rays

		make_float3(bg_color_R, bg_color_G, bg_color_B),

		m_ptr,
		s_ptr,
		q_ptr,
		RGBA_ptr,
		kappa_ptr,

		T_threshold,
		nullptr
	);
}

// *** *** *** *** ***

void CPyOptiXFACETRenderer::Forward_pca(
	torch::Tensor &conditioning_variable,
	torch::Tensor &features_P_hit_object,
	torch::Tensor &A1, torch::Tensor &b1, torch::Tensor &A2, torch::Tensor &b2,
	torch::Tensor &A3, torch::Tensor &b3, torch::Tensor &override_colors,
	torch::Tensor &O, torch::Tensor &v, torch::Tensor &bitmap,
	float bg_color_R, float bg_color_G, float bg_color_B,
	torch::Tensor &m, torch::Tensor &s, torch::Tensor &q,
	torch::Tensor &RGBA, torch::Tensor &kappa, float T_threshold
) {
	renderer.Forward_inference(
		(half2 *)conditioning_variable.data_ptr(),
		conditioning_stride_of(conditioning_variable, false),
		(float *)features_P_hit_object.data_ptr(),
		(half *)A1.data_ptr(), (float *)b1.data_ptr(),
		(half *)A2.data_ptr(), (float *)b2.data_ptr(),
		(half *)A3.data_ptr(), (float *)b3.data_ptr(),
		(float3 *)O.data_ptr(), (float3 *)v.data_ptr(), (float3 *)bitmap.data_ptr(),
		nullptr, nullptr, nullptr, O.sizes()[0],
		make_float3(bg_color_R, bg_color_G, bg_color_B),
		(float3 *)m.data_ptr(), (float2 *)s.data_ptr(), (float4 *)q.data_ptr(),
		(float4 *)RGBA.data_ptr(), (float *)kappa.data_ptr(), T_threshold,
		(float3 *)override_colors.data_ptr()
	);
}

// *** *** *** *** ***

void CPyOptiXFACETRenderer::Forward_inference_with_geometry(
	torch::Tensor &conditioning_variable,
	torch::Tensor &features_P_hit_object,

	torch::Tensor &A1,
	torch::Tensor &b1,
	torch::Tensor &A2,
	torch::Tensor &b2,
	torch::Tensor &A3,
	torch::Tensor &b3,

	torch::Tensor &O,
	torch::Tensor &v,
	torch::Tensor &bitmap,
	torch::Tensor &normal,
	torch::Tensor &alpha,
	torch::Tensor &expected_depth_numerator,

	float bg_color_R, float bg_color_G, float bg_color_B,

	torch::Tensor &m,
	torch::Tensor &s,
	torch::Tensor &q,
	torch::Tensor &RGBA,
	torch::Tensor &kappa,

	float T_threshold
) {
	c10::IntArrayRef size1 = O.sizes();
	renderer.Forward_inference(
		(half2 *)conditioning_variable.data_ptr(),
		conditioning_stride_of(conditioning_variable, false),
		(float *)features_P_hit_object.data_ptr(),

		(half *)A1.data_ptr(),
		(float *)b1.data_ptr(),
		(half *)A2.data_ptr(),
		(float *)b2.data_ptr(),
		(half *)A3.data_ptr(),
		(float *)b3.data_ptr(),

		(float3 *)O.data_ptr(),
		(float3 *)v.data_ptr(),
		(float3 *)bitmap.data_ptr(),
		(float3 *)normal.data_ptr(),
		(float *)alpha.data_ptr(),
		(float *)expected_depth_numerator.data_ptr(),

		size1[0],
		make_float3(bg_color_R, bg_color_G, bg_color_B),

		(float3 *)m.data_ptr(),
		(float2 *)s.data_ptr(),
		(float4 *)q.data_ptr(),
		(float4 *)RGBA.data_ptr(),
		(float *)kappa.data_ptr(),

		T_threshold,
		nullptr
	);
}

// *** *** *** *** ***

void CPyOptiXFACETRenderer::Backward(
	torch::Tensor &conditioning_variable,
	torch::Tensor &features_P_hit_object,

	torch::Tensor &A1,
	torch::Tensor &b1,
	torch::Tensor &A2,
	torch::Tensor &b2,
	torch::Tensor &A3,
	torch::Tensor &b3,

	torch::Tensor &O,
	torch::Tensor &v,

	float bg_color_R, float bg_color_G, float bg_color_B,

	torch::Tensor &m,
	torch::Tensor &s,
	torch::Tensor &q,
	torch::Tensor &RGBA,
	torch::Tensor &kappa,

	torch::Tensor &I,
	torch::Tensor &dL_dI,
	torch::Tensor &dL_dRGB,
	torch::Tensor &dL_dA,
	torch::Tensor &dL_d_kappa,
	torch::Tensor &dL_dw3,
	torch::Tensor &dL_db3,
	torch::Tensor &dL_dw2,
	torch::Tensor &dL_db2,
	torch::Tensor &dL_dw1,
	torch::Tensor &dL_db1,
	torch::Tensor &dL_d_conditioning,
	torch::Tensor &dL_d_deatures,
	torch::Tensor &dL_dm,
	torch::Tensor &dL_ds,
	torch::Tensor &dL_dq,

	float T_threshold,

	torch::Tensor &depth_reg_accums,
	torch::Tensor &depth_normal_reg_prefix_sums,
	float lambda_depth,
	float reg_a,
	float reg_b,

	torch::Tensor &depth_and_index,
	torch::Tensor &surface_normal,
	torch::Tensor &normal_reg_accums,
	float lambda_normal
) {
	half2 *conditioning_variable_ptr = (half2 *)conditioning_variable.data_ptr();
	int conditioning_stride = conditioning_stride_of(conditioning_variable, true);
	if ((conditioning_stride != 0) && (conditioning_stride != 48) && (dL_d_conditioning.numel() != 0) &&
		((dL_d_conditioning.dim() != 2) || (dL_d_conditioning.size(1) != 2 * conditioning_stride) || !dL_d_conditioning.is_contiguous()))
		throw std::runtime_error("dL_d_conditioning must match the compact descriptor width");
	float *features_P_hit_object_ptr = (float *)features_P_hit_object.data_ptr();

	half *A1_ptr = (half *)A1.data_ptr();
	float *b1_ptr = (float *)b1.data_ptr();
	half *A2_ptr = (half *)A2.data_ptr();
	float *b2_ptr = (float *)b2.data_ptr();
	half *A3_ptr = (half *)A3.data_ptr();
	float *b3_ptr = (float *)b3.data_ptr();

	float3 *O_ptr = (float3 *)O.data_ptr();
	float3 *v_ptr = (float3 *)v.data_ptr();

	float3 *m_ptr = (float3 *)m.data_ptr();
	float2 *s_ptr = (float2 *)s.data_ptr();
	float4 *q_ptr = (float4 *)q.data_ptr();
	float4 *RGBA_ptr = (float4 *)RGBA.data_ptr();
	float *kappa_ptr = (float *)kappa.data_ptr();

	float3 *I_ptr = (float3 *)I.data_ptr();
	float3 *dL_dI_ptr = (float3 *)dL_dI.data_ptr();
	// Fixed-point scratch; see fixed_point_resolve above.
	torch::Tensor dL_dRGB_fixed = fixed_point_scratch(dL_dRGB);
	torch::Tensor dL_dA_fixed = fixed_point_scratch(dL_dA);
	torch::Tensor dL_d_kappa_fixed = fixed_point_scratch(dL_d_kappa);
	torch::Tensor dL_dm_fixed = fixed_point_scratch(dL_dm);
	torch::Tensor dL_ds_fixed = fixed_point_scratch(dL_ds);
	torch::Tensor dL_dq_fixed = fixed_point_scratch(dL_dq);
	torch::Tensor dL_d_conditioning_fixed = fixed_point_scratch(dL_d_conditioning);
	torch::Tensor dL_d_deatures_fixed = fixed_point_scratch(dL_d_deatures);
	float *dL_dRGB_ptr = (float *)dL_dRGB_fixed.data_ptr();
	float *dL_dA_ptr = (float *)dL_dA_fixed.data_ptr();
	float *dL_d_kappa_ptr = (float *)dL_d_kappa_fixed.data_ptr();
	float *dL_dw3_ptr = (float *)dL_dw3.data_ptr();
	float *dL_db3_ptr = (float *)dL_db3.data_ptr();
	float *dL_dw2_ptr = (float *)dL_dw2.data_ptr();
	float *dL_db2_ptr = (float *)dL_db2.data_ptr();
	float *dL_dw1_ptr = (float *)dL_dw1.data_ptr();
	float *dL_db1_ptr = (float *)dL_db1.data_ptr();
	float *dL_d_conditioning_ptr = dL_d_conditioning.numel() == 0
		? nullptr : (float *)dL_d_conditioning_fixed.data_ptr();
	float *dL_d_deatures_ptr = (float *)dL_d_deatures_fixed.data_ptr();
	float *dL_dm_ptr = (float *)dL_dm_fixed.data_ptr();
	float *dL_ds_ptr = (float *)dL_ds_fixed.data_ptr();
	float *dL_dq_ptr = (float *)dL_dq_fixed.data_ptr();

	float4 *depth_reg_accums_ptr = (float4 *)depth_reg_accums.data_ptr();
	float4 *depth_normal_reg_prefix_sums_ptr = (float4 *)depth_normal_reg_prefix_sums.data_ptr();

	float2 *depth_and_index_ptr = (float2 *)depth_and_index.data_ptr();
	float3 *surface_normal_ptr = (float3 *)surface_normal.data_ptr();
	float4 *normal_reg_accums_ptr = (float4 *)normal_reg_accums.data_ptr();

	c10::IntArrayRef size1 = O.sizes();

	renderer.Backward(
		conditioning_variable_ptr,
		conditioning_stride,
		features_P_hit_object_ptr,

		A1_ptr,
		b1_ptr,
		A2_ptr,
		b2_ptr,
		A3_ptr,
		b3_ptr,

		O_ptr,
		v_ptr,

		size1[0], // number of rays

		make_float3(bg_color_R, bg_color_G, bg_color_B),

		m_ptr,
		s_ptr,
		q_ptr,
		RGBA_ptr,
		kappa_ptr,

		I_ptr,
		dL_dI_ptr,
		dL_dRGB_ptr,
		dL_dA_ptr,
		dL_d_kappa_ptr,
		dL_dw3_ptr,
		dL_db3_ptr,
		dL_dw2_ptr,
		dL_db2_ptr,
		dL_dw1_ptr,
		dL_db1_ptr,
		dL_d_conditioning_ptr,
		dL_d_deatures_ptr,
		dL_dm_ptr,
		dL_ds_ptr,
		dL_dq_ptr,

		T_threshold,

		depth_reg_accums_ptr,
		depth_normal_reg_prefix_sums_ptr,
		lambda_depth,
		reg_a,
		reg_b,

		depth_and_index_ptr,
		surface_normal_ptr,
		normal_reg_accums_ptr,
		lambda_normal
	);

	// Convert the fixed-point accumulators back into the caller's tensors.
	fixed_point_resolve(dL_dRGB, dL_dRGB_fixed, "dL_dRGB");
	fixed_point_resolve(dL_dA, dL_dA_fixed, "dL_dA");
	fixed_point_resolve(dL_d_kappa, dL_d_kappa_fixed, "dL_d_kappa");
	fixed_point_resolve(dL_dm, dL_dm_fixed, "dL_dm");
	fixed_point_resolve(dL_ds, dL_ds_fixed, "dL_ds");
	fixed_point_resolve(dL_dq, dL_dq_fixed, "dL_dq");
	fixed_point_resolve(dL_d_conditioning, dL_d_conditioning_fixed, "dL_d_conditioning");
	fixed_point_resolve(dL_d_deatures, dL_d_deatures_fixed, "dL_d_deatures");
}

// *** *** *** *** ***

