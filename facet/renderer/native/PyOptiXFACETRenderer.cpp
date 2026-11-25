#ifdef _WIN32
	#define NOMINMAX

	#include "framework.h"
#endif

#include "CPyOptiXFACETRenderer.h"
#include "GenerateRays.h"
#include "constants.h"

// *** *** *** *** ***

PYBIND11_MODULE(PYOPTIXFACETRENDERER, m) {
	m.doc() = "Python OptiX FACET Renderer";
	m.attr("HIT_BUFFER_SIZE") = HIT_BUFFER_SIZE;

	// *********************************************************************************************

	pybind11::class_<CPyOptiXFACETRenderer>(m, "CPyOptiXFACETRenderer")
		.def(pybind11::init<int, float, int>(),
			py::arg("number_of_sides"),
			py::arg("chi_square_squared_radius"),
			py::arg("max_batch_size")
		)
		.def("SetGeometry", &CPyOptiXFACETRenderer::SetGeometry, "Sets the means, scales and the quaternions of the Gaussians")
		.def("Render", &CPyOptiXFACETRenderer::Render, "Computes the t parameters and the indices of the Gaussians intersected by the rays")
		.def("TraceHits", &CPyOptiXFACETRenderer::TraceHits, "Computes retained hits and total candidate counts for diagnostics")
		.def("GetMedianDepth_base",
			&CPyOptiXFACETRenderer::GetMedianDepth_base,
			"Gets the transmittance-median depth and primitive index for the base model"
		)
		
		.def("Forward_training_base",
			&CPyOptiXFACETRenderer::Forward_training_base,
			"Renders the image for the base model training phase"
		)
		.def("Forward_inference_base",
			&CPyOptiXFACETRenderer::Forward_inference_base,
			"Renders the image for the base model inference phase"
		)
		.def("Forward_inference_base_with_geometry",
			&CPyOptiXFACETRenderer::Forward_inference_base_with_geometry,
			"Renders base RGB, alpha, expected depth, and composited normals"
		)
		.def("Backward_base",
			&CPyOptiXFACETRenderer::Backward_base,
			"Computes the base model gradient"
		)


		.def("GetMedianDepth",
			&CPyOptiXFACETRenderer::GetMedianDepth,
			"Gets the transmittance-median depth and primitive index for FACET"
		)
		.def("Forward_training",
			&CPyOptiXFACETRenderer::Forward_training,
			"Renders the image for the training phase"
		)
		.def("Forward_inference",
			&CPyOptiXFACETRenderer::Forward_inference,
			"Renders the image for the inference phase"
		)
		.def("Forward_pca",
			&CPyOptiXFACETRenderer::Forward_pca,
			"Renders PCA colors for per-Gaussian appearance embeddings"
		)
		.def("Forward_inference_with_geometry",
			&CPyOptiXFACETRenderer::Forward_inference_with_geometry,
			"Renders FACET RGB, alpha, expected depth, and composited normals"
		)
		.def("Backward",
			&CPyOptiXFACETRenderer::Backward,
			"Computes the gradient"
		);

	// *********************************************************************************************

	m.def("GenerateRays", &GenerateRays, "Generates the direction vectors of the rays");
}
