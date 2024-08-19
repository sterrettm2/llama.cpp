#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

bool ggml_try_graph_compute_onednn(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

bool ggml_try_onednn_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, struct ggml_tensor * dst, const int ith, const int nth);

#ifdef __cplusplus
}
#endif
