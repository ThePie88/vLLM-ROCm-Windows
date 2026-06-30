// Minimal torch.ops._C bindings for the first batch of vLLM fused ops, built 1:1 from
// vLLM's own csrc kernels on native Windows. Schemas copied verbatim from
// csrc/torch_bindings.cpp so vLLM's call sites dispatch here unchanged.
#include <torch/library.h>
#include <torch/extension.h>

#include "ops.h"  // vLLM declarations (silu_and_mul, rms_norm, fused_add_rms_norm, rotary_embedding)

TORCH_LIBRARY(_C, m) {
  m.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
  m.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);

  m.def("rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> ()");
  m.impl("rms_norm", torch::kCUDA, &rms_norm);

  m.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor weight, float epsilon) "
      "-> ()");
  m.impl("fused_add_rms_norm", torch::kCUDA, &fused_add_rms_norm);

  m.def(
      "rotary_embedding(Tensor positions, Tensor! query, Tensor!? key, int head_size, "
      "Tensor cos_sin_cache, bool is_neox) -> ()");
  m.impl("rotary_embedding", torch::kCUDA, &rotary_embedding);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
