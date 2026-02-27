#include <sstream>

#include "src/array.h"
#include "src/stream.h"

namespace fast_ops {

// JS-friendly signature for the callable returned by metalKernel.
// The raw MetalKernelFunction has required params (vector<pair>, bool) that
// kizunapi can't auto-omit. This wrapper makes them optional.
using JsKernelFunction = std::function<std::vector<mx::array>(
    const std::vector<mx::array>&,
    const std::vector<mx::Shape>&,
    const std::vector<mx::Dtype>&,
    std::tuple<int, int, int>,
    std::tuple<int, int, int>,
    std::optional<std::vector<std::pair<std::string, mx::fast::TemplateArg>>>,
    std::optional<float>,
    std::optional<bool>,
    mx::StreamOrDevice)>;

JsKernelFunction MetalKernel(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    std::optional<std::string> header,
    std::optional<bool> ensure_row_contiguous,
    std::optional<bool> atomic_outputs) {
  auto kernel = mx::fast::metal_kernel(
      name, input_names, output_names, source,
      header.value_or(""),
      ensure_row_contiguous.value_or(true),
      atomic_outputs.value_or(false));
  return [kernel = std::move(kernel)](
      const std::vector<mx::array>& inputs,
      const std::vector<mx::Shape>& output_shapes,
      const std::vector<mx::Dtype>& output_dtypes,
      std::tuple<int, int, int> grid,
      std::tuple<int, int, int> threadgroup,
      std::optional<std::vector<std::pair<std::string, mx::fast::TemplateArg>>> template_args,
      std::optional<float> init_value,
      std::optional<bool> verbose,
      mx::StreamOrDevice s) -> std::vector<mx::array> {
    return kernel(
        inputs, output_shapes, output_dtypes, grid, threadgroup,
        template_args.value_or(std::vector<std::pair<std::string, mx::fast::TemplateArg>>{}),
        init_value,
        verbose.value_or(false),
        s);
  };
}

mx::array Rope(const mx::array& x,
               int dims,
               bool traditional,
               std::optional<float> base,
               float scale,
               std::variant<int, mx::array> offset,
               const std::optional<mx::array>& freqs,
               mx::StreamOrDevice s) {
  if (std::holds_alternative<int>(offset)) {
    return mx::fast::rope(x, dims, traditional, base, scale,
                          std::get<int>(offset), freqs, s);
  } else {
    return mx::fast::rope(x, dims, traditional, base, scale,
                          std::get<mx::array>(offset), freqs, s);
  }
}

mx::array ScaledDotProductAttention(
    const mx::array& queries,
    const mx::array& keys,
    const mx::array& values,
    const float scale,
    const std::variant<std::monostate, std::string, mx::array>& mask,
    mx::StreamOrDevice s) {
  bool has_mask = !std::holds_alternative<std::monostate>(mask);
  bool has_str_mask =
      has_mask && std::holds_alternative<std::string>(mask);
  bool has_arr_mask = has_mask && std::holds_alternative<mx::array>(mask);

  if (has_mask) {
    if (has_str_mask) {
      auto mask_str = std::get<std::string>(mask);
      if (mask_str != "causal") {
        std::ostringstream msg;
        msg << "[scaled_dot_product_attention] invalid mask option '"
            << mask_str << "'. Must be 'causal', or an array.";
        throw std::invalid_argument(msg.str());
      }
      return mx::fast::scaled_dot_product_attention(
          queries, keys, values, scale, mask_str, {}, s);
    } else {
      auto mask_arr = std::get<mx::array>(mask);
      return mx::fast::scaled_dot_product_attention(
          queries, keys, values, scale, "", {mask_arr}, s);
    }

  } else {
    return mx::fast::scaled_dot_product_attention(
        queries, keys, values, scale, "", {}, s);
  }
}

}  // namespace fast_ops

void InitFast(napi_env env, napi_value exports) {
  napi_value fast = ki::CreateObject(env);
  ki::Set(env, exports, "fast", fast);

  ki::Set(env, fast,
          "rmsNorm", &mx::fast::rms_norm,
          "layerNorm", &mx::fast::layer_norm,
          "rope", &fast_ops::Rope,
          "scaledDotProductAttention", &fast_ops::ScaledDotProductAttention,
          "metalKernel", &fast_ops::MetalKernel);
}
