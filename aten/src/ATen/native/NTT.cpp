#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/Activation.h>
#include <ATen/native/miopen/ntt_zkp.c>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <ATen/ScalarOps.h>
#if defined(C10_MOBILE) && defined(USE_XNNPACK)
#include <ATen/native/xnnpack/Engine.h>
#endif
#include <ATen/core/DistributionsHelper.h>

#include <c10/util/irange.h>
#include <c10/core/ScalarType.h>
#if AT_MKLDNN_ENABLED()
#include <ATen/native/mkldnn/MKLDNNCommon.h>
#include <ATen/native/mkldnn/Utils.h>
#endif

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/celu_native.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/clamp_min.h>
#include <ATen/ops/elu.h>
#include <ATen/ops/elu_backward_native.h>
#include <ATen/ops/elu_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/gelu_backward_native.h>
#include <ATen/ops/gelu_native.h>
#include <ATen/ops/hardshrink_backward_native.h>
#include <ATen/ops/hardshrink_native.h>
#include <ATen/ops/hardsigmoid_backward_native.h>
#include <ATen/ops/hardsigmoid_native.h>
#include <ATen/ops/hardswish_backward_native.h>
#include <ATen/ops/hardswish_native.h>
#include <ATen/ops/hardtanh.h>
#include <ATen/ops/hardtanh_backward_native.h>
#include <ATen/ops/hardtanh_native.h>
#include <ATen/ops/infinitely_differentiable_gelu_backward_native.h>
#include <ATen/ops/leaky_relu.h>
#include <ATen/ops/leaky_relu_backward.h>
#include <ATen/ops/leaky_relu_backward_native.h>
#include <ATen/ops/leaky_relu_native.h>
#include <ATen/ops/log_sigmoid_backward_native.h>
#include <ATen/ops/log_sigmoid_forward.h>
#include <ATen/ops/log_sigmoid_forward_native.h>
#include <ATen/ops/log_sigmoid_native.h>
#include <ATen/ops/mish_backward_native.h>
#include <ATen/ops/mish_native.h>
#include <ATen/ops/prelu_native.h>
#include <ATen/ops/_prelu_kernel.h>
#include <ATen/ops/_prelu_kernel_native.h>
#include <ATen/ops/_prelu_kernel_backward_native.h>
#include <ATen/ops/relu6_native.h>
#include <ATen/ops/relu_native.h>
#include <ATen/ops/rrelu_native.h>
#include <ATen/ops/rrelu_with_noise.h>
#include <ATen/ops/rrelu_with_noise_backward_native.h>
#include <ATen/ops/rrelu_with_noise_native.h>
#include <ATen/ops/selu_native.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/silu_backward_native.h>
#include <ATen/ops/silu_native.h>
#include <ATen/ops/softplus.h>
#include <ATen/ops/softplus_backward_native.h>
#include <ATen/ops/softplus_native.h>
#include <ATen/ops/softshrink_backward_native.h>
#include <ATen/ops/softshrink_native.h>
#include <ATen/ops/tanh.h>
#include <ATen/ops/threshold_backward_native.h>
#include <ATen/ops/threshold_native.h>
#include <ATen/ops/zeros_like.h>

#include <utility>
#include <vector>
#endif

#include <execinfo.h>
#include <stdlib.h>
#include <math.h>
namespace at {
namespace native {
Tensor ntt_zkp_cpu(const Tensor& input);
Tensor intt_zkp_cpu(const Tensor& input);
Tensor coset_ntt_zkp_cpu(const Tensor& input);
Tensor coset_intt_zkp_cpu(const Tensor& input);

Tensor ntt_zkp_cpu(const Tensor& input) {
    auto out_sizes = input.numel();
    auto ptr = input.mutable_data_ptr<uint64_t>();
    NTT(ptr, true, out_sizes);
    return input;
}

Tensor intt_zkp_cpu(const Tensor& input) {
    std::cout<<"call intt on cpu"<<std::endl;
    auto out_sizes = input.numel();
    auto ptr = input.mutable_data_ptr<uint64_t>();
    iNTT(ptr, false, out_sizes);

    return input;
}

Tensor coset_ntt_zkp_cpu(const Tensor& input) {
    auto out_sizes = input.numel();
    auto ptr = input.mutable_data_ptr<uint64_t>();
    NTT_coset(ptr, true, out_sizes);
    return input;
}

Tensor coset_intt_zkp_cpu(const Tensor& input) {
    auto out_sizes = input.numel();
    auto ptr = input.mutable_data_ptr<uint64_t>();
    iNTT_coset(ptr, false, out_sizes);

    return input;
}

}}  // namespace at::native
