#include <stddef.h>
#include <stdint.h>

#include <ATen/native/biginteger/cuda/sppark-ntt/ntt.cuh>
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorOperators.h>
#include <ATen/core/TensorBody.h>
#include <ATen/core/interned_strings.h>
#include <ATen/ops/copy.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>

#include <math.h>

//temporarily set device_id to 0, set InputOutputOrder to NN
namespace at {
namespace native {

// typedef enum {
//     NN = 0,
//     NR = 1,
//     RN = 2,
//     RR = 3,
// } NTTInputOutputOrder;

// typedef enum {
//     Forward = 0,
//     Inverse = 1,
// } NTTDirection;

// typedef enum {
//     Standard = 0,
//     Coset = 1,
// } NTTType;


// extern RustError compute_ntt(
//     size_t device_id,
//     void* inout,
//     uint32_t lg_domain_size,
//     NTTInputOutputOrder ntt_order,
//     NTTDirection ntt_direction,
//     NTTType ntt_type
// );
// Tensor ntt_zkp_gpu(const Tensor& inout, NTTInputOutputOrder order);
// Tensor intt_zkp_gpu(const Tensor& inout, NTTInputOutputOrder order);
// Tensor coset_ntt_zkp_gpu(const Tensor& inout, NTTInputOutputOrder order);
// Tensor coset_intt_zkp_gpu(const Tensor& inout, NTTInputOutputOrder order);


Tensor ntt_zkp_gpu(const Tensor& inout) {
    AT_DISPATCH_FR_MONT_TYPES(inout.scalar_type(), "ntt_cuda", [&] {
        auto len = inout.numel() / 4;
        auto self_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(inout.mutable_data_ptr<scalar_t>());
        // auto self_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(inout.mutable_data_ptr<scalar_t>());
        uint32_t lg_domain_size = log2(len);
        RustError err = compute_ntt(
            0,
            self_ptr,
            lg_domain_size,  
            NTT::InputOutputOrder::NN,
            NTT::Direction::forward,
            NTT::Type::standard
        );
    });

    return inout;
}

template <typename T>
__global__ void to_base_kernel(const int64_t N, T* data) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < N) {
    data[i].from();
  }
}

Tensor intt_zkp_gpu(const Tensor& inout) {
    Tensor output = inout.clone();
    std::cout<<"call intt on gpu"<<std::endl;
    auto len = output.numel()/4;
    AT_DISPATCH_FR_MONT_TYPES(output.scalar_type(), "to_mont_cuda", [&] {
        auto self_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(output.mutable_data_ptr<scalar_t>());
        uint32_t lg_domain_size = log2(len);
        RustError err = compute_ntt(
            0,
            self_ptr,
            lg_domain_size,
            NTT::InputOutputOrder::NN,
            NTT::Direction::inverse,
            NTT::Type::standard
        );

        // int64_t N = output.numel() / num_uint64(output.scalar_type());
        // TORCH_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
        // int64_t grid = (N + block_work_size() - 1) / block_work_size();
        // auto stream = at::cuda::getCurrentCUDAStream();
        // to_base_kernel<<<grid, num_threads(), 0, stream>>>(N, self_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();


    });
    return output;
}

Tensor coset_ntt_zkp_gpu(const Tensor& inout) {
    auto len = inout.numel() / 4;
    AT_DISPATCH_FR_MONT_TYPES(inout.scalar_type(), "to_mont_cuda", [&] {
    auto self_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(inout.mutable_data_ptr<scalar_t>());
    uint32_t lg_domain_size = log2(len);
    RustError err = compute_ntt(
        0,
        self_ptr,
        lg_domain_size,
        NTT::InputOutputOrder::NN,
        NTT::Direction::forward,
        NTT::Type::coset
    );
    });
    return inout;
}

Tensor coset_intt_zkp_gpu(const Tensor& inout) {
    auto len = inout.numel() / 4;
    AT_DISPATCH_FR_MONT_TYPES(inout.scalar_type(), "to_mont_cuda", [&] {
    auto self_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(inout.mutable_data_ptr<scalar_t>());
    uint32_t lg_domain_size = log2(len);
    RustError err = compute_ntt(
        0,
        self_ptr,
        lg_domain_size,
        NTT::InputOutputOrder::NN,
        NTT::Direction::inverse,
        NTT::Type::coset
    );
    });
    return inout;
}
}
}