#pragma once

#include <cassert>
#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include "ATen/native/biginteger/cuda/sppark-util/gpu_t.cuh"
#include "ATen/native/biginteger/cuda/ff/bls12-381.hpp"
#include "kernels/algorithm.cuh"
#include <cuda.h>

namespace at { 
namespace native {

enum class InputOutputOrder { NN, NR, RN, RR };
enum class Direction { forward, inverse };
enum class Type { standard, coset };
enum class Algorithm { GS, CT };


void compute_ntt(size_t device_id, BLS12_381_Fr_G1* inout, 
                 BLS12_381_Fr_G1* partial_twiddles,
                 BLS12_381_Fr_G1* radix_twiddles,
                 BLS12_381_Fr_G1* radix_middles,
                 BLS12_381_Fr_G1* partial_group_gen_powers,
                 BLS12_381_Fr_G1* Domain_size_inverse,
                 uint32_t lg_domain_size,
                 InputOutputOrder ntt_order,
                 Direction ntt_direction,
                 Type ntt_type);
}//namespace native
}//namespace at