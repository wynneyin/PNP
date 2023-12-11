#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <ATen/native/cuda/thread_constants.h>
#include "ATen/native/biginteger/cuda/sppark-util/gpu_t.cuh"
#include "parameters/bls12_381.h"
#include "gen_twiddles.cuh"

namespace at { 
namespace native {
    void NTTParameters(bool inverse, int id, BLS12_381_Fr_G1* data_ptr);
}//namespace native
}//namespace at