#include <stddef.h>
#include <stdint.h>
#include <ATen/ATen.h>
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

static void params_zkp(int gpu_id, bool is_intt,
                  Tensor& partial_twiddles, Tensor& radix_twiddles,
                  Tensor& radix_middles,
                  Tensor& partial_group_gen_powers, Tensor& domain_size_inverse){
    AT_DISPATCH_FR_MONT_TYPES(radix_twiddles.scalar_type(), "load_ntt_params_cuda", [&] {                
        auto pt_ptr = reinterpret_cast<BLS12_381_Fr_G1 (*)[WINDOW_SIZE]>(partial_twiddles.mutable_data_ptr<scalar_t>());
        auto rp_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(radix_twiddles.mutable_data_ptr<scalar_t>());
        auto rpm_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(radix_middles.mutable_data_ptr<scalar_t>());
        auto prp_ptr = reinterpret_cast<BLS12_381_Fr_G1 (*)[WINDOW_SIZE]>(partial_group_gen_powers.mutable_data_ptr<scalar_t>());
        auto size_inverse_ptr = domain_size_inverse.mutable_data_ptr<uint64_t>();

        NTTParameters ntt_parameters(is_intt,gpu_id);
        ntt_parameters.initNTTParameters(pt_ptr,
                                     rp_ptr+64+128+256+512, rp_ptr, rp_ptr+64, rp_ptr+64+128, rp_ptr+64+128+256,
                                     rpm_ptr, rpm_ptr+64*64, rpm_ptr+64*64+4096*64, rpm_ptr+64*64+4096*64+128*128, rpm_ptr+64*64+4096*64+128*128+256*256,
                                     prp_ptr, size_inverse_ptr);
    });
}

TensorList params_zkp_cuda(int64_t domain_size, int64_t gpu_id, bool is_intt) {
    // Tensor partial_twiddles = params[0];
    // Tensor radix_twiddles = params[1];
    // Tensor radix_middles = params[2];
    // Tensor partial_group_gen_powers = params[3];
    // Tensor domain_size_inverse = params[4];

    TensorList params;
    auto windows_size = 1<<14; //LG_WINDOW_SIZE ((MAX_LG_DOMAIN_SIZE + 1) / 2) 

    auto partial_twiddles = zeros({windows_size, 4}, kBLS12_381_Fq_G1_Mont);
    auto radix_twiddles = zeros({64 + 128 + 256 + 512 + 32, 4}, kBLS12_381_Fq_G1_Mont);
    auto radix_middles = zeros({64*64 + 4096*64 + 128*128 + 256*256 + 512*512, 4}, kBLS12_381_Fq_G1_Mont);
    auto partial_group_gen_powers = zeros({windows_size, 4}, kBLS12_381_Fq_G1_Mont);
    auto domain_size_inverse = zeros({domain_size+1, 4} , kULong);

    params_zkp(gpu_id, is_intt,
              partial_twiddles, radix_twiddles, radix_middles,
              partial_group_gen_powers, domain_size_inverse);
    return params;
}

// Tensor& params_zkp_cuda_(int gpu_id, bool is_intt,
//                   Tensor& partial_twiddles, Tensor& radix_twiddles,
//                   Tensor& radix_middle,
//                   Tensor& partial_group_gen_powers, Tensor& Domain_size_inverse) {
//     params_zkp(gpu_id, is_intt, window_size,
//               partial_twiddles, radix_twiddles,
//               radix_middle,
//               partial_group_gen_powers, Domain_size_inverse);
//     return inout;
// }

// Tensor& params_zkp_out_cuda(const Tensor& inout, Tensor& output) {
//     copy(output, inout);
//     ntt_zkp(output);
//     return output;
// }
// static void ntt_zkp_v2(Tensor& inout, TensorList params, int gpu_id, bool is_intt, int window_size) {
//     auto len = inout.numel() / num_uint64(inout.scalar_type());
//     uint32_t lg_domain_size = log2(len);
//     TORCH_CHECK(len == 1<<lg_domain_size, "NTT Length check!");
//     Tensor partial_twiddles = params[0];
//     Tensor radix_twiddles = params[1];
//     Tensor radix_middles = params[2];
//     Tensor partial_group_gen_powers = params[3];
//     Tensor domain_size_inverse = params[4];
//     AT_DISPATCH_FR_MONT_TYPES(inout.scalar_type(), "ntt_cuda", [&] {
//         auto self_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(inout.mutable_data_ptr<scalar_t>());
//         auto pt_ptr = reinterpret_cast<BLS12_381_Fr_G1 (*)[window_size]>(partial_twiddles.mutable_data_ptr<scalar_t>());
//         auto rp_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(radix_twiddles.mutable_data_ptr<scalar_t>());
//         auto rpm_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(radix_middles.mutable_data_ptr<scalar_t>());
//         auto prp_ptr = reinterpret_cast<BLS12_381_Fr_G1 (*)[window_size]>(partial_group_gen_powers.mutable_data_ptr<scalar_t>());
//         auto size_inverse_ptr = domain_size_inverse.mutable_data_ptr<uint32_t>();

//         NTTParameters ntt_parameters(pt_ptr, 
//                                         rp_ptr+64+128+256+512, rp_ptr, rp_ptr+64, rp_ptr+64+128, rp_ptr+64+128+256,
//                                         rpm_ptr, rpm_ptr+64*64, rpm_ptr+64*64+4096*64, rpm_ptr+64*64+4096*64+128*128, rpm_ptr+64*64+4096*64+128*128+256*256,
//                                         prp_ptr, size_inverse_ptr);
//         compute_ntt(
//             0,
//             self_ptr,
//             lg_domain_size,  
//             InputOutputOrder::NN,
//             Direction::forward,
//             Type::standard
//         );
//         C10_CUDA_KERNEL_LAUNCH_CHECK();
//     });
// }


static void ntt_zkp(Tensor& inout,
                    Tensor& partial_twiddles, Tensor& radix_twiddles,
                    Tensor& radix_middles,
                    Tensor& partial_group_gen_powers, Tensor& domain_size_inverse) {
    auto len = inout.numel() / num_uint64(inout.scalar_type());
    uint32_t lg_domain_size = log2(len);
    TORCH_CHECK(len == 1<<lg_domain_size, "NTT Length check!");
    AT_DISPATCH_FR_MONT_TYPES(inout.scalar_type(), "ntt_cuda", [&] {
        auto pt_ptr = reinterpret_cast<BLS12_381_Fr_G1 (*)[WINDOW_SIZE]>(partial_twiddles.mutable_data_ptr<scalar_t>());
        auto rp_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(radix_twiddles.mutable_data_ptr<scalar_t>());
        auto rpm_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(radix_middles.mutable_data_ptr<scalar_t>());
        auto prp_ptr = reinterpret_cast<BLS12_381_Fr_G1 (*)[WINDOW_SIZE]>(partial_group_gen_powers.mutable_data_ptr<scalar_t>());
        auto size_inverse_ptr = domain_size_inverse.mutable_data_ptr<uint64_t>();
        auto self_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(inout.mutable_data_ptr<scalar_t>());
        
        compute_ntt(
            0,
            self_ptr,
            pt_ptr,
            rp_ptr,
            rpm_ptr,
            prp_ptr,
            size_inverse_ptr,
            lg_domain_size,  
            InputOutputOrder::NN,
            Direction::forward,
            Type::standard
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

static void intt_zkp(Tensor& inout,
                    Tensor& partial_twiddles, Tensor& radix_twiddles,
                    Tensor& radix_middles,
                    Tensor& partial_group_gen_powers, Tensor& domain_size_inverse) {
    auto len = inout.numel() / num_uint64(inout.scalar_type());
    uint32_t lg_domain_size = log2(len);
    TORCH_CHECK(len == 1<<lg_domain_size, "NTT Length check!");
    AT_DISPATCH_FR_MONT_TYPES(inout.scalar_type(), "intt_cuda", [&] {
        auto pt_ptr = reinterpret_cast<BLS12_381_Fr_G1 (*)[WINDOW_SIZE]>(partial_twiddles.mutable_data_ptr<scalar_t>());
        auto rp_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(radix_twiddles.mutable_data_ptr<scalar_t>());
        auto rpm_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(radix_middles.mutable_data_ptr<scalar_t>());
        auto prp_ptr = reinterpret_cast<BLS12_381_Fr_G1 (*)[WINDOW_SIZE]>(partial_group_gen_powers.mutable_data_ptr<scalar_t>());
        auto size_inverse_ptr = domain_size_inverse.mutable_data_ptr<uint64_t>();
        auto self_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(inout.mutable_data_ptr<scalar_t>());
        compute_ntt(
            0,
            self_ptr,
            pt_ptr,
            rp_ptr,
            rpm_ptr,
            prp_ptr,
            size_inverse_ptr,
            lg_domain_size,  
            InputOutputOrder::NN,
            Direction::inverse,
            Type::standard
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

static void ntt_coset_zkp(Tensor& inout,
                    Tensor& partial_twiddles, Tensor& radix_twiddles,
                    Tensor& radix_middles,
                    Tensor& partial_group_gen_powers, Tensor& domain_size_inverse) {
    auto len = inout.numel() / num_uint64(inout.scalar_type());
    uint32_t lg_domain_size = log2(len);
    TORCH_CHECK(len == 1<<lg_domain_size, "NTT Length check!");
    AT_DISPATCH_FR_MONT_TYPES(inout.scalar_type(), "ntt_coset_cuda", [&] {
        auto pt_ptr = reinterpret_cast<BLS12_381_Fr_G1 (*)[WINDOW_SIZE]>(partial_twiddles.mutable_data_ptr<scalar_t>());
        auto rp_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(radix_twiddles.mutable_data_ptr<scalar_t>());
        auto rpm_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(radix_middles.mutable_data_ptr<scalar_t>());
        auto prp_ptr = reinterpret_cast<BLS12_381_Fr_G1 (*)[WINDOW_SIZE]>(partial_group_gen_powers.mutable_data_ptr<scalar_t>());
        auto size_inverse_ptr = domain_size_inverse.mutable_data_ptr<uint64_t>();
        auto self_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(inout.mutable_data_ptr<scalar_t>());
        compute_ntt(
            0,
            self_ptr,
            pt_ptr,
            rp_ptr,
            rpm_ptr,
            prp_ptr,
            size_inverse_ptr,
            lg_domain_size,  
            InputOutputOrder::NN,
            Direction::forward,
            Type::coset
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

static void intt_coset_zkp(Tensor& inout,
                    Tensor& partial_twiddles, Tensor& radix_twiddles,
                    Tensor& radix_middles,
                    Tensor& partial_group_gen_powers, Tensor& domain_size_inverse) {
    auto len = inout.numel() / num_uint64(inout.scalar_type());
    uint32_t lg_domain_size = log2(len);
    TORCH_CHECK(len == 1<<lg_domain_size, "NTT Length check!");
    AT_DISPATCH_FR_MONT_TYPES(inout.scalar_type(), "intt_coset_cuda", [&] {
        auto pt_ptr = reinterpret_cast<BLS12_381_Fr_G1 (*)[WINDOW_SIZE]>(partial_twiddles.mutable_data_ptr<scalar_t>());
        auto rp_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(radix_twiddles.mutable_data_ptr<scalar_t>());
        auto rpm_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(radix_middles.mutable_data_ptr<scalar_t>());
        auto prp_ptr = reinterpret_cast<BLS12_381_Fr_G1 (*)[WINDOW_SIZE]>(partial_group_gen_powers.mutable_data_ptr<scalar_t>());
        auto size_inverse_ptr = domain_size_inverse.mutable_data_ptr<uint64_t>();
        auto self_ptr = reinterpret_cast<BLS12_381_Fr_G1*>(inout.mutable_data_ptr<scalar_t>());
        compute_ntt(
            0,
            self_ptr,
            pt_ptr,
            rp_ptr,
            rpm_ptr,
            prp_ptr,
            size_inverse_ptr,
            lg_domain_size,  
            InputOutputOrder::NN,
            Direction::inverse,
            Type::coset
        );
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

Tensor ntt_zkp_cuda(const Tensor& inout, TensorList params) {
    Tensor output = inout.clone();
    Tensor partial_twiddles = params[0];
    Tensor radix_twiddles = params[1];
    Tensor radix_middles = params[2];
    Tensor partial_group_gen_powers = params[3];
    Tensor domain_size_inverse = params[4];   
    ntt_zkp(output,
            partial_twiddles, radix_twiddles, radix_middles,
            partial_group_gen_powers, domain_size_inverse);
    return output;
}

Tensor& ntt_zkp_cuda_(Tensor& inout, TensorList params) {
    Tensor partial_twiddles = params[0];
    Tensor radix_twiddles = params[1];
    Tensor radix_middles = params[2];
    Tensor partial_group_gen_powers = params[3];
    Tensor domain_size_inverse = params[4];                    
    ntt_zkp(inout,
            partial_twiddles, radix_twiddles, radix_middles,
            partial_group_gen_powers, domain_size_inverse);
    return inout;
}

Tensor& ntt_zkp_out_cuda(const Tensor& inout, TensorList params, Tensor& output) {
    copy(output, inout);
    Tensor partial_twiddles = params[0];
    Tensor radix_twiddles = params[1];
    Tensor radix_middles = params[2];
    Tensor partial_group_gen_powers = params[3];
    Tensor domain_size_inverse = params[4];       
    ntt_zkp(output,
            partial_twiddles, radix_twiddles, radix_middles,
            partial_group_gen_powers, domain_size_inverse);
    return output;
}

Tensor intt_zkp_cuda(const Tensor& inout, TensorList params) {
    Tensor output = inout.clone();
    Tensor partial_twiddles = params[0];
    Tensor radix_twiddles = params[1];
    Tensor radix_middles = params[2];
    Tensor partial_group_gen_powers = params[3];
    Tensor domain_size_inverse = params[4];
    intt_zkp(output,
             partial_twiddles, radix_twiddles, radix_middles,
             partial_group_gen_powers, domain_size_inverse);
    return output;
}

Tensor& intt_zkp_cuda_(Tensor& inout, TensorList params) {
    Tensor partial_twiddles = params[0];
    Tensor radix_twiddles = params[1];
    Tensor radix_middles = params[2];
    Tensor partial_group_gen_powers = params[3];
    Tensor domain_size_inverse = params[4];                           
    intt_zkp(inout,
             partial_twiddles, radix_twiddles, radix_middles,
             partial_group_gen_powers, domain_size_inverse);
    return inout;
}

Tensor& intt_zkp_out_cuda(const Tensor& inout, TensorList params, Tensor& output) {
    Tensor partial_twiddles = params[0];
    Tensor radix_twiddles = params[1];
    Tensor radix_middles = params[2];
    Tensor partial_group_gen_powers = params[3];
    Tensor domain_size_inverse = params[4]; 
    copy(output, inout);
    intt_zkp(output,
             partial_twiddles, radix_twiddles, radix_middles,
             partial_group_gen_powers, domain_size_inverse);
    return output;
}

Tensor ntt_coset_zkp_cuda(const Tensor& inout, TensorList params) {
    Tensor output = inout.clone();
    Tensor partial_twiddles = params[0];
    Tensor radix_twiddles = params[1];
    Tensor radix_middles = params[2];
    Tensor partial_group_gen_powers = params[3];
    Tensor domain_size_inverse = params[4];
    ntt_coset_zkp(output,
                  partial_twiddles, radix_twiddles, radix_middles,
                  partial_group_gen_powers, domain_size_inverse);
    return output;
}

Tensor& ntt_coset_zkp_cuda_(Tensor& inout, TensorList params) {
    Tensor partial_twiddles = params[0];
    Tensor radix_twiddles = params[1];
    Tensor radix_middles = params[2];
    Tensor partial_group_gen_powers = params[3];
    Tensor domain_size_inverse = params[4];     
    ntt_coset_zkp(inout,
                  partial_twiddles, radix_twiddles, radix_middles,
                  partial_group_gen_powers, domain_size_inverse);
    return inout;
}

Tensor& ntt_coset_zkp_out_cuda(const Tensor& inout, TensorList params, Tensor& output) {
    Tensor partial_twiddles = params[0];
    Tensor radix_twiddles = params[1];
    Tensor radix_middles = params[2];
    Tensor partial_group_gen_powers = params[3];
    Tensor domain_size_inverse = params[4];
    copy(output, inout);
    ntt_coset_zkp(output,
                  partial_twiddles, radix_twiddles, radix_middles,
                  partial_group_gen_powers, domain_size_inverse);
    return output;
}

Tensor intt_coset_zkp_cuda(const Tensor& inout, TensorList params) {
    Tensor output = inout.clone();
    Tensor partial_twiddles = params[0];
    Tensor radix_twiddles = params[1];
    Tensor radix_middles = params[2];
    Tensor partial_group_gen_powers = params[3];
    Tensor domain_size_inverse = params[4];
    intt_coset_zkp(output,
                   partial_twiddles, radix_twiddles, radix_middles,
                   partial_group_gen_powers, domain_size_inverse);
    return output;
}

Tensor& intt_coset_zkp_cuda_(Tensor& inout, TensorList params) {
    Tensor partial_twiddles = params[0];
    Tensor radix_twiddles = params[1];
    Tensor radix_middles = params[2];
    Tensor partial_group_gen_powers = params[3];
    Tensor domain_size_inverse = params[4];    
    intt_coset_zkp(inout,
                   partial_twiddles, radix_twiddles, radix_middles,
                   partial_group_gen_powers, domain_size_inverse);
    return inout;
}

Tensor& intt_coset_zkp_out_cuda(const Tensor& inout, TensorList params, Tensor& output) {
    Tensor partial_twiddles = params[0];
    Tensor radix_twiddles = params[1];
    Tensor radix_middles = params[2];
    Tensor partial_group_gen_powers = params[3];
    Tensor domain_size_inverse = params[4]; 
    copy(output, inout);
    intt_coset_zkp(output,
                   partial_twiddles, radix_twiddles, radix_middles,
                   partial_group_gen_powers, domain_size_inverse);
    return output;
}

}//namespace native
}//namespace at