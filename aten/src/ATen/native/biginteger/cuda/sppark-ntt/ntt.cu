#include "ATen/native/biginteger/cuda/sppark-ntt/ntt.cuh"

// #ifndef __CUDA_ARCH__
namespace at { 
namespace native {
    
void NTT::bit_rev(BLS12_381_Fr_G1* d_out, const BLS12_381_Fr_G1* d_inp,
                        uint32_t lg_domain_size, stream_t& stream)
{
    assert(lg_domain_size <= MAX_LG_DOMAIN_SIZE);

    size_t domain_size = (size_t)1 << lg_domain_size;

    if (domain_size <= WARP_SZ)
        bit_rev_permutation
            <<<1, domain_size, 0, stream>>>
            (d_out, d_inp, lg_domain_size);
    else if (d_out == d_inp)
        bit_rev_permutation
            <<<domain_size/WARP_SZ, WARP_SZ, 0, stream>>>
            (d_out, d_inp, lg_domain_size);
    else if (domain_size < 1024)
        bit_rev_permutation_aux
            <<<1, domain_size / 8, domain_size * sizeof(BLS12_381_Fr_G1), stream>>>
            (d_out, d_inp, lg_domain_size);
    else
        bit_rev_permutation_aux
            <<<domain_size / 1024, 1024 / 8, 1024 * sizeof(BLS12_381_Fr_G1), stream>>>
            (d_out, d_inp, lg_domain_size);

    CUDA_OK(cudaGetLastError());
}

void NTT::LDE_powers(BLS12_381_Fr_G1* inout, bool innt, bool bitrev,
                        uint32_t lg_domain_size, uint32_t lg_blowup,
                        stream_t& stream, bool ext_pow)
{
    size_t domain_size = (size_t)1 << lg_domain_size;
    const auto gen_powers =
        NTTParameters::all(innt)[stream]->partial_group_gen_powers;

    if (domain_size < WARP_SZ)
        LDE_distribute_powers<<<1, domain_size, 0, stream>>>
                                (inout, lg_blowup, bitrev, gen_powers, ext_pow);
    else if (domain_size < 512)
        LDE_distribute_powers<<<domain_size / WARP_SZ, WARP_SZ, 0, stream>>>
                                (inout, lg_blowup, bitrev, gen_powers, ext_pow);
    else
        LDE_distribute_powers<<<domain_size / 512, 512, 0, stream>>>
                                (inout, lg_blowup, bitrev, gen_powers, ext_pow);

    CUDA_OK(cudaGetLastError());
}

void NTT::NTT_internal(BLS12_381_Fr_G1* d_inout, uint32_t lg_domain_size,
                            InputOutputOrder order, Direction direction,
                            Type type, stream_t& stream,
                            bool coset_ext_pow)
{
    // Pick an NTT algorithm based on the input order and the desired output
    // order of the data. In certain cases, bit reversal can be avoided which
    // results in a considerable performance gain.

    const bool intt = direction == Direction::inverse;
    const auto& ntt_parameters = *NTTParameters::all(intt)[stream];
    bool bitrev;
    Algorithm algorithm;

    switch (order) {
        case InputOutputOrder::NN:
            bit_rev(d_inout, d_inout, lg_domain_size, stream);
            bitrev = true;
            algorithm = Algorithm::CT;
            break;
        case InputOutputOrder::NR:
            bitrev = false;
            algorithm = Algorithm::GS;
            break;
        case InputOutputOrder::RN:
            bitrev = true;
            algorithm = Algorithm::CT;
            break;
        case InputOutputOrder::RR:
            bitrev = true;
            algorithm = Algorithm::GS;
            break;
        default:
            assert(false);
    }

    if (!intt && type == Type::coset)
        LDE_powers(d_inout, intt, bitrev, lg_domain_size, 0, stream,
                    coset_ext_pow);

    switch (algorithm) {
        case Algorithm::GS:
            GS_NTT(d_inout, lg_domain_size, intt, ntt_parameters, stream);
            break;
        case Algorithm::CT:
            CT_NTT(d_inout, lg_domain_size, intt, ntt_parameters, stream);
            break;
    }

    if (intt && type == Type::coset)
        LDE_powers(d_inout, intt, !bitrev, lg_domain_size, 0, stream,
                    coset_ext_pow);

    if (order == InputOutputOrder::RR)
        bit_rev(d_inout, d_inout, lg_domain_size, stream);
}

RustError NTT::Base(const gpu_t& gpu, BLS12_381_Fr_G1* inout, uint32_t lg_domain_size,
                        InputOutputOrder order, Direction direction,
                        Type type, bool coset_ext_pow)
{
    if (lg_domain_size == 0)
        return RustError{cudaSuccess};

    try {

        gpu.select();

        size_t domain_size = (size_t)1 << lg_domain_size;

        // cudaEvent_t start, stop;
        // cudaEventCreate(&start);
        // cudaEventCreate(&stop);
        // cudaEventRecord(start, 0);
        
        NTT_internal(inout, lg_domain_size, order, direction, type, gpu,
                    coset_ext_pow);

        gpu.sync();

        // cudaEventRecord(stop, 0);
        // cudaEventSynchronize(stop);

        // float elapsed;
        // cudaEventElapsedTime(&elapsed, start, stop);

        // std::cout << "NTT_internal: " << elapsed << " ms" << std::endl;


    } catch (const cuda_error& e) {
        gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }

    return RustError{cudaSuccess};
}

RustError compute_ntt(size_t device_id, BLS12_381_Fr_G1* inout, uint32_t lg_domain_size,
                      NTT::InputOutputOrder ntt_order,
                      NTT::Direction ntt_direction,
                      NTT::Type ntt_type)
{
    auto& gpu = select_gpu(device_id);

    return NTT::Base(gpu, inout, lg_domain_size,
                     ntt_order, ntt_direction, ntt_type);
}


}//namespace native
}//namespace at