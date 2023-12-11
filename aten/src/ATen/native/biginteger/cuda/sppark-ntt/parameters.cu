#include "parameters.cuh"

// Maximum domain size supported. Can be adjusted at will, but with the
// target field in mind. Most fields handle up to 2^32 elements, BLS12-377
// can handle up to 2^47, alt_bn128 - 2^28...
namespace at { 
namespace native {

void NTTParameters(bool inverse, int id, BLS12_381_Fr_G1* data_ptr)
{
    stream_t& gpu = select_gpu(id);

    const size_t blob_sz = 64 + 128 + 256 + 512 + 32;
    
    const size_t partial_sz = WINDOW_NUM * WINDOW_SIZE;

    const size_t inverse_sz = S + 1;
    

    BLS12_381_Fr_G1* Inverse_roots_of_unity = (BLS12_381_Fr_G1*)gpu.Dmalloc(inverse_sz * sizeof(BLS12_381_Fr_G1));
    
    cudaMemcpyAsync(Inverse_roots_of_unity, inverse_roots_of_unity, 
                    inverse_sz * sizeof(BLS12_381_Fr_G1), cudaMemcpyHostToDevice,
                    gpu);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    BLS12_381_Fr_G1* Forward_roots_of_unity = (BLS12_381_Fr_G1*)gpu.Dmalloc(inverse_sz * sizeof(BLS12_381_Fr_G1));
    
    cudaMemcpyAsync(Forward_roots_of_unity, forward_roots_of_unity, 
                    inverse_sz * sizeof(BLS12_381_Fr_G1), cudaMemcpyHostToDevice,
                    gpu);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const BLS12_381_Fr_G1* roots = inverse ? (const BLS12_381_Fr_G1*)(Inverse_roots_of_unity)
                                : (const BLS12_381_Fr_G1*)Forward_roots_of_unity;
    

    generate_partial_twiddles<<<WINDOW_SIZE/32, 32, 0, gpu>>>
        (data_ptr, roots + MAX_LG_DOMAIN_SIZE);
    C10_CUDA_KERNEL_LAUNCH_CHECK();


    BLS12_381_Fr_G1* Group_gen_inverse = (BLS12_381_Fr_G1*)gpu.Dmalloc(sizeof(BLS12_381_Fr_G1));
    
    cudaMemcpyAsync(Group_gen_inverse, group_gen_inverse, 
                        sizeof(BLS12_381_Fr_G1), cudaMemcpyHostToDevice,
                        gpu);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    BLS12_381_Fr_G1* Group_gen = (BLS12_381_Fr_G1*)gpu.Dmalloc(sizeof(BLS12_381_Fr_G1));
    
    cudaMemcpyAsync(Group_gen, group_gen, 
                        sizeof(BLS12_381_Fr_G1), cudaMemcpyHostToDevice,
                        gpu);
    C10_CUDA_KERNEL_LAUNCH_CHECK();


    generate_partial_twiddles<<<WINDOW_SIZE/32, 32, 0, gpu>>>
        (data_ptr + partial_sz, inverse ? Group_gen_inverse
                                            : Group_gen);
    C10_CUDA_KERNEL_LAUNCH_CHECK();


    generate_all_twiddles<<<blob_sz/32, 32, 0, gpu>>>(data_ptr + 2 * partial_sz,
                                                        roots+6,
                                                        roots+7,
                                                        roots+8,
                                                        roots+9,
                                                        roots+10);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    generate_radixX_twiddles_X<<<16, 64, 0, gpu>>>(data_ptr + 2 * partial_sz + blob_sz, 64, roots+12);
    generate_radixX_twiddles_X<<<16, 64, 0, gpu>>>(data_ptr + 2 * partial_sz + blob_sz + 64*64, 4096, roots+18);
    generate_radixX_twiddles_X<<<16, 128, 0, gpu>>>(data_ptr + 2 * partial_sz + blob_sz + 64*64 + 4096*64, 128, roots+14);
    generate_radixX_twiddles_X<<<16, 256, 0, gpu>>>(data_ptr + 2 * partial_sz + blob_sz + 64*64 + 4096*64 + 128*128, 256, roots+16);
    generate_radixX_twiddles_X<<<16, 512, 0, gpu>>>(data_ptr + 2 * partial_sz + blob_sz + 64*64 + 4096*64 + 128*128 + 256*256, 512, roots+18);
    
    cudaMemcpyAsync(data_ptr + 2 * partial_sz + blob_sz + 64*64 + 4096*64 + 128*128 + 256*256 + 512*512, domain_size_inverse, 
                            inverse_sz * sizeof(BLS12_381_Fr_G1), cudaMemcpyHostToDevice,
                            gpu);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}//namespace native
}//namespace at
