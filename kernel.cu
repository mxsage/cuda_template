#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// CUDA utilities
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>
#include <helper_math.h>

// Rendering elevation
__global__ void RENDER(float* image_output, int w, int h)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i = index % w;
    int j = index / h;
    if (i < w && j < h)
    {
        uint ii_out = 4 * (j * w + i);

        image_output[ii_out + 0] = 1;
        image_output[ii_out + 1] = 0;
        image_output[ii_out + 2] = 0;
        image_output[ii_out + 3] = 1;
    }
}

extern "C" void render(float* img, int width, int height)
{
    int imgBlockSize = 64;
    int imgNumBlocks = (width * height + imgBlockSize - 1) / imgBlockSize;

    RENDER <<<imgNumBlocks, imgBlockSize>>> (img, width, height);
    checkCudaErrors(cudaDeviceSynchronize());
}