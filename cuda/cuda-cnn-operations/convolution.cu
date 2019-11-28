#include <iostream>
#include <stdio.h>

#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda_runtime.h>


#include "convolution_common.h"

//#include "convolution.cuh"
#include "convolution_test.cuh"

using namespace std;
extern "C" float convolution(
        float *input,
        float *kernel,
        float *output,

        int width_input,
        int height_input,
        int deep_input,

        int width_kernel,
        int height_kernel,
        int deep_kernel,
        int long_kernel,

        int width_output,
        int height_output,
        int deep_output,


        int stride_x,
        int stride_y,
        int padding_x,
        int padding_y
)
{



    dim3 grid(width_output, height_output, deep_output);
    dim3 thread(width_kernel,height_kernel,deep_kernel);


    cudaError_t error;
    cudaEvent_t start;
    error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Record the start event
    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    Convolution_kernel<<< grid, thread >>>(
            input,
            kernel,
            output,

            width_input,
            height_input,
            deep_input,

            width_kernel,
            height_kernel,
            deep_kernel,
            long_kernel,

            width_output,
            height_output,
            deep_output,

            stride_x,
            stride_y,
            padding_x,
            padding_y
    );

    // Record the stop event
    error = cudaEventRecord(stop, NULL);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    // Wait for the stop event to complete
    error = cudaEventSynchronize(stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    float msecTotal = 0.0f;
    error = cudaEventElapsedTime(&msecTotal, start, stop);

    if (error != cudaSuccess)
    {
        fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    return msecTotal;
}