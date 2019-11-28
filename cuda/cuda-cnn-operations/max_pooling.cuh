#include <helper_math.h>


__global__ void max_pooling_kernel(
        float *input,
        float *output,

        int width_input,
        int height_input,
        int deep_input,

        int width_output,
        int height_output,
        int deep_output,


        int stride_x,
        int stride_y,
        int padding_x,
        int padding_y
){
    int data_x = blockIdx.x * stride_x - padding_x + threadIdx.x;
    int data_y = blockIdx.y * stride_y - padding_y + threadIdx.y;
    int data_z = blockIdx.z;

    __shared__ float pool_values[10][10];
    __shared__ float max_value;
    max_value = -100.0f;

    pool_values[threadIdx.x][threadIdx.y] = input[data_z * height_input * width_input + data_y * width_input + data_x];

    for(int i=0; i<blockDim.x; i++){
        for(int j=0; j<blockDim.y; j++){

            if(pool_values[i][j] > max_value){
                max_value = pool_values[i][j];
            }
        }
    }

    output[blockIdx.z * width_output * height_output + blockIdx.y * width_output + blockIdx.x] = max_value;
}