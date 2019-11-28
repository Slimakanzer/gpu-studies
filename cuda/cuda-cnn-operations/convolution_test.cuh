
__global__ void Convolution_kernel(
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
//// right
// right
    int data_x = blockIdx.x * stride_x - padding_x + threadIdx.x;
    int data_y = blockIdx.y * stride_y - padding_y + threadIdx.y;
    int data_z = threadIdx.z;

    float out_val;
    __shared__ float value[3][3][512];

    if(
            (
                    (data_x + (int)threadIdx.x)<0)
            || ((data_y + (int)threadIdx.y)<0)
            || ((data_x + (int) threadIdx.x)>width_input)
            || ((data_y + (int) threadIdx.y) > height_input)

            ){
        value[threadIdx.x][threadIdx.y][threadIdx.z] = 0.0f;
    } else {
        value[threadIdx.x][threadIdx.y][threadIdx.z] = input[data_z * height_input * width_input + data_y * width_input + data_x]
                                                       * kernel[
                                                               blockIdx.z * height_kernel * width_kernel * deep_kernel
                                                               + threadIdx.z * height_kernel * width_kernel
                                                               + threadIdx.y * width_kernel
                                                               + threadIdx.x
                                                       ];
    }
    __syncthreads();
    out_val=0.0f;
    for(int i=0; i< deep_kernel; i++){
        for(int j=0; j< height_kernel; j++){
            for(int k=0; k<width_kernel; k++){
                out_val +=value[k][j][i];
            }
        }
    }




    output[blockIdx.z * width_output * height_output + blockIdx.y * width_output + blockIdx.x] = out_val;
}