__global__ void softmax_kernel(
        float *input,
        float *output,
        int size
        ){
    __shared__ float cash_summ;
    cash_summ=0.0f;

#pragma unroll
    for(int i=0; i<size; i++){
        cash_summ += input[i] * input[i];

    }
    int id = threadIdx.x;

    output[id] = (input[id] * input[id])/cash_summ;

}