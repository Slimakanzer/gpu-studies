// System includes
#include <iostream>

// Include CUDA runtime
#include <cuda_runtime.h>

// Helper functions for CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#include "ops.h"
#include "inc/structs.h"
#define EXIT_FAILUR 1


// Need to initialize synyethic data
void initialize(float *data,int size, float number){
    for(int i=0; i<size; i++){
        data[i]=number;
    }
}

int convolution_prepare(
        bool use_synthetic,
        std::string path_to_dataset,
        std::string path_to_labels,
        dim3 &dimsLayer,
        dim3 &dimsFilter,

        int stride_x,
        int stride_y,
        int padding_x,
        int padding_y
) {
    std::cout<<"Execution convolution function"<<std::endl;
    int
            H_out,
            W_out;

    tensor_3D
            input_layer,
            output_layer;

    tensor_4D
            kernel;

    if(use_synthetic){
        input_layer.name="synthetic_input_layer";
        input_layer.W=dimsLayer.x;
        input_layer.H=dimsLayer.y;
        input_layer.D=dimsLayer.z;
        input_layer.value.resize(input_layer.W * input_layer.H * input_layer.D);

        initialize(&input_layer.value[0],input_layer.value.size(), 1.0f);

        kernel.name="synthetic_kernel_layer";
        kernel.H=dimsFilter.x;
        kernel.W=dimsFilter.y;
        kernel.D=dimsLayer.z;
        kernel.L=dimsFilter.z;
        kernel.value.resize(kernel.W * kernel.H * kernel.D * kernel.L);

        initialize(&kernel.value[0],kernel.value.size(), 2.1f);


        output_layer.D=kernel.L;

    } else{
        exit(0);
    }

    // Outer layer size
    W_out = ((input_layer.W - kernel.W + 2*padding_x) / stride_x) + 1;
    H_out = ((input_layer.H - kernel.H + 2*padding_y) / stride_y) + 1;

    output_layer.H = H_out;
    output_layer.W = W_out;
    output_layer.value.resize(output_layer.W * output_layer.H * output_layer.D);


    float
            *input_value,
            *kernel_value,
            *output_value;

    // Allocate memory GPU
    {
        // Allocate memory input layer
        checkCudaErrors(
                cudaMalloc(
                        (void **) &input_value, input_layer.value.size() * sizeof(float)
                )
        );

        checkCudaErrors(
                cudaMalloc(
                        (void **) &kernel_value, kernel.value.size() * sizeof(float)
                )
        );

        checkCudaErrors(
                cudaMalloc(
                        (void **) &output_value, output_layer.value.size() * sizeof(float)
                )
        );


    }

    // Copy host to device
    {
        std::cout << "Copy host memory to device" << std::endl;
        // Copy input layer
        checkCudaErrors(
                cudaMemcpyAsync(
                        input_value, &input_layer.value[0], input_layer.value.size() * sizeof(float), cudaMemcpyHostToDevice
                )
        );

        checkCudaErrors(
                cudaMemcpyAsync(
                        kernel_value, &kernel.value[0], kernel.value.size() * sizeof(float), cudaMemcpyHostToDevice
                )
        );

    }


    std::cout<<"Executing Kernel"<<std::endl;

    // Execute kernel
    float msecTotal = convolution(
            input_value,
            kernel_value,
            output_value,

            input_layer.W,
            input_layer.H,
            input_layer.D,

            kernel.W,
            kernel.H,
            kernel.D,
            kernel.L,

            output_layer.W,
            output_layer.H,
            output_layer.D,

            stride_x,
            stride_y,
            padding_x,
            padding_y
    );




    // Copy result device to host
    {
        std::cout << "Copy result from device to host" << std::endl;
        checkCudaErrors(
                cudaMemcpy(
                        &output_layer.value[0], output_value, output_layer.value.size() * sizeof(float), cudaMemcpyDeviceToHost
                )
        );
    }

    std::cout<<output_layer.value.size()<<"   "<<output_layer.value[1];


    std::cout<<"Total msec operation on GPU: "<<msecTotal<<std::endl;


    // Clean memory GPU
    {
        std::cout << "Cleaning GPU and host memory..." << std::endl;
        checkCudaErrors(cudaFree(input_value));
        checkCudaErrors(cudaFree(kernel_value));
        checkCudaErrors(cudaFree(output_value));
    }

    // Clean memory CPU
    return 0;
}

int max_pooling_prepare(
        bool use_synthetic,
        std::string path_to_dataset,
        std::string path_to_labels,
        dim3 &dimsLayer,
        dim3 &dimsFilter,

        int stride_x,
        int stride_y,
        int padding_x,
        int padding_y
){
    std::cout<<"Execution max-pooling function"<<std::endl;
    int
            H_kernel,
            W_kernel,
            H_out,
            W_out;

    tensor_3D
            input_layer,
            output_layer;

    H_kernel = dimsFilter.y;
    W_kernel = dimsFilter.x;

    if(use_synthetic){

        input_layer.name="synthetic_input_layer";
        input_layer.W=dimsLayer.x;
        input_layer.H=dimsLayer.y;
        input_layer.D=dimsLayer.z;
        input_layer.value.resize(input_layer.W * input_layer.H * input_layer.D);
        initialize(&input_layer.value[0],input_layer.value.size(), 1.0f);

    } else{
        exit(0);
    }

    // Testing values of array
//    for(int i=0; i<input_layer.value.size(); i++){
//        std::cout<<input_layer.value[i];
//    }

    // Outer layer size
    W_out = ((input_layer.W - W_kernel + 2*padding_x) / stride_x) + 1;
    H_out = ((input_layer.H - H_kernel + 2*padding_y) / stride_y) + 1;

    output_layer.H = H_out;
    output_layer.W = W_out;
    output_layer.D = input_layer.D;

    output_layer.value.resize(output_layer.W * output_layer.H * output_layer.D);

    // Test size of output values
//    std::cout<<output_layer.value.size();

    float
            *input_value,
            *output_value;

    // Allocate memory GPU
    {
        // Allocate memory input layer
        checkCudaErrors(
                cudaMalloc(
                        (void **) &input_value, input_layer.value.size() * sizeof(float)
                )
        );

        checkCudaErrors(
                cudaMalloc(
                        (void **) &output_value, output_layer.value.size() * sizeof(float)
                )
        );


    }


    // Copy host to device
    {
        std::cout << "Copy host memory to device" << std::endl;
        // Copy input layer
        checkCudaErrors(
                cudaMemcpyAsync(
                        input_value, &input_layer.value[0], input_layer.value.size() * sizeof(float), cudaMemcpyHostToDevice
                )
        );
    }

    // Execute kernel
    std::cout<<"Executing Kernel"<<std::endl;
    float msecTotal = max_pooling(
            input_value,
            output_value,

            input_layer.W,
            input_layer.H,
            input_layer.D,

            W_kernel,
            H_kernel,

            output_layer.W,
            output_layer.H,
            output_layer.D,

            stride_x,
            stride_y,
            padding_x,
            padding_y
    );




    // Copy result device to host
    {
        std::cout << "Copy result from device to host" << std::endl;
        checkCudaErrors(
                cudaMemcpy(
                        &output_layer.value[0], output_value, output_layer.value.size() * sizeof(float), cudaMemcpyDeviceToHost
                )
        );
    }

    std::cout<<"Total msec operation on GPU: "<<msecTotal<<std::endl;

    // Clean memory GPU
    {
        std::cout << "Cleaning GPU and host memory..." << std::endl;
        checkCudaErrors(cudaFree(input_value));
        checkCudaErrors(cudaFree(output_value));
    }

    return 0;




}

int softmax_prepare(
        bool use_synthetic,
        std::string path_to_dataset,
        std::string path_to_labels,
        dim3 &dimsLayer
) {
    std::cout<<"Executing softmax function"<<std::endl;
    int
            size;

    size = dimsLayer.x;

    std::vector<float>
            input(size),
            output(size);

    if(use_synthetic){

        initialize(&input[0], size, 1.0f);
    } else{
        exit(0);
    }

    float
            *input_value,
            *output_value;

    {
        checkCudaErrors(
                cudaMalloc(
                        (void **) &input_value, input.size() * sizeof(float)
                )
        );

        checkCudaErrors(
                cudaMalloc(
                        (void **) &output_value, output.size() * sizeof(float)
                )
        );
    }


    // Copy host to device
    {
        std::cout << "Copy host memory to device" << std::endl;
        // Copy input layer
        checkCudaErrors(
                cudaMemcpyAsync(
                        input_value, &input[0], input.size() * sizeof(float), cudaMemcpyHostToDevice
                )
        );
    }

    // Execute kernel
    std::cout<<"Executing Kernel"<<std::endl;
    float msecTotal = softmax(
            input_value,
            output_value,
            size
            );

    // Copy result device to host
    {
        std::cout << "Copy result from device to host" << std::endl;
        checkCudaErrors(
                cudaMemcpy(
                        &output[0], output_value, output.size() * sizeof(float), cudaMemcpyDeviceToHost
                )
        );
    }

    std::cout<<"Total msec operation on GPU: "<<msecTotal<<std::endl;

    // Clean memory GPU
    {
        std::cout << "Cleaning GPU and host memory..." << std::endl;
        checkCudaErrors(cudaFree(input_value));
        checkCudaErrors(cudaFree(output_value));
    }

    return 0;
}
