#include "def.h"
#include "info.cpp"
#include "check.cpp"
#include "matrixMul_kernel.cu"
using namespace std;


int mul(dim3 &dimA, dim3 &dimB, dim3 &block){
    unsigned int sizeDataA, sizeDataB, sizeDataC;
    sizeDataA = dimA.x*dimA.y*sizeof(float);
    sizeDataB = dimB.x*dimB.y*sizeof(float);
    sizeDataC = dimA.y*dimB.x*sizeof(float);

    float *hA = (float*)malloc(sizeDataA);
    float *hB = (float*)malloc(sizeDataB);
    float *hC = (float*)malloc(sizeDataC);

    float *dA, *dB, *dC;

    initialize(hA, dimA.x*dimA.y, 1.0f);
    initialize(hB, dimB.x*dimB.y, 1.1f);


    cudaMalloc((void**)&dA, sizeDataA);
    cudaMalloc((void**)&dB, sizeDataB);
    cudaMalloc((void**)&dC, sizeDataC);

    cudaMemcpy(dA, hA, sizeDataA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeDataB, cudaMemcpyHostToDevice);


    
    dim3 grid(dimB.x/block.x, dimA.y/block.y, 1); //need to cut grid by the blocks

    info();
    cudaEvent_t start;
    cudaEventCreate(&start);
    cudaEvent_t stop;
    cudaEventCreate(&stop);
    cudaEventRecord(start, NULL);

    mul<<<grid, block>>>(dC, dA, dB, dimA.x, dimB.x);   //initialize kernel 

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);

    cudaMemcpy(hC, dC, sizeDataC, cudaMemcpyDeviceToHost);

    double Gflops = ((2.0* (double)dimA.x * (double)dimA.y * (double)dimB.x) * 1.0e-9f) / (msecTotal / 1000.0f);
    cout<<"     SUCCES: Total msec: "<<msecTotal<<endl;
    cout<<"             Total Gflops: "<<Gflops<<endl;

    if(check(hC, hA, hB, dimA, dimB)){
        free(hA);
        free(hB);
        free(hC);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        return 0;
    }else{
        free(hA);
        free(hB);
        free(hC);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        return 1;
    }
}
int main()
{     
    int A_x, A_y, B_x, B_y;
    cout<<"Enter the size of marices"<<endl;
    cout<<"Size of matrix A(x,y): "<<endl;
    cin>>A_x>>A_y;
    cout<<"Size of matrix B(x,y): "<<endl;
    cin>>B_x>>B_y;
    if(A_x!=B_y){
        cerr<<"ERROR: A.x doesn't equals B.y"<<endl;
        return 1;
    }

    dim3 dimA(A_x, A_y);
    dim3 dimB(B_x, B_y);
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    
    if(mul(dimA, dimB, block)==0){
        return 0;
    }else {
        cerr<<"     CHECK ERROR: mul faild"<<endl;
        return 1;
    }
}
