#include "info.h"
using namespace std;
void info(){
    cudaSetDevice(0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    int bytesOfGlobalMem = deviceProp.totalGlobalMem;
    int MBytesOfGlobalMem = bytesOfGlobalMem/(1024*1024);
    cout<<"Device name: "<<deviceProp.name<<endl;
    cout<<"Total amount of global memory: "<<deviceProp.totalGlobalMem<<" (MBytes: "<<MBytesOfGlobalMem<<")"<<endl;
    cout<<"Multiprocessors: "<<deviceProp.multiProcessorCount<<endl;
}
