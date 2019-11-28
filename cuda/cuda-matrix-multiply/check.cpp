#include <iostream>
#include "def.h"
using namespace std;

bool check(float *C, float *A, float *B, dim3 dimA, dim3 dimB){
    for(int i=0; i<dimA.y*dimB.x; i++){
        int k = i%dimB.x;
        int j = i/dimB.x;
        
        float val = 0.0f;
        for(int l = 0; l<dimA.x; l++){
            val +=A[j*dimA.x+l]*B[k+dimB.x*l];
        }

        if(val!=C[k+j*dimB.x]){
            return false;
        }
    }

    return true;

}

void initialize(float *data, int size, float val){
    for (int i = 0; i < size; ++i)
    {
        data[i] = val;
    }
}