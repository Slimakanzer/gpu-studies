#include <stdio.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32