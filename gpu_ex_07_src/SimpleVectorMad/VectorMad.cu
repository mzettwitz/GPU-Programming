
// Includes
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

// Variables
float* h_A;
float* h_B;
float* h_C;
float* d_A;
float* d_B;
float* d_C;

// Functions
void Cleanup(void);
void RandomInit(float*, int);

// Device code
__global__ void VecAdd(const float* A, const float* B, float* C)
{
    int i = threadIdx.x;    
    C[i] = A[i] + B[i];	
}

// Host code
int main(int argc, char** argv)
{
    printf("Simple vector addition\n");
    int N = 256;
    size_t size = N * sizeof(float);    

    // Allocate input vectors h_A, h_B and h_C in host memory
    h_A = (float*)malloc(size);
    if (h_A == 0) Cleanup();
    h_B = (float*)malloc(size);
    if (h_B == 0) Cleanup();
    h_C = (float*)malloc(size);
    if (h_C == 0) Cleanup();
	
    // Initialize input vectors
    RandomInit(h_A, N);
    RandomInit(h_B, N);	
	
    // Allocate vectors in device memory
    CUDA_SAFE_CALL( cudaMalloc((void**)&d_A, size) );
    CUDA_SAFE_CALL( cudaMalloc((void**)&d_B, size) );
    CUDA_SAFE_CALL( cudaMalloc((void**)&d_C, size) );		

    // Copy vectors from host memory to device memory
    CUDA_SAFE_CALL( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) );	

	// Invoke kernel
	VecAdd<<<1, N>>>(d_A, d_B, d_C);
	
#ifdef _DEBUG
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
#endif


	// Copy result from device memory to host memory
    // h_C contains the result in host memory
    CUDA_SAFE_CALL( cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost) );

    // Verify result
	// TODO: Print out E and verify the result.
    int i = 0;
    for (i = 0; i < N; ++i) 
	{
        float sum = h_A[i] + h_B[i];
		printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
        if (fabs(h_C[i] - sum) > 1e-5)
            break;
    }
    printf("%s \n", (i == N) ? "PASSED" : "FAILED");
    
    Cleanup();
}

void Cleanup(void)
{
    // Free device memory
    if (d_A)
        cudaFree(d_A);
    if (d_B)
        cudaFree(d_B);
    if (d_C)
        cudaFree(d_C);

	// TODO: Free device memory of D and E.	

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);

	// TODO: Free host memory of D and E.	
        
    CUDA_SAFE_CALL( cudaThreadExit() );
        
    printf("\nPress ENTER to exit...\n");
    fflush( stdout);
    fflush( stderr);
    getchar();    

    exit(0);
}

// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; ++i)
        data[i] = rand() / (float)RAND_MAX;
}
