
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

// DONE: Add Variables
float* h_D;
float* h_E;
float* d_D;
float* d_E;

// Functions
void Cleanup(void);
void RandomInit(float*, int);

// Device code
__global__ void VecAdd(const float* A, const float* B, float* C)
{
    int i = threadIdx.x;    
    C[i] = A[i] + B[i];	
}

// DONE: Kernel 2
__global__ void VecAdd2(const float* A, const float* B, float* C, const float* D, float* E)
{
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
	E[i] = A[i] + B[i] * D[i];
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

	// DONE: Allocate host memory for h_D and h_E
	h_D = (float*)malloc(size);
	if (h_D == 0) Cleanup();
	h_E = (float*)malloc(size);
	if (h_E == 0) Cleanup();
	
    // Initialize input vectors
    RandomInit(h_A, N);
    RandomInit(h_B, N);	
	
    // Allocate vectors in device memory
    CUDA_SAFE_CALL( cudaMalloc((void**)&d_A, size) );
    CUDA_SAFE_CALL( cudaMalloc((void**)&d_B, size) );
    CUDA_SAFE_CALL( cudaMalloc((void**)&d_C, size) );	

	// DONE: Allocate device memory for input array 'd' and output array 'e'
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_D, size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&d_E, size));

    // Copy vectors from host memory to device memory
    CUDA_SAFE_CALL( cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice) );

	// DONE: Fill d[] with random data and copy it to the device
	RandomInit(h_D, N);
	CUDA_SAFE_CALL(cudaMemcpy(d_D, h_D, size, cudaMemcpyHostToDevice));
	
	// Invoke kernel
	//VecAdd<<<1, N>>>(d_A, d_B, d_C);
	VecAdd2<<<1, N>>>(d_A, d_B, d_C, d_D, d_E);
	
#ifdef _DEBUG
	CUDA_SAFE_CALL( cudaThreadSynchronize() );
#endif


	// Copy result from device memory to host memory
    // h_C contains the result in host memory
    CUDA_SAFE_CALL( cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost) );

	// DONE: Copy result E from device to the host
	CUDA_SAFE_CALL(cudaMemcpy(h_E, d_E, size, cudaMemcpyDeviceToHost));

    // Verify result
	// DONE: Print out E and verify the result.
    int i = 0;
    for (i = 0; i < N; ++i) 
	{
        float sum = h_A[i] + h_B[i];
		float sum2 = h_A[i] + h_B[i] * h_D[i];
		printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
		printf("%f + %f * %f = %f\n", h_A[i], h_B[i], h_D[i], h_E[i]);
        if (fabs(h_C[i] - sum) > 1e-5)
            break;
		if (fabs(h_E[i] - sum2) > 1e-5)
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

	// DONE: Free device memory of D and E.
	if (d_D)
		cudaFree(d_D);
	if (d_E)
		cudaFree(d_E);

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);

	// DONE: Free host memory of D and E.	
	if (h_D)
		free(h_D);
	if (h_E)
		free(h_E);
        
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
