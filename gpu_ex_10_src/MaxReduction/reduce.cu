
// Includes
#include <stdio.h>

// Type of the array in which we search for the maximum.
// If you use float, don't forget to type %f in the printf later on..
#define TYPE int

//#define USE_NAIVE

// Variables
TYPE* h_A;
TYPE* d_A;

// Functions
void Cleanup(void);
void WorstCaseInit(TYPE*, int);

__device__ __host__ TYPE cumax(TYPE a, TYPE b)
{
	return a > b ? a : b;
}

// Schema des naiven Ansatz
// o o o o o o o o  n=1
// |/  |/  |/  |/
// o   o   o   o    n=2
// |  /    |  /
// | /     | /
// o       o        n=4
// |      /
// |    /
// |  /
// |/
// o				Ergebnis

__global__ void reduce_max_naive(TYPE* A, int n)
{
	int i = blockIdx.x * n;
	A[2*i] = cumax( A[2*i], A[2*i+n]);
}


template <unsigned int blockSize,unsigned int loadSize>
__global__ void reduce_max_not_naive(TYPE* A)
{
	//index, extern to specify size at runtime
	extern __shared__ TYPE cache[];


	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * loadSize) + threadIdx.x;
	//during load from global, u
	cache[tid] = cumax(A[i], A[i + blockDim.x]);
	if (loadSize > 2){
		cache[tid] = cumax(cache[tid], A[i + blockDim.x * 2]);
		cache[tid] = cumax(cache[tid], A[i + blockDim.x * 3]);
	}
	if (loadSize > 4){
		cache[tid] = cumax(cache[tid], A[i + blockDim.x * 4]);
		cache[tid] = cumax(cache[tid], A[i + blockDim.x * 5]);
		cache[tid] = cumax(cache[tid], A[i + blockDim.x * 6]);
		cache[tid] = cumax(cache[tid], A[i + blockDim.x * 7]);
	}
	if (loadSize > 8)
	{
		cache[tid] = cumax(cache[tid], A[i + blockDim.x * 8]);
		cache[tid] = cumax(cache[tid], A[i + blockDim.x * 9]);
		cache[tid] = cumax(cache[tid], A[i + blockDim.x * 10]);
		cache[tid] = cumax(cache[tid], A[i + blockDim.x * 11]);
		cache[tid] = cumax(cache[tid], A[i + blockDim.x * 12]);
		cache[tid] = cumax(cache[tid], A[i + blockDim.x * 13]);
		cache[tid] = cumax(cache[tid], A[i + blockDim.x * 14]);
		cache[tid] = cumax(cache[tid], A[i + blockDim.x * 15]);
	}

	__syncthreads();

	//unrolled loop
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			cache[tid] = cumax(cache[tid], cache[tid + 256]);
		}
		__syncthreads();
	}
	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			cache[tid] = cumax(cache[tid], cache[tid + 128]);
		}
		__syncthreads();
	}
	if (blockSize >= 128)
	{
		if (tid < 64)
		{
			cache[tid] = cumax(cache[tid], cache[tid + 64]);
		}
		__syncthreads();
	}
	if (tid < 32)
	{
		if(blockSize >= 64)cache[tid] = cumax(cache[tid], cache[tid + 32]);
		__syncthreads();
		if(blockSize >= 32)cache[tid] = cumax(cache[tid], cache[tid + 16]);
		__syncthreads();
		if(blockSize >= 16)cache[tid] = cumax(cache[tid], cache[tid + 8]);
		__syncthreads();
		if(blockSize >= 8)cache[tid] = cumax(cache[tid], cache[tid + 4]);
		__syncthreads();
		if(blockSize >= 4)cache[tid] = cumax(cache[tid], cache[tid + 2]);
		__syncthreads();
		if(blockSize >= 2)cache[tid] = cumax(cache[tid], cache[tid + 1]);
	}

	if (tid == 0)
	{
		A[blockIdx.x] = cache[0];
	}
}

// Host code
int main(int argc, char** argv)
{
	printf("Reduce\n");
	int N = 1 << 15;
	int Nh = N / 2;
	size_t size = N * sizeof(TYPE);

	// Allocate input vector h_A
	h_A = (TYPE*)malloc(size);

	// Initialize input vector
	WorstCaseInit(h_A, N);

	// Allocate vector in device memory
	cudaMalloc((void**)&d_A, size);

	// Copy vector from host memory to device memory
	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

	int threads = 128; // 256
	int load = 8;
	int gridSize = N / threads / load;

	// Start tracking of elapsed time.
	cudaEvent_t     start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

#ifdef USE_NAIVE	// Naive approach

	for (int n=1; n<N; n*=2)
		reduce_max_naive<<<Nh/n,1>>>(d_A, n);	

#else				// Better approach
	switch (threads)
	{
	case 64:
		reduce_max_not_naive<64,8> << < gridSize, threads, sizeof(TYPE) * threads >> >(d_A);
		reduce_max_not_naive<64,8> << < gridSize, threads, sizeof(TYPE)*threads >> > (d_A);
		break;
	case 128:
		reduce_max_not_naive<128,8> << < gridSize, threads, sizeof(TYPE) * threads >> >(d_A);
		reduce_max_not_naive<128,8> << < gridSize, threads, sizeof(TYPE)*threads >> > (d_A);
		break;
	case 256:
		reduce_max_not_naive<256,8> << < gridSize, threads, sizeof(TYPE) * threads >> >(d_A);
		reduce_max_not_naive<256,8> << < gridSize, threads, sizeof(TYPE)*threads >> > (d_A);
		break;
	case 512:
		reduce_max_not_naive<512,8> << < gridSize, threads, sizeof(TYPE) * threads >> >(d_A);
		reduce_max_not_naive<512,8> << < gridSize, threads, sizeof(TYPE)*threads >> > (d_A);
		break;
	}
	
#endif

	// End tracking of elapsed time.
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	float   elapsedTime;
	cudaEventElapsedTime( &elapsedTime, start, stop );
	printf( "Time: %f ms\n", elapsedTime );
	cudaEventDestroy( start );
	cudaEventDestroy( stop );

	// Find the truth... :)
	TYPE maximum = 0;
	for (int i = 0; i < N; ++i) 	
		maximum = cumax(h_A[i], maximum);
	
	// Copy result (first element only) from device memory to host memory
    cudaMemcpy(h_A, d_A, sizeof(TYPE), cudaMemcpyDeviceToHost);

	// Validate result from GPU.
	if (maximum == h_A[0])
		printf("PASSED: %i == %i", maximum, h_A[0]);
	else printf("FAILED: %i != %i", maximum, h_A[0]);
    
    Cleanup();
}

void Cleanup(void)
{
    // Free device memory
    if (d_A)
        cudaFree(d_A);
  
    // Free host memory
    if (h_A)
        free(h_A);  
        
    cudaThreadExit();
        
    printf("\nPress ENTER to exit...\n");
    fflush( stdout);
    fflush( stderr);
    getchar();    

    exit(0);
}

void WorstCaseInit(TYPE* data, int n)
{
	// Using a list sorted in ascending order is the worst case.
    for (int i = 0; i < n; ++i)
		data[i] = (TYPE)(i);
}
