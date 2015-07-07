
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

__global__ void reduce_max_not_naive(TYPE* A)
{
	//index, extern to specify size at runtime
	extern __shared__ TYPE cache[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockDim.x * 2) + threadIdx.x;
	cache[tid] =  cumax(A[i], A[i + blockDim.x]);
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
	{	
		if (tid < s)
		{
			cache[tid] = cumax(cache[tid], cache[tid + s]);
		}
		__syncthreads();
		
	}

	/*if (tid < 32)
	{
		cache[tid] = cumax(cache[tid], cache[tid + 32]);
		cache[tid] = cumax(cache[tid], cache[tid + 16]);
		cache[tid] = cumax(cache[tid], cache[tid + 8]);
		cache[tid] = cumax(cache[tid], cache[tid + 4]);
		cache[tid] = cumax(cache[tid], cache[tid + 2]);
		cache[tid] = cumax(cache[tid], cache[tid + 1]);
	}*/

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

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);

	int maxThreadsPerBlock = prop.maxThreadsPerBlock;
	int threads = maxThreadsPerBlock /4;
	// Start tracking of elapsed time.
	cudaEvent_t     start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

#ifdef USE_NAIVE	// Naive approach

	for (int n=1; n<N; n*=2)
		reduce_max_naive<<<Nh/n,1>>>(d_A, n);	

#else				// Better approach
	for (int i = 1; i < N; i *=threads)
	{
		reduce_max_not_naive << < N / threads/2, threads, sizeof(TYPE) * threads>> >(d_A);

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
