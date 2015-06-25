
#include "common.h"
#include <stdlib.h>
#include <GL/freeglut.h>
#include <math.h>


#define DIM 512
#define blockSize 8
#define blurRadius 8
#define effectiveBlockSize (blockSize+2*blurRadius)

float sourceColors[DIM*DIM];
float *sourceDevPtr;
float *targetDevPtr;
float *targetBlurDevPtr;

float readBackPixels[DIM*DIM];

// variable to setup the memory type (kernal call)
short memory = 2;

// DONE: time addicted variable
int a = 0;

// DONE: declare new texture memory
texture<float> tex;

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:
		exit(0);
		break;
	case 'q':
		memory++;
		memory %= 3;
		break;

	}
	glutPostRedisplay();
}

// Kernels
// DONE: implement a transformation kernel (diagonal shift/translation)
__global__ void transform(float* sourceDevPtr, float* targetDevPtr, int a)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int index2 = index;

	// translate in diagonal direction (use pixel as vector)
	int2 pixelPos = { (threadIdx.x + a) % DIM, (blockIdx.x + a) % DIM };

	// borders
	if (pixelPos.x > DIM && pixelPos.x < 0)

		pixelPos.x = 0;

	if (pixelPos.y > DIM && pixelPos.y < 0)
		pixelPos.y = 0;

	// interesting fact: the performance dumps if you proof the negation:
	// if (!(term)) ...

	// convert into 1d
	index2 = pixelPos.x + pixelPos.y * blockDim.x;

	targetDevPtr[index] = sourceDevPtr[index2];

}

// DONE: implement a boxcar filter kernel
__global__ void boxcar(float* targetDevPtr, float* targetBlurDevPtr, int kernelsize)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int index2 = index;

	// use pixel as vector
	int2 pixelPos = { threadIdx.x, blockIdx.x };

	// blurred grey value
	float grey = 0.f;

	if (kernelsize < 2)
		targetBlurDevPtr[index] = targetDevPtr[index];
	else
	{
		// borders	
		for (int i = -(kernelsize + 1) / 2; i <(kernelsize + 1) / 2; i++)	// iterate through kernel columns
		{
			for (int j = -(kernelsize + 1) / 2; j <(kernelsize + 1) / 2; j++)	// iterate through kernel rows
			{
				if (pixelPos.x + i <= DIM && pixelPos.x - i >= 0
					&& pixelPos.y + j <= DIM && pixelPos.y - j >= 0)	// zero padding
				{
					// convert into 1d
					index2 = pixelPos.x + i + (pixelPos.y + j) * blockDim.x;

					// add partial grey value to the target value
					grey += (targetDevPtr[index2]);

				}
			}
		}
		grey /=  float(kernelsize*kernelsize);
		targetBlurDevPtr[index] = grey;
	}
}



// TODO: implement a boxcar filter kernel using texture memory
__global__ void boxcarTex(float* targetBlurDevPtr, int kernelsize)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int index2 = index;

	// use pixel as vector
	int2 pixelPos = { threadIdx.x, blockIdx.x };

	// blurred grey value
	float grey = 0.f;

	if (kernelsize < 2)
		targetBlurDevPtr[index] = tex1Dfetch(tex, index2);
	else
	{
		// borders	
		for (int i = -(kernelsize + 1) / 2; i <(kernelsize + 1) / 2; i++)	// iterate through kernel columns
		{
			for (int j = -(kernelsize + 1) / 2; j <(kernelsize + 1) / 2; j++)	// iterate through kernel rows
			{
				if (pixelPos.x + i <= DIM && pixelPos.x - i >= 0
					&& pixelPos.y + j <= DIM && pixelPos.y - j >= 0)	// zero padding
				{
					// convert into 1d
					index2 = pixelPos.x + i + (pixelPos.y + j) * blockDim.x;

					// add partial grey value to the target value
					grey += (tex1Dfetch(tex, index2));

				}
			}
		}
		grey /= float(kernelsize*kernelsize);
		targetBlurDevPtr[index] = grey;
	}
}

__global__ void boxcarShared(float* targetDevPtr, float* targetBlurDevPtr)
{
	int tidX = threadIdx.x + blockIdx.x * blockSize - blurRadius;
	int tidY = threadIdx.y + blockIdx.y * blockSize - blurRadius;
	tidX = max(min(tidX, DIM - 1), 0);
	tidY = max(min(tidY, DIM - 1), 0);
	int tid = tidY *DIM + tidX;


	int cacheIndexX = threadIdx.x;
	int cacheIndexY = threadIdx.y;

	__shared__ float cache[effectiveBlockSize][effectiveBlockSize];

	//fill
	cache[cacheIndexX][cacheIndexY] = targetDevPtr[tid];

	__syncthreads();

	float outputPixel = 0.0f;
	if (cacheIndexY - blurRadius> 0 && cacheIndexX< effectiveBlockSize - blurRadius && cacheIndexX - blurRadius>0 && cacheIndexY < effectiveBlockSize - blurRadius)
	{
		for (int i = -blurRadius; i <= blurRadius; ++i)
		{
			for (int j = -blurRadius; j <= blurRadius; ++j)
			{
				outputPixel += cache[cacheIndexX + i][cacheIndexY + j];
			}
		}
	}
	//outputPixel = cache[cacheIndexX][cacheIndexY];
	outputPixel = outputPixel / (blurRadius * blurRadius);
	targetBlurDevPtr[tid] = outputPixel;

}


/*
__global__ void boxcarShared2(float* targetDevPtr, float* targetBlurDevPtr)
{
	
	
	
	

	__shared__ float cache[effectiveBlockSize*effectiveBlockSize];

	int pixelX = threadIdx.x + blockIdx.x * blockSize - blurRadius;
	int pixelY = threadIdx.y + blockIdx.y * blockSize - blurRadius;
	

	pixelX = max(min(pixelX, DIM - 1), 0);
	pixelY = max(min(pixelY, DIM - 1), 0);

	int cacheIndex = threadIdx.x + threadIdx.y * effectiveBlockSize;
	int pixelPos = pixelX + +pixelY * DIM;

	cache[cacheIndex] = targetDevPtr[pixelPos];
	if (threadIdx.x == 0 && blockIdx.x == 0 && threadIdx.y == 0 && blockIdx.y == 0)
		printf(threadIdx.x + "");
		
	__syncthreads();

	float outputPixel = 0.0f;
	if (threadIdx.x >= blurRadius && 
		threadIdx.x <= effectiveBlockSize - blurRadius &&
		threadIdx.y >= blurRadius && 
		threadIdx.y <= effectiveBlockSize - blurRadius)
	{
		for (int i = -blurRadius; i <= blurRadius; ++i)
		{
			for (int j = -blurRadius; i <= blurRadius; ++j)
			{
				outputPixel += cache[threadIdx.x + i + (threadIdx.y + j) * effectiveBlockSize];
			}
		}
	}

	targetBlurDevPtr[pixelPos] = outputPixel / (blurRadius*blurRadius);

	
}

*/


__global__ void boxcarShared3(float* source, float* target)
{
	__shared__ float cache[effectiveBlockSize * effectiveBlockSize];

	int pixelX = threadIdx.x + blockIdx.x * blockSize - blurRadius;
	int pixelY = threadIdx.y + blockIdx.y * blockSize - blurRadius;

	int cacheIndexX = threadIdx.x;
	int cacheIndexY = threadIdx.y;

	int cacheIndex = threadIdx.x + threadIdx.y * effectiveBlockSize;

	//borders
	if (0 <= pixelX < DIM && 0 <= pixelY < DIM)
	{
		cache[cacheIndex] = source[pixelX + pixelY * DIM];
	}

	__syncthreads();

	float blurredValue = 0;
	int tempX;
	int tempY;

	//Borders
	if (blurRadius <= cacheIndexX &&
		blurRadius <= cacheIndexY  &&
		cacheIndexX <= effectiveBlockSize - blurRadius &&
		cacheIndexY <= effectiveBlockSize - blurRadius &&
		pixelX < DIM &&
		pixelY < DIM)
	{

		for (int i = -blurRadius; i<blurRadius; ++i)
		{
			for (int j = -blurRadius; j<blurRadius; ++j)
			{
				tempX = i + cacheIndexX;
				tempY = j + cacheIndexY;
				blurredValue += cache[tempX + tempY * effectiveBlockSize];
			}
		}

		blurredValue = blurredValue / (blurRadius*blurRadius);
		target[pixelX + pixelY * DIM] = blurredValue;

	}

}
void display(void)
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// DONE: Transformationskernel auf sourceDevPtr anwenden
	transform << < DIM, DIM >> >(sourceDevPtr, targetDevPtr, a);
	a++;

	// DONE: Zeitmessung starten (see cudaEventCreate, cudaEventRecord)
	cudaEvent_t start, stop;
	float time;
	CUDA_SAFE_CALL(cudaEventCreate(&start));
	CUDA_SAFE_CALL(cudaEventCreate(&stop));
	CUDA_SAFE_CALL(cudaEventRecord(start, 0));

	int kernelsize = 10;
	dim3 blockDim(effectiveBlockSize, effectiveBlockSize);
	dim3 gridDim(DIM / blockSize, DIM / blockSize);
	// DONE: Kernel mit Blur-Filter ausführen.
	switch (memory)
	{
	case 0:
		boxcar << < DIM, DIM >> >(targetDevPtr, targetBlurDevPtr, kernelsize);
		break;
	case 1:
		boxcarTex << < DIM, DIM >> >(targetBlurDevPtr, kernelsize);
		break;
	case 2:
		boxcarShared3 << <gridDim, blockDim >> >(targetDevPtr, targetBlurDevPtr);
		break;
	}

	// DONE: Zeitmessung stoppen und fps ausgeben (see cudaEventSynchronize, cudaEventElapsedTime, cudaEventDestroy)
	CUDA_SAFE_CALL(cudaEventRecord(stop, 0));
	CUDA_SAFE_CALL(cudaEventSynchronize(stop));
	CUDA_SAFE_CALL(cudaEventElapsedTime(&time, start, stop));
	CUDA_SAFE_CALL(cudaEventDestroy(start));
	CUDA_SAFE_CALL(cudaEventDestroy(stop));

	switch (memory)
	{
	case 0:
		printf("Elapsed time: %f ms using global memory\n", time);
		break;
	case 1:
		printf("Elapsed time: %f ms using texture memory\n", time);
		break;
	case 2:
		printf("Elapsed time: %f ms using shared memory\n", time);
		break;
	}


	// Ergebnis zur CPU zuruecklesen
	//CUDA_SAFE_CALL( cudaMemcpy( readBackPixels, targetDevPtr, DIM*DIM*4, cudaMemcpyDeviceToHost ) ); // task01	
	CUDA_SAFE_CALL(cudaMemcpy(readBackPixels, targetBlurDevPtr, DIM*DIM * 4, cudaMemcpyDeviceToHost));

	// Ergebnis zeichnen (ja, jetzt gehts direkt wieder zur GPU zurueck...) 
	glDrawPixels(DIM, DIM, GL_LUMINANCE, GL_FLOAT, readBackPixels);
	glutSwapBuffers();
}

// clean up memory allocated on the GPU
void cleanup() {
	CUDA_SAFE_CALL(cudaFree(sourceDevPtr));

	// TODO: Aufräumen zusätzlich angelegter Ressourcen.
	CUDA_SAFE_CALL(cudaUnbindTexture(tex));
	CUDA_SAFE_CALL(cudaFree(targetDevPtr));
	CUDA_SAFE_CALL(cudaFree(targetBlurDevPtr));
	CUDA_SAFE_CALL(cudaFree(readBackPixels));
}

int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(DIM, DIM);
	glutCreateWindow("Memory Types");
	glutKeyboardFunc(keyboard);
	glutIdleFunc(display);
	glutDisplayFunc(display);

	// mit Schachbrettmuster füllen
	for (int i = 0; i < DIM*DIM; i++) {

		int x = (i % DIM) / (DIM / 8);
		int y = (i / DIM) / (DIM / 8);

		if ((x + y) % 2)
			sourceColors[i] = 1.0f;
		else
			sourceColors[i] = 0.0f;
	}

	// alloc memory on the GPU
	CUDA_SAFE_CALL(cudaMalloc((void**)&sourceDevPtr, DIM*DIM * 4));
	CUDA_SAFE_CALL(cudaMemcpy(sourceDevPtr, sourceColors, DIM*DIM * 4, cudaMemcpyHostToDevice));

	// DONE: Weiteren Speicher auf der GPU für das Bild nach der Transformation und nach dem Blur allokieren.
	// cudaMalloc( (void**)&devPtr, imageSize );
	CUDA_SAFE_CALL(cudaMalloc((void**)&targetDevPtr, DIM*DIM * 4));
	CUDA_SAFE_CALL(cudaMalloc((void**)&readBackPixels, DIM*DIM * 4));
	CUDA_SAFE_CALL(cudaMalloc((void**)&targetBlurDevPtr, DIM*DIM * 4));

	// DONE: Binding des Speichers des Bildes an eine Textur mittels cudaBindTexture.
	//cudaBindTexture( NULL, texName, devPtr, imageSize ); // use direct size, not sizeof()!!!
	CUDA_SAFE_CALL(cudaBindTexture(NULL, tex, targetDevPtr, DIM*DIM * 4));



	glutMainLoop();

	cleanup();
}
