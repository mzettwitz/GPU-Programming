
#include "common.h"
#include <stdlib.h>
#include <GL/freeglut.h>
#include <math.h>


#define DIM 512
#define blockSize 8
#define blurRadius 6
#define effectiveBlockSize (blockSize+2*blurRadius)

float sourceColors[DIM*DIM];
float *sourceDevPtr;
float *targetDevPtr;
float *targetBlurDevPtr;

float readBackPixels[DIM*DIM];

// DONE: time addicted variable
int a = 0;

// DONE: declare new texture memory
//texture<float> tex;

void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:
		exit(0);
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

	// boarders
	if (pixelPos.x > DIM && pixelPos.x < 0)
		pixelPos.x = 0 ;
	
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

	// blured grey value
	float grey = 0.f;

	// borders
	for (int i = -kernelsize / 2; i < kernelsize / 2; i++)	// iterate through kernel columns
	{
		for (int j = -kernelsize / 2; i < kernelsize / 2; j++)	// iterate through kernel rows
		{
			if (pixelPos.x + i < DIM && pixelPos.x - i > 0
				&& pixelPos.y + j < DIM && pixelPos.y + j > 0)	// zero padding
			{
				// convert into 1d
				index2 = pixelPos.x + i + pixelPos.y + j * blockDim.x;

				// add partial grey value to the target value
				grey += targetDevPtr[index2] / float(kernelsize);
			}
		}
	}

	targetBlurDevPtr[index] = grey;

}

void display(void)	
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// DONE: Transformationskernel auf sourceDevPtr anwenden
	transform <<< DIM, DIM >>>(sourceDevPtr, targetDevPtr, a);
	a++;


	// TODO: Zeitmessung starten (see cudaEventCreate, cudaEventRecord)

	// TODO: Kernel mit Blur-Filter ausführen.
	//boxcar << < DIM, DIM >> >(targetDevPtr, targetBlurDevPtr, blurRadius);

	// TODO: Zeitmessung stoppen und fps ausgeben (see cudaEventSynchronize, cudaEventElapsedTime, cudaEventDestroy)

	// Ergebnis zur CPU zuruecklesen
    CUDA_SAFE_CALL( cudaMemcpy( readBackPixels, targetDevPtr, DIM*DIM*4, cudaMemcpyDeviceToHost ) );
	
	//CUDA_SAFE_CALL(cudaMemcpy(readBackPixels, targetBlurDevPtr, DIM*DIM * 4, cudaMemcpyDeviceToHost));

	// Ergebnis zeichnen (ja, jetzt gehts direkt wieder zur GPU zurueck...) 
	glDrawPixels( DIM, DIM, GL_LUMINANCE, GL_FLOAT, readBackPixels );
	glutSwapBuffers();
}

// clean up memory allocated on the GPU
void cleanup() {
    CUDA_SAFE_CALL( cudaFree( sourceDevPtr ) );     

	// TODO: Aufräumen zusätzlich angelegter Ressourcen.
	//CUDA_SAFE_CALL(cudaUnbindTexture(tex));
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
	for (int i = 0 ; i < DIM*DIM ; i++) {

		int x = (i % DIM) / (DIM/8);
		int y = (i / DIM) / (DIM/8);

		if ((x + y) % 2)
			sourceColors[i] = 1.0f;
		else
			sourceColors[i] = 0.0f;
	}

	// alloc memory on the GPU
	CUDA_SAFE_CALL( cudaMalloc( (void**)&sourceDevPtr, DIM*DIM*4 ) );
    CUDA_SAFE_CALL( cudaMemcpy( sourceDevPtr, sourceColors, DIM*DIM*4, cudaMemcpyHostToDevice ) );

	// DONE: Weiteren Speicher auf der GPU für das Bild nach der Transformation und nach dem Blur allokieren.
	// cudaMalloc( (void**)&devPtr, imageSize );
	CUDA_SAFE_CALL(cudaMalloc((void**)&targetDevPtr, DIM*DIM*4));
	CUDA_SAFE_CALL(cudaMalloc((void**)&readBackPixels, DIM*DIM*4));
	CUDA_SAFE_CALL(cudaMalloc((void**)&targetBlurDevPtr, DIM*DIM*4));

	// DONE: Binding des Speichers des Bildes an eine Textur mittels cudaBindTexture.
	//cudaBindTexture( NULL, texName, devPtr, imageSize );
	//CUDA_SAFE_CALL(cudaBindTexture(NULL, tex, sourceDevPtr, sizeof(sourceDevPtr)));


	glutMainLoop();

	cleanup();
}
