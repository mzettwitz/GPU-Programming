
#include <stdio.h>
#include <math.h>
#include "common.h"
#include "bmp.h"
#include <stdlib.h>
#include <GL/freeglut.h>

#define DIM 512
#define blockSize 8

#define PI 3.1415926535897932f
#define centerX (DIM/2)
#define centerY (DIM/2)

float sourceColors[DIM*DIM];	// host memory for source image
float readBackPixels[DIM*DIM];	// host memory for swirled image

float *sourceDevPtr;			// device memory for source image
float *swirlDevPtr;				// device memory for swirled image

// DONE: Add  host variables to control the swirl
float a = 0.f;
float b = 0.f;

__global__ void swirlKernel( float *sourcePtr, float *targetPtr, float a, float b) 
{
	int index = 0;
    // DONE: Index berechnen	
	index = threadIdx.x + blockIdx.x * blockDim.x;

	// TODO: Den Swirl invertieren.
	// Add variables
	float r = 0.f;
	float angle = 0.f;

	// Create a vector: x = centerX - thread(x dim), y = centerY - block(y dim)
	float x = centerX - threadIdx.x;
	float y = centerY - blockIdx.x;

	// Compute radius of actual pixel via length of an vector using CUDA Math
	// http://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE
	r = hypotf(x, y);	

	angle = a * powf(r, b);

	// Compute rotated vector
	float x2 = x * cosf(angle) - y * sinf(angle);
	float y2 = x * sinf(angle) + y * cosf(angle);

	// Transform rotated vector into 1d
	int index2 = int(x2) + int(y2) * blockDim.x;

	targetPtr[index] = sourcePtr[index2];    // funny shifts till it collapses
}

void display(void)	
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// DONE: Swirl Kernel aufrufen.
	swirlKernel <<<DIM, DIM >>>(sourceDevPtr, swirlDevPtr, a, b);

	// DONE: Ergebnis zu host memory zuruecklesen.
	CUDA_SAFE_CALL(cudaMemcpy(readBackPixels, swirlDevPtr, sizeof(readBackPixels), cudaMemcpyDeviceToHost));

	// Ergebnis zeichnen (ja, jetzt gehts direkt wieder zur GPU zurueck...) 
	glDrawPixels( DIM, DIM, GL_LUMINANCE, GL_FLOAT, readBackPixels );

	glutSwapBuffers();
}

// clean up memory allocated on the GPU
void cleanup() {
    CUDA_SAFE_CALL( cudaFree( sourceDevPtr ) ); 
    CUDA_SAFE_CALL( cudaFree( swirlDevPtr ) ); 
}

// GLUT callback function for keyboard input
void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 27:
		exit(0);
		break;
	case 'q': // decrese a
		a = a - 0.01f;
		printf("a: %.2f ,  b: %.2f \r", a, b);
		break;
	case 'w': // increse a
		a = a + 0.01f;
		printf("a: %.2f ,  b: %.2f \r", a, b);
		break;
	case 'a': // decrese b
		b = b - 0.01f;
		printf("a: %.2f ,  b: %.2f \r", a, b);
		break;
	case 's': // increse b
		b = b + 0.01f;
		printf("a: %.2f ,  b: %.2f \r", a, b);
		break;
	}
	glutPostRedisplay();
}

int main(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(DIM, DIM);
	glutCreateWindow("Simple OpenGL CUDA");
	glutKeyboardFunc(keyboard);
	glutIdleFunc(display);
	glutDisplayFunc(display);


	std::cout << "Keys:" << std::endl;
	std::cout << "  Modifiy a: -(q), +(w)" << std::endl;
	std::cout << "  Modifiy b: -(a), +(s)" << std::endl;


	// load bitmap	
	Bitmap bmp = Bitmap("who-is-that.bmp");
	if (bmp.isValid())
	{		
		for (int i = 0 ; i < DIM*DIM ; i++) {
			sourceColors[i] = bmp.getR(i/DIM, i%DIM) / 255.0f;
		}
	}

	// DONE: allocate memory at sourceDevPtr on the GPU and copy sourceColors into it.
	CUDA_SAFE_CALL(cudaMalloc((void**)&sourceDevPtr, sizeof(sourceColors)));
	CUDA_SAFE_CALL(cudaMemcpy(sourceDevPtr, sourceColors, sizeof(sourceColors), cudaMemcpyHostToDevice));
	
	// DONE: allocate memory at swirlDevPtr for the unswirled image.	
	CUDA_SAFE_CALL(cudaMalloc((void**)&swirlDevPtr, sizeof(readBackPixels)));

	glutMainLoop();

	cleanup();
}