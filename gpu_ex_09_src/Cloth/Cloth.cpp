#include "Cloth.h"
#include <GL/glew.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "common.h"
#include <iostream>
#include <string>
#include <fstream>


void updateCloth(float3* newPos, float3* oldPos, float3* impacts, float3* velocity,
	float deltaTime, float stepsize);

unsigned int memSize = sizeof(float)* 3 * RESOLUTION_X*RESOLUTION_Y;


ClothSim::ClothSim() : ping(0)
{
	vboPos[0] = 0;
	vboPos[1] = 0;


	// Initialize mesh
	float ratio = RESOLUTION_Y / (float)RESOLUTION_X;
	float* m_hPos = new float[3 * RESOLUTION_X*RESOLUTION_Y];
	int j = 0;
	for (int x = 0; x<RESOLUTION_X; ++x)
	{
		for (int y = 0; y<RESOLUTION_Y; ++y)
		{
			m_hPos[j * 3] = x / (float)RESOLUTION_X - 0.5f;
			m_hPos[j * 3 + 1] = 1;
			m_hPos[j * 3 + 2] = y / (float)RESOLUTION_Y * ratio - (ratio);
			++j;
		}
	}

	// allocate device memory for intermediate impacts and velocities.
	CUDA_SAFE_CALL(cudaMalloc((void**)&devPtrImpact, memSize));
	CUDA_SAFE_CALL(cudaMalloc((void**)&devPtrVelocity, memSize));
	cudaMemset(devPtrImpact, 0, RESOLUTION_X*RESOLUTION_Y*sizeof(float3));
	cudaMemset(devPtrVelocity, 0, RESOLUTION_X*RESOLUTION_Y*sizeof(float3));

	// DONE: Erzeugen der VBOs für die Positionen und Verbindung zu CUDA herstellen.
	// generate VBOs (PixelBufferObject)
	glGenBuffers(2, vboPos);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vboPos[0]);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, memSize, m_hPos, GL_DYNAMIC_DRAW_ARB);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	//glBufferData (target, size, data, usage)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vboPos[1]);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, memSize, m_hPos, GL_DYNAMIC_DRAW_ARB);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	//Connection to CUDA
	cudaGraphicsGLRegisterBuffer(&cudaPos[0], vboPos[0], cudaGraphicsMapFlagsNone);
	cudaGraphicsGLRegisterBuffer(&cudaPos[1], vboPos[1], cudaGraphicsMapFlagsNone);

	// DONE: VBO vboNormal erzeugen und mit cudaNormal verknüpfen. Das VBO braucht keine initialen Daten (NULL übergeben).
	glGenBuffers(1, &vboNormal);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, vboNormal);
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, memSize, NULL, GL_DYNAMIC_DRAW_ARB);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	//Connection to CUDA
	cudaGraphicsGLRegisterBuffer(&cudaNormal, vboNormal, cudaGraphicsMapFlagsNone);

	delete[] m_hPos;
}

ClothSim::~ClothSim()
{
	CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(cudaPos[0]));
	CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(cudaPos[1]));

	// DONE: cudaNormal freigeben
	CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(cudaNormal));
	glDeleteBuffers(2, (const GLuint*)vboPos);

	// DONE: vboNormal freigeben
	glDeleteBuffers(1, &vboNormal);

	CUDA_SAFE_CALL(cudaFree(devPtrImpact));
	CUDA_SAFE_CALL(cudaFree(devPtrVelocity));
}

void ClothSim::update(GLfloat deltaTime)
{
	// Lokale Variablen, in die die Pointer auf die Daten der CUDA-Ressourcen abgelegt werden können.
	float* oldPos = NULL;
	float* newPos = NULL;
	float* normals = NULL;

	// DONE: Map cudaPos (Hinweis: cudaGraphicsMapResources)
	cudaGraphicsMapResources(2, cudaPos);

	// DONE: Map cudaNormal
	cudaGraphicsMapResources(1, &cudaNormal);

	// DONE: Pointer auf die Daten von cudaPos[ping] und cudaPos[1-ping] beschaffen. (Hinweis: cudaGraphicsResourceGetMappedPointer)
	cudaGraphicsResourceGetMappedPointer((void**)&oldPos, &memSize, cudaPos[ping]);
	cudaGraphicsResourceGetMappedPointer((void**)&newPos, &memSize, cudaPos[1 - ping]);

	// DONE: Pointer auf die Daten von cudaNormal beschaffen.
	cudaGraphicsResourceGetMappedPointer((void**)&normals, &memSize, cudaNormal);

	// Launch update
	float stepSize = 0.5f; // steers how quickly the iterative refinement converges	
	updateCloth((float3*)newPos, (float3*)oldPos, (float3*)devPtrImpact, (float3*)devPtrVelocity, deltaTime, stepSize);

	// DONE: Unmap cudaNormal
	cudaGraphicsUnmapResources(1, &cudaNormal);

	// DONE: Unmap cudaPos (Hinweis: cudaGraphicsUnmapResources)
	cudaGraphicsUnmapResources(2, (cudaGraphicsResource**)cudaPos);

	// Swap ping pong roles.
	ping = 1 - ping;
}

unsigned int ClothSim::getVBOPos(unsigned int p) const
{
	return vboPos[p];
}

unsigned int ClothSim::getVBONormal() const
{
	return vboNormal;
}

unsigned int ClothSim::getResolutionX() const
{
	return RESOLUTION_X;
}

unsigned int ClothSim::getResolutionY() const
{
	return RESOLUTION_Y;
}

unsigned int ClothSim::getPingStatus() const
{
	return ping;
}
