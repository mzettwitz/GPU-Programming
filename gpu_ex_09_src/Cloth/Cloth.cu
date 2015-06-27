
// Includes
#include "CudaMath.h"
#include "Cloth.h"

// Computes the impacts between two points that are connected by a constraint in order to satisfy the constraint a little better.
__device__ float3 computeImpact(float3 me, float3 other, float stepsize, float h)
{
	const float aimedDistance = 1.0 / (float)RESOLUTION_X;
	float3 dir = other-me;
	float ldir = length(dir);
	if (ldir==0) return dir;
	float e = (ldir - aimedDistance) * 0.5;
	return dir/ldir * e / (h*h) * stepsize;
}

// Simple collision detection against a sphere at (0,0,0) with radius SPHERE_RADIUS and skin width SKIN_WIDTH.
__device__ float3 sphereCollision(float3 p, float h)
{
	// TODO: Testen, ob Punkt im inneren der Kugel ist. Wenn ja, dann eine Kraft berechnen, die sie wieder heraus bewegt.
}

// -----------------------------------------------------------------------------------------------
// Aufsummieren der Kr�fte, die von den benachbarten Gitterpunkten ausge�bt werden.
// impacts += ...
__global__ void computeImpacts(float3* oldPos, float3* impacts, float stepsize, float h)
{
	// TODO: Positionen der benachbarten Gitterpunkte und des eigenen Gitterpunktes ablesen.

	/////// Do we have information about neighboors above and under our position?
	/////// If yes - do we have to access them using blockIdx.y? 
	/////// Blocks are Streaming Multiprocessors, right? So we are hardly limited by using no threads?
	/*
	float3 pos = oldPos[blockIdx.x];
	if (blockIdx.x - 1 >= 0)
	{
		float3 posLeft = oldPos[blockIdx.x - 1];
		impacts[blockIdx.x] += computeImpact << <1, 1 >> > (pos, posLeft, stepsize, h);
	}
	if (blockId.x + 1 < gridDim.x)
	{
		float3 posRight = oldPos[blockIdx.x + 1];
		impacts[blockIdx.x] += computeImpact << <1, 1 >> > (pos, posRight, stepsize, h);
	}
	if (blockIdx.y - 1 >= 0)
	{
		float3 posTop = oldPos[blockIdx.y - 1];
		impacts[blockIdx.x] += computeImpact << <1, 1 >> > (pos, posTop, stepsize, h);
	}
	if (blockIdx.x + 1 < gridDim.y)
	{
		float3 posBottom = oldPos[blockIdx.y + 1];
		impacts[blockIdx.x] += computeImpact << <1, 1 >> > (pos, posBottom, stepsize, h);
	}
	*/
	// TODO: Kollisionsbehandlung mit Kugel durchf�hren.


	// TODO: Mit jedem Nachbar besteht ein Constraint. Dementsprechend f�r jeden Nachbar 
	//		 computeImpact aufrufen und die Ergebnisse aufsummieren.
		// see above
	

	// TODO: Die Summe der Kr�fte auf "impacts" des eigenen Gitterpunkts addieren.	
}	

// -----------------------------------------------------------------------------------------------
// Preview-Step
__global__ void previewSteps(	float3* newPos, float3* oldPos, float3* impacts, float3* velocity,								
								float h)
{
	// TODO: Berechnen, wo wir w�ren, wenn wir eine Integration von der bisherigen Position 
	//		 mit der bisherigen Geschwindigkeit und den neuen Impulsen durchf�hren.
	int index = blockIdx.x ; // right access?
	newPos[index] = oldPos[index] + (velocity[index] + impacts[index] - make_float3(0, GRAVITY, 0) * h) * h;
	//don't know whether operators are overloaded

}

// -----------------------------------------------------------------------------------------------
// Integrate velocity
__global__ void integrateVelocity(	float3* impacts, float3* velocity, float h)
{
	// TODO: Update velocity.
	int index = blockIdx.x; //right access?
	velocity[index] = velocity[index] * LINEAR_DAMPING + (impacts[index] - make_float3(0, GRAVITY, 0)) * h;
}

// -----------------------------------------------------------------------------------------------
// Test-Funktion die nur dazu da ist, damit man etwas sieht, sobald die VBOs gemapped werden...
__global__ void test( float3* newPos, float3* oldPos, float h)
{
	newPos[blockIdx.x] = oldPos[blockIdx.x] + make_float3(0, -h, 0);
}

// -----------------------------------------------------------------------------------------------
void updateCloth(	float3* newPos, float3* oldPos, float3* impacts, float3* velocity,					
					float h, float stepsize)
{
	dim3 blocks(RESOLUTION_X, RESOLUTION_Y-1, 1);
	dim3 blocks2(RESOLUTION_X, RESOLUTION_Y - 2, 1);

	// -----------------------------
	// Clear impacts
	cudaMemset(impacts, 0, RESOLUTION_X*RESOLUTION_Y*sizeof(float3));

	// Updating constraints is an iterative process.
	// The more iterations we apply, the stiffer the cloth become.
	for (int i=0; i<10; ++i)
	{
		// -----------------------------
		// TODO: previewSteps Kernel aufrufen (Vorhersagen, wo die Gitterpunkte mit den aktuellen Impulsen landen w�rden.)
		// newpos = oldpos + (velocity + impacts * h) * h
		previewSteps << <blocks2, 1 >> > (newPos, oldPos, impacts, velocity, h);
		
		// -----------------------------
		// TODO: computeImpacts Kernel aufrufen (Die Impulse neu berechnen, sodass die Constraints besser eingehalten werden.)
		// impacts += ...

	}

	// -----------------------------
	// TODO: Approximieren der Normalen
	
	// -----------------------------
	// TODO: Integrate velocity kernel ausf�hren
	// Der kernel berechnet:  velocity = velocity * LINEAR_DAMPING + (impacts - (0,GRAVITY,0)) * h 	
	integrateVelocity << <blocks, 1 >> > (impacts, velocity, h);
}