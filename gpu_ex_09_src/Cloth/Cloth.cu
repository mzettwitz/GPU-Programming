

// Includes
#include "CudaMath.h"
#include "Cloth.h"

// Computes the impacts between two points that are connected by a constraint in order to satisfy the constraint a little better.
__device__ float3 computeImpact(float3 me, float3 other, float stepsize, float h)
{
	const float aimedDistance = 1.0 / (float)RESOLUTION_X;
	float3 dir = other - me;
	float ldir = length(dir);
	if (ldir == 0) return dir;
	float e = (ldir - aimedDistance) * 0.5;
	return dir / ldir * e / (h*h) * stepsize;
}

// Simple collision detection against a sphere at (0,0,0) with radius SPHERE_RADIUS and skin width SKIN_WIDTH.
__device__ float3 sphereCollision(float3 p, float h)
{
	// DONE: Testen, ob Punkt im inneren der Kugel ist. Wenn ja, dann eine Kraft berechnen, die sie wieder heraus bewegt.

	//provided that point is in center of skin
	//add the half skin thickness to point p
	p.x += SKIN_WIDTH / 2;
	p.y += SKIN_WIDTH / 2;
	p.z += SKIN_WIDTH / 2;

	// compute distance from p to sphere center
	float dist = length(p);

	if (dist<SPHERE_RADIUS)		// inside sphere
		p = (p * (SPHERE_RADIUS / dist)) / h;
	else
	{
		p.x = 0;
		p.y = 0;
		p.z = 0;
	}

	return p;
}

// -----------------------------------------------------------------------------------------------
// Aufsummieren der Kräfte, die von den benachbarten Gitterpunkten ausgeübt werden.
// impacts += ...
__global__ void computeImpacts(float3* oldPos, float3* impacts, float stepsize, float h)
{
	// DONE: Positionen der benachbarten Gitterpunkte und des eigenen Gitterpunktes ablesen.
	int x = blockIdx.x;
	int y = blockIdx.y;

	int index = x * RESOLUTION_Y + y;

	float3 sumImpacts = make_float3(0, 0, 0);

	if (y - 1 >= 0) 				// top
		sumImpacts += computeImpact(oldPos[index], oldPos[index - 1], stepsize, h);

	if (y + 1 < RESOLUTION_Y) 		// bottom
		sumImpacts += computeImpact(oldPos[index], oldPos[index + 1], stepsize, h);

	if (x - 1 >= 0)  				// left
		sumImpacts += computeImpact(oldPos[index], oldPos[index - RESOLUTION_Y], stepsize, h);

	if (x + 1 < RESOLUTION_X) 		// right
		sumImpacts += computeImpact(oldPos[index], oldPos[index + RESOLUTION_Y], stepsize, h);


	// DONE: Kollisionsbehandlung mit Kugel durchführen.
	sumImpacts += sphereCollision(oldPos[index], h);


	// DONE: Mit jedem Nachbar besteht ein Constraint. Dementsprechend für jeden Nachbar 
	//		 computeImpact aufrufen und die Ergebnisse aufsummieren.
	// see above


	// DONE: Die Summe der Kräfte auf "impacts" des eigenen Gitterpunkts addieren.
	impacts[index] += sumImpacts;
}

// -----------------------------------------------------------------------------------------------
// Preview-Step
__global__ void previewSteps(float3* newPos, float3* oldPos, float3* impacts, float3* velocity,
	float h)
{
	// DONE: Berechnen, wo wir wären, wenn wir eine Integration von der bisherigen Position 
	//		 mit der bisherigen Geschwindigkeit und den neuen Impulsen durchführen.
	//newpos = oldpos + (velocity + impacts * h) * h

	int index = blockIdx.x * RESOLUTION_Y + blockIdx.y;
	newPos[index] = oldPos[index] + (velocity[index] + (impacts[index]) * h) * h;

}

// -----------------------------------------------------------------------------------------------
// Integrate velocity
__global__ void integrateVelocity(float3* impacts, float3* velocity, float h)
{
	// DONE: Update velocity.	
	int index = blockIdx.x * RESOLUTION_Y + blockIdx.y;
	velocity[index] = velocity[index] * LINEAR_DAMPING + (impacts[index] - make_float3(0, GRAVITY, 0)) * h;
}

// -----------------------------------------------------------------------------------------------
// Test-Funktion die nur dazu da ist, damit man etwas sieht, sobald die VBOs gemapped werden...
__global__ void  test(float3* newPos, float3* oldPos, float h)
{
	newPos[blockIdx.x] = oldPos[blockIdx.x] + make_float3(0, -h, 0);
}

// Approximate normal addicted to neighbors
__global__ void  computeNormal(float3* normals, float3* position)
{
	int x = blockIdx.x;
	int y = blockIdx.y;

	// index variables
	int index = x * RESOLUTION_Y + y;
	int top_pos = index - 1;
	int bottom_pos = index + 1;
	int left_pos = index - RESOLUTION_Y;
	int right_pos = index + RESOLUTION_Y;

	float3 normal = make_float3(0, 0, 0);

	// borders: top, bot, left, right
	if (y - 1 >= 0 && y + 1 < RESOLUTION_Y && x - 1 >= 0 && x + 1 < RESOLUTION_X) 
		normal = cross((position[top_pos] - position[bottom_pos]), (position[right_pos] - position[left_pos]));

	normals[index] = normalize(normal);
}

// -----------------------------------------------------------------------------------------------
void updateCloth(float3* newPos, float3* oldPos, float3* impacts, float3* velocity,
	float h, float stepsize, float3* normals)
{
	dim3 blocks(RESOLUTION_X, RESOLUTION_Y - 1, 1);

	// -----------------------------
	// Clear impacts
	cudaMemset(impacts, 0, RESOLUTION_X*RESOLUTION_Y*sizeof(float3));

	// Updating constraints is an iterative process.
	// The more iterations we apply, the stiffer the cloth become.
	for (int i = 0; i<10; ++i)
	{
		// -----------------------------
		// DONE: previewSteps Kernel aufrufen (Vorhersagen, wo die Gitterpunkte mit den aktuellen Impulsen landen würden.)
		// newpos = oldpos + (velocity + impacts * h) * h
		previewSteps << <blocks, 1 >> >(newPos, oldPos, impacts, velocity, h);

		// -----------------------------
		// DONE: computeImpacts Kernel aufrufen (Die Impulse neu berechnen, sodass die Constraints besser eingehalten werden.)
		// impacts += ...
		computeImpacts << <blocks, 1 >> >(newPos, impacts, stepsize, h);

	}

	// -----------------------------
	// DONE: Approximieren der Normalen
	computeNormal << <blocks, 1 >> > (normals, newPos);

	// -----------------------------
	// DONE: Integrate velocity kernel ausführen
	// Der kernel berechnet:  velocity = velocity * LINEAR_DAMPING + (impacts - (0,GRAVITY,0)) * h 	
	integrateVelocity << <blocks, 1 >> > (impacts, velocity, h);
}