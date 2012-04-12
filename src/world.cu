
#include "world.cuh"


__global__ void step(
			int numParticles,
			float4* positions,
	   	   	float4* velocities,
	   	   	float dt){

	int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	if (idx >= numParticles) return;	// out of bounds

  	positions[idx].x += velocities[idx].x*dt;
  	positions[idx].y += velocities[idx].y*dt;
  	positions[idx].z += velocities[idx].z*dt;

	// check boundaries
	if(positions[idx].y < GROUND_HEIGHT) positions[idx].y = GROUND_HEIGHT;
}

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(
		int numParticles, 
		float4* positions,
	   	float4* velocities,
	   	float dt)
{
	dim3 threadLayout(BLOCK_SIZE, 1, 1);
	int blockCnt = numParticles / BLOCK_SIZE;
	if(blockCnt*BLOCK_SIZE < numParticles) blockCnt++;
	dim3 blockLayout(blockCnt, 1);
    // execute the kernel
   	step<<< blockLayout, threadLayout >>>(numParticles, positions, velocities, dt);
}
