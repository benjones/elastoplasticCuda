
#include "world.cuh"


__global__ void step(float4* positions,
	   	   float4* velocities,
	   	   float dt){


  positions[threadIdx.x].x += velocities[threadIdx.x].x*dt;
  positions[threadIdx.x].y += velocities[threadIdx.x].y*dt;
  positions[threadIdx.x].z += velocities[threadIdx.x].z*dt;

}

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(
		int numParticles, 
		float4* positions,
	   	float4* velocities,
	   	float dt)
{
    // execute the kernel
   	step<<<numParticles, 1>>>(positions, velocities, dt);
}
