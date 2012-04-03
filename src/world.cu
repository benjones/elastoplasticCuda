
#include "world.cuh"


__global__ void step(vec3* positions,
	   	   vec3* velocities,
	   	   double dt){


  positions[threadIdx.x].x -= velocities[threadIdx.x].x*dt;
  positions[threadIdx.x].y -= velocities[threadIdx.x].y*dt;
  positions[threadIdx.x].z -= velocities[threadIdx.x].z*dt;

}

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(int numParticles, 
		vec3* positions,
	   	vec3* velocities,
	   	double dt)
{
    // execute the kernel
   	step<<<numParticles, 1>>>(positions, velocities, dt);
}
