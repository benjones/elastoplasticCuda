#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "world.cuh"

// declarations
extern "C" 
void launch_kernel(int numParticles, vec3* positions, vec3* velocities, double dt);

int main(int argc, char **argv){
  
  int numParticles = 10;
  vec3* positions_h = (vec3*) malloc(sizeof(vec3)*numParticles);
  vec3* velocities_h = (vec3*) malloc(sizeof(vec3)*numParticles);


  vec3* positions_d, *velocities_d;
  cudaMemcpy(positions_d, positions_h, sizeof(vec3)*numParticles, 
	     cudaMemcpyHostToDevice);
  cudaMemcpy(velocities_d, velocities_h, sizeof(vec3)*numParticles, 
	     cudaMemcpyHostToDevice);

  bool animating = true;
  double dt = .001;
	int frames = 100;
	int frameCnt = 0;

  while(animating){
    launch_kernel(numParticles, positions_d, velocities_d, dt);
    
    
    // TODO - figure out a better exit condition
    frameCnt++;
    if(frameCnt > frames) animating = false;
    
  }
  
}
