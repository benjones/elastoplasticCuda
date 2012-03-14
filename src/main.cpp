

#include "world.h"



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

  while(animating){
    step<<<numparticles, 1>>>(positions_d, velocities_d, dt);

  }

}
