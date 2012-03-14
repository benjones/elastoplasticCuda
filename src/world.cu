
#include "world.h"


__global__ void step(vec3* positions,
	   	   vec3* velocities,
	   	   double dt){


  positions[threadidx.x] -= velocities[threadidx.x]*dt;

}
