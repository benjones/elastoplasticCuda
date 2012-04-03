
#ifndef _WORLD_H
#define _WORLD_H

#include "matLib.h"

__global__ void step(vec3* positions,
		     vec3* velocities,
		     double dt);


#endif
