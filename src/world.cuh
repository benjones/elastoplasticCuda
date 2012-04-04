#ifndef _world_cuh_
#define _world_cuh_ 

struct vec3{
  float x,y,z;

};

// use global extern call "launch_kernel" defined in world.cu to access from .cpp files
// since the compiler can't seem to handle __global__
//__global__ void step(vec3* positions,
//		     vec3* velocities,
//		     double dt);

#endif	// _world_cuh_
