#ifndef _world_cuh_
#define _world_cuh_ 

#define GROUND_HEIGHT -1.0f
#define NUM_NEIGHBORS 128

// TODO - we may want to turn this into an aligned struct for coalesced memory access...
// USE "float4" cuda struct instead of vec3 !!!!!!!!!!!!!!!!
//struct vec3{
//  float x,y,z;

//};

// use global extern call "launch_kernel" defined in world.cu to access from .cpp files
// since the compiler can't seem to handle __global__
//__global__ void step(vec3* positions,
//		     vec3* velocities,
//		     float dt);

#endif	// _world_cuh_
