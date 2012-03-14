

struct vec3{
  float x,y,z;

};

__global__ void step(vec3* positions,
		     vec3* velocities,
		     double dt);
