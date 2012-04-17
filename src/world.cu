
#include "world.cuh"
#include "matLib.h"

#include "device_functions.h"

const float kernelRadius = .6f;

const float density = 1000.0f;

const float lambda = 10000.0f;
const float mu = 10000.0f;


__global__ void step(
			int numParticles,
			float4* positions,
	   	   	float4* velocities,
			float4* embedded,
			float4* forces,
			float* masses,
	   	   	float dt){


	int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	if (idx >= numParticles) return;	// out of bounds


	

	//compute basis matrix;
	mat3 A = matZero();

	mat3 rhs1 = matZero();
	mat3 rhs2 = matZero();
	

	float wSum = 0;
	for(int i = 0; i < numParticles; ++i){
	  if( i != idx){
	    float4 vij = vecSub(embedded[i], embedded[idx]) ;
	    float wij = sphKernel(kernelRadius, vecMag(vij));
	    A = matAdd(A, matScale(outerProduct(vij, vij), wij));
	    
	    rhs1 = matAdd(rhs1, matScale(outerProduct(vecSub(positions[i], positions[idx]), vij), wij));
	    rhs1 = matAdd(rhs1, matScale(outerProduct(vecSub(velocities[i], velocities[idx]), vij), wij));
	    
	    
	    wSum += wij;
	    
	  }
	}
	
	mat3 AU, AV;
	float4 AS;
	SVD(A, AU, AS, AV);

	

	float volume = sqrtf(AS.x*AS.y*AS.z/(1 + wSum*wSum*wSum));
	float mass = 1.0f;//density/volume;
	masses[idx] = mass;

	mat3 Ainv = pseudoInverse(AU, AS, AV);;
	mat3 F = matMult(rhs1,Ainv);
	mat3 FDot = matMult(rhs2,Ainv);

	mat3 FU, FV;
	float4 FS;
	SVD(F, FU, FS, FV);
	float4 ones = make_float4(1.0f, 1.0f, 1.0f, 0.0f);
	float4 strain = vecSub(FS, ones);
	float lTrace = lambda*(strain.x + strain.y + strain.z);
	mat3 stress = matScale(matDiag(strain), 2*mu);
	stress.m00 += lTrace;
	stress.m11 += lTrace;
	stress.m22 += lTrace;


	//un-diagonalize
	stress = matMult(matMult(FU, stress), matTranspose(FV));
	
	//gravity
	//atomicAdd(&(forces[idx].y), -9.81f*mass);
	forces[idx].y -= 9.81f*mass;
	
	//add forces:
	mat3 FE = matScale(matMult(stress, Ainv),-2.0*volume);
	for(int i = 0; i < numParticles; ++i){
	  if(i != idx){
	    //recompute vector and weights...
	    float4 vij = vecSub(embedded[i], embedded[idx]) ;
	    float wij = sphKernel(kernelRadius, vecMag(vij));
	    float4 force = vecScale(matVecMult(FE, vij),wij);
	    
	    //atomicAdd(&(forces[i].x), force.x);
	    //atomicAdd(&(forces[i].y), force.y);
	    //atomicAdd(&(forces[i].z), force.z);
	    
	    //atomicAdd(&(forces[idx].x), -force.x);
	    //atomicAdd(&(forces[idx].y), -force.y);
	    //atomicAdd(&(forces[idx].z), -force.z);
		forces[idx].x -= force.x;
		forces[idx].y -= force.y;
		forces[idx].z -= force.z;
	
	    //todo damping forces
    
	  }
	  }
	


}

__global__ void clearForces(int numParticles, float4* forces){
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if(idx < numParticles){
    forces[idx].x = 0;
    forces[idx].y = 0;
    forces[idx].z = 0;
  }
}

__global__ void integrateForces(int numParticles, float4* forces,
				float4* positions,
				float4* velocities,
				float* masses,
				float dt){
  
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if(idx < numParticles){
    float mass = 100.0f;//masses[idx];
    //symplectic euler:
    velocities[idx].x += forces[idx].x*dt/mass;
    velocities[idx].y += forces[idx].y*dt/mass;
    velocities[idx].z += forces[idx].z*dt/mass;
    
    positions[idx].x += velocities[idx].x*dt;
    positions[idx].y += velocities[idx].y*dt;
    positions[idx].z += velocities[idx].z*dt;
    
    
    
    // check boundaries
    if(positions[idx].y < GROUND_HEIGHT) {
      
      positions[idx].y = GROUND_HEIGHT;
      //velocities[idx].y = 0;
    }
  }
}


// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(
		int numParticles, 
		float4* positions,
	   	float4* velocities,
		float4* embedded,
		float4* forces,
		float* masses,
	   	float dt)
{
	dim3 threadLayout(BLOCK_SIZE, 1, 1);
	int blockCnt = numParticles / BLOCK_SIZE;
	if(blockCnt*BLOCK_SIZE < numParticles) blockCnt++;
	dim3 blockLayout(blockCnt, 1);
    // execute the kernel
	//std::cout << "-- kernel launch --"  << std::endl;
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
	  std::cout << "error before launching kernel fxns: " << cudaGetErrorString(err) << std::endl;
	  exit(1);
	} 
	clearForces<<<blockLayout, threadLayout>>>(numParticles, forces);
	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(cudaSuccess != err){
	  std::cout << "error in clear forces kernel: " << cudaGetErrorString(err) << std::endl;
	  exit(1);
	} else {
		//std::cout << "*** clear forces success!" << std::endl;
	}


   	step<<< blockLayout, threadLayout >>>(numParticles, positions, velocities, embedded,forces, masses, dt);
	cudaThreadSynchronize();

	err = cudaGetLastError();
	if(cudaSuccess != err){
	  std::cout << "error in step kernel: " << cudaGetErrorString(err) << std::endl;
	  exit(1);
	} else {
		//std::cout << "*** step success!" << std::endl;
	}

	integrateForces<<<blockLayout, threadLayout >>>(numParticles, forces, positions, velocities, masses, dt);
	cudaThreadSynchronize();

	err = cudaGetLastError();
	if(cudaSuccess != err){
	  std::cout << "error in integrate kernel: " << cudaGetErrorString(err) << std::endl;
	  exit(1);
	} else {
		//std::cout << "*** integrate success!" << std::endl;
	}

}
