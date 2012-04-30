#include "world.cuh"
#include "matLib.h"
#include "cutil.h"
#include "device_functions.h"
#include "cuPrintf.cu"

#define BLOCK_SIZE 64

const float kernelRadius = 2.0f;

const float density = 100.0f;

const float lambda = 1000000.0f;
const float mu = 1000000.0f;

const float forceIntMultiplier = 10000000.0f;		//integer force value = float * multiplier

__global__ void validateIndices(int numParticles, int* knnIndices){
	int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if(idx >= numParticles) return;
	
	int j = knnIndices[numParticles*(0)+idx];
	cuPrintf("KNNi %d: %d \n", idx, j);
	

//	for(int i=0; i<NUM_NEIGHBORS && i<numParticles; i++){
//		//int j = knnIndices[numParticles*i + idx]-1;
//		int j = numParticles*i + idx;		
//		cuPrintf(" %d:%d \n", i, j);
//		//int nIdx = numParticles*i + idx;
//		//int val = knnIndices[nIdx]-1;
//		//if (val < 0)
//		//	knnIndices[nIdx] = 0;
//		//if (val >= numParticles)
//		//	knnIndices[nIdx] = numParticles-1;
//	}

}

extern "C"
void launchValidateIndices(int numParticles, int* knnIndices){
	dim3 threadLayout(BLOCK_SIZE, 1, 1);
	int blockCnt = numParticles / BLOCK_SIZE;
	if(blockCnt*BLOCK_SIZE < numParticles) blockCnt++;
	dim3 blockLayout(blockCnt, 1);
	validateIndices<<<blockLayout, threadLayout>>>(numParticles, knnIndices);
	cudaThreadSynchronize();
	cudaPrintfDisplay(stdout, true);
}

__global__ void calculateForcesForNextFrame(
			int numParticles,
			float4* positions,
	   	   	float4* velocities,
			float4* embedded,
			float4* forces,
#ifndef USE_ATOMIC_FLOAT
			int4* externalForces,
#endif			
			float* masses,
			int* knnIndices,
	   	   	float dt){


	int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;

	if (idx >= numParticles) return;	// out of bounds


	

	//compute basis matrix;
	mat3 A = matZero();

	mat3 rhs1 = matZero();
	mat3 rhs2 = matZero();
	

	float wSum = 0;

	for(int i = 0; i < NUM_NEIGHBORS && i<numParticles; ++i){
		int j = knnIndices[numParticles*i + idx]-1;	// knn fxn returns indices as 1-indexed
	//for(int j = 0; j < numParticles; ++j){
		if (j == idx) continue;
	  //float4 vij = vecSub(embedded[j], embedded[idx]) ;
/*
		if(j<0)
			if(j==-1)
				float4 eMark = embedded[j];
			else
				float4 eMark = embedded[j];
		else if(j>=numParticles)
			if(j==numParticles+1)
				float4 eMark = embedded[j];
			else
				float4 eMark = embedded[j];
*/

		//float4 b = embedded[idx];		
		//float4 a = embedded[j];
		
		float4 vij = make_float4(
				embedded[j].x - embedded[idx].x, 
				embedded[j].y - embedded[idx].y, 
				embedded[j].z - embedded[idx].z, 0);
	  
	  float wij = sphKernel(kernelRadius, vecMag(vij));

	  A = matAdd(A, matScale(outerProduct(vij, vij), wij));
	  
	  //rhs1 = matAdd(rhs1, matScale(outerProduct(vecSub(positions[j], positions[idx]), vij), wij));
	  //rhs2 = matAdd(rhs1, matScale(outerProduct(vecSub(velocities[j], velocities[idx]), vij), wij));
	    
	  rhs1 = matAdd(rhs1, matScale(outerProduct(make_float4(
				positions[j].x - positions[idx].x, 
				positions[j].y - positions[idx].y, 
				positions[j].z - positions[idx].z, 0), vij), wij));
	  rhs2 = matAdd(rhs1, matScale(outerProduct(make_float4(
				velocities[j].x - velocities[idx].x, 
				velocities[j].y - velocities[idx].y, 
				velocities[j].z - velocities[idx].z, 0), vij), wij));

	  wSum += wij;
	    
	  
	}
	
	mat3 AU, AV;
	float4 AS;
	SVD(A, AU, AS, AV);

	

	float volume = sqrtf(AS.x*AS.y*AS.z/(1 + wSum*wSum*wSum));
	float mass = density/volume;
	masses[idx] = mass;

	mat3 Ainv = pseudoInverse(AU, AS, AV);;
	mat3 F = matMult(rhs1,Ainv);
	mat3 FDot = matMult(rhs2,Ainv);

	mat3 FU, FV;
	float4 FS;
	SVD(F, FU, FS, FV);
	float4 ones = make_float4(1.0f, 1.0f, 1.0f, 0.0f);
	//float4 strain = vecSub(FS, ones);
	float4 strain = make_float4(
				FS.x - ones.x, 
				FS.y - ones.y, 
				FS.z - ones.z, 0);
	float lTrace = lambda*(strain.x + strain.y + strain.z);
	mat3 stress = matScale(matDiag(strain), 2*mu);
	stress.m00 += lTrace;
	stress.m11 += lTrace;
	stress.m22 += lTrace;


	//un-diagonalize
	stress = matMult(matMult(FU, stress), matTranspose(FV));
	
	//gravity
#ifdef USE_ATOMIC_FLOAT	
	atomicAdd(&(forces[idx].y), -9.81f*mass);
#else
	forces[idx].y -= 9.81f*mass;
#endif

	//add forces:
	mat3 FE = matScale(matMult(stress, Ainv),-2.0*volume);
	for(int i = 0; i < NUM_NEIGHBORS && i<numParticles; ++i){
	int j = knnIndices[idx + numParticles*i]-1; // make zero-indexed
	//for(int j = 0; j < numParticles; ++j){
	if (j == idx) continue;
	  //recompute vector and weights...
	  //float4 vij = vecSub(embedded[j], embedded[idx]) ;
	  float4 vij = make_float4(
				embedded[j].x - embedded[idx].x, 
				embedded[j].y - embedded[idx].y, 
				embedded[j].z - embedded[idx].z, 0);
	  float wij = sphKernel(kernelRadius, vecMag(vij));
	  float4 force = vecScale(matVecMult(FE, vij),wij);
	    
#ifdef USE_ATOMIC_FLOAT
	  atomicAdd(&(forces[j].x), force.x);
	  atomicAdd(&(forces[j].y), force.y);
	  atomicAdd(&(forces[j].z), force.z);

	  atomicAdd(&(forces[idx].x), -force.x);
	  atomicAdd(&(forces[idx].y), -force.y);
	  atomicAdd(&(forces[idx].z), -force.z);


	  // multiply and convert to int so atomic add can be used... 
	  // later sync and convert back so that external forces can be incorporated
#else //USE_ATOMIC_FLOAT
	  atomicAdd(&(externalForces[i].x), (int)(force.x*forceIntMultiplier));
	  atomicAdd(&(externalForces[i].y), (int)(force.y*forceIntMultiplier));
	  atomicAdd(&(externalForces[i].z), (int)(force.z*forceIntMultiplier));
	  
	  forces[idx].x -= force.x;
	  forces[idx].y -= force.y;
	  forces[idx].z -= force.z;
	  
	    
#endif //USE_ATOMIC_FLOAT
	
	    //todo damping forces
    
	
	}
	


}

__global__ void clearForces(int numParticles, float4* forces 
#ifndef USE_ATOMIC_FLOAT
			    , int4* externalForces
#endif
			    ){
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if(idx < numParticles){
    forces[idx].x = 0;
    forces[idx].y = 0;
    forces[idx].z = 0;
#ifndef USE_ATOMIC_FLOAT
    externalForces[idx].x = 0;
    externalForces[idx].y = 0;
    externalForces[idx].z = 0;
#endif
  }
}

__global__ void incorporateExternalForces(
			int numParticles,
			float4* forces,
			int4* externalForces)
{
	int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if(idx < numParticles){
	    forces[idx].x += ((float)externalForces[idx].x)/forceIntMultiplier;
	    forces[idx].y += ((float)externalForces[idx].y)/forceIntMultiplier;
	    forces[idx].z += ((float)externalForces[idx].z)/forceIntMultiplier;
  	}
}

__global__ void integrateForces(int numParticles, float4* forces,
				float4* positions,
				float4* velocities,
				float* masses,
				int* knnIndices,
				float dt){
  
  int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  if(idx < numParticles){
    float mass = masses[idx];
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
#ifndef USE_ATOMIC_FLOAT
		int4* externalForces,
#endif
		float* masses,
		int* const knnIndices,
	   	float dt)
{

	
	//std::cout << "knnIndices at launch start: " <<  knnIndices << std::endl;


	dim3 threadLayout(BLOCK_SIZE, 1, 1);
	int blockCnt = numParticles / BLOCK_SIZE;
	if(blockCnt*BLOCK_SIZE < numParticles) blockCnt++;
	dim3 blockLayout(blockCnt, 1, 1);

	//validateIndices<<< blockLayout, threadLayout >>>(numParticles, knnIndices);

    // execute the kernel
	//std::cout << "-- kernel launch --"  << std::endl;
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
	  std::cout << "error before launching kernel fxns: " << cudaGetErrorString(err) << std::endl;
	  exit(1);
	} 
	clearForces<<<blockLayout, threadLayout>>>(numParticles, forces 
#ifndef USE_ATOMIC_FLOAT
						   , externalForces
#endif
						   );
	cudaThreadSynchronize();
	err = cudaGetLastError();
	if(cudaSuccess != err){
	  std::cout << "error in clear forces kernel: " << cudaGetErrorString(err) << std::endl;
	  exit(1);
	} else {
		//std::cout << "*** clear forces success!" << std::endl;
	}

	//validateIndices<<< blockLayout, threadLayout >>>(numParticles, knnIndices);

   	calculateForcesForNextFrame<<< blockLayout, threadLayout >>>(numParticles, 
								     positions, velocities, 
								     embedded, forces, 
#ifndef USE_ATOMIC_FLOAT
								     externalForces, 
#endif
								     masses, knnIndices, dt);
	CUT_CHECK_ERROR("before sync");

	cudaThreadSynchronize();

	CUT_CHECK_ERROR("after sync");

	err = cudaGetLastError();
	if(cudaSuccess != err){
	  std::cout << "error in calcForce kernel: " << cudaGetErrorString(err) << std::endl;
	  exit(1);
	} else {
		//std::cout << "*** calcForce success!" << std::endl;
	}
#ifndef USE_ATOMIC_FLOAT
   	incorporateExternalForces<<< blockLayout, threadLayout >>>(numParticles, forces, externalForces);
	cudaThreadSynchronize();

	err = cudaGetLastError();
	if(cudaSuccess != err){
	  std::cout << "error in incorpForces kernel: " << cudaGetErrorString(err) << std::endl;
	  exit(1);
	} else {
		//std::cout << "*** incorpForces success!" << std::endl;
	}
#endif

	integrateForces<<<blockLayout, threadLayout >>>(numParticles, forces, positions, velocities, masses, knnIndices, dt);
	cudaThreadSynchronize();

	err = cudaGetLastError();
	if(cudaSuccess != err){
	  std::cout << "error in integrate kernel: " << cudaGetErrorString(err) << std::endl;
	  exit(1);
	} else {
		//std::cout << "*** integrate success!" << std::endl;
	}

}

/////////////////////////// KNN STUFF /////////////////////////
__global__ void transposePositions(int numParticles, float4* positions_d, float* knnParticles_d){
	int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  	if(idx < numParticles){
		knnParticles_d[idx] = positions_d[idx].x;
		knnParticles_d[numParticles + idx] = positions_d[idx].y;
		knnParticles_d[(numParticles << 1) + idx] = positions_d[idx].z;
	}

}

// external wrapper to be called in main.cpp
extern "C" void prepPointsForKNN(int numParticles, float4* positions_d, float* knnParticles_d){
	// transpose positions into knn format
	dim3 threadLayout(BLOCK_SIZE, 1, 1);
	int blockCnt = numParticles / BLOCK_SIZE;
	if(blockCnt*BLOCK_SIZE < numParticles) blockCnt++;
	dim3 blockLayout(blockCnt, 1);
	transposePositions<<<blockLayout, threadLayout>>>(numParticles, positions_d, knnParticles_d);
	cudaThreadSynchronize();
}



