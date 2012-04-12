#include <cstdlib>
#include "matLib.h"

const unsigned numTests = 16;

void __global__ svdKernel(mat3* As, bool* results){

  mat3 A = As[threadIdx.x];
  results[threadIdx.x] = checkSVD(A);
  
}


int main(int argc, char ** argv){


  mat3 A_h = {1,1,0, 0, 20, 0, 0, 0, 1};

  mat3 U_h, V_h;
  vec3 S_h;

  mat3 *A_d, *U_d, *V_d;
  vec3 *S_d;


  mat3 As[numTests];
  bool testResults[numTests];
  

  for(unsigned i = 0; i < numTests; ++i){
    As[i].m00 = (float)rand()*10.0f/(float)RAND_MAX;
    As[i].m01 = (float)rand()/(float)RAND_MAX;
    As[i].m02 = (float)rand()/(float)RAND_MAX;
    As[i].m10 = (float)rand()/(float)RAND_MAX;
    As[i].m11 = (float)rand()*10.0f/(float)RAND_MAX;
    As[i].m12 = (float)rand()/(float)RAND_MAX;
    As[i].m20 = (float)rand()/(float)RAND_MAX;
    As[i].m21 = (float)rand()/(float)RAND_MAX;
    As[i].m22 = (float)rand()*10.0f/(float)RAND_MAX;

  }
  
  mat3 * As_d;
  bool * testResults_d;
  cudaMalloc(&As_d, numTests*sizeof(mat3));
  cudaMalloc(&testResults_d, numTests*sizeof(bool));

  cudaMemcpy(As_d, As, sizeof(As), cudaMemcpyHostToDevice);
  
  svdKernel<<<1, numTests>>>(As_d, testResults_d);
  
  cudaThreadSynchronize();
  cudaMemcpy(testResults, testResults_d, sizeof(testResults), cudaMemcpyDeviceToHost);
  for(unsigned i = 0; i < numTests; ++i){
    std::cout << "test: " << i << "gpu: " << testResults[i] << " cpu: " << checkSVD(As[i]) << 
      std::endl;
    printMatrix(As[i]);
  }


  std::cout << matApproxEquals(matMult(As[0],pseudoInverse(As[0])), matIdentity()) << std::endl;

  return 0;
}
