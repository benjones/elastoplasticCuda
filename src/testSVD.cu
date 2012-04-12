#include <cstdlib>
#include "matLib.h"

const unsigned numTests = 16;

void __global__ svdKernel(mat3* As, bool* results){

  mat3 A = As[threadIdx.x];
  //results[threadIdx.x] = checkSVD(A);
  
}


int main(int argc, char ** argv){


  mat3 A_h = {1,1,0, 0, 20, 0, 0, 0, 1};

  mat3 U_h, V_h;
  vec3 S_h;

  mat3 *A_d, *U_d, *V_d;
  vec3 *S_d;

  /*cudaMalloc(&A_d, sizeof(mat3));
  cudaMalloc(&U_d, sizeof(mat3));
  cudaMalloc(&V_d, sizeof(mat3));
  cudaMalloc(&S_d, sizeof(vec3));
  
  cudaMemcpy(A_d, &A_h, sizeof(mat3), cudaMemcpyHostToDevice);

  svdKernel<<<1,1>>>(A_d, U_d, S_d, V_d);

  cudaThreadSynchronize();

  cudaMemcpy(&U_h, U_d, sizeof(mat3), cudaMemcpyDeviceToHost);
  cudaMemcpy(&V_h, V_d, sizeof(mat3), cudaMemcpyDeviceToHost);
  cudaMemcpy(&S_h, S_d, sizeof(vec3), cudaMemcpyDeviceToHost);

  printVector(S_h);*/


  //host version:
  SVD(A_h, U_h, S_h, V_h);
  
  std::cout << "output: ";
  printMatrix(U_h);

  std::cout << "s";
  printVector(S_h);

  std::cout << "v";
  printMatrix(V_h);

  mat3 vtv, uut;
  matMult(V_h, matTranspose(V_h), vtv);
  printMatrix(vtv);

  matMult(matTranspose(U_h), U_h, uut);
  printMatrix(uut);

  std::cout << "checkSVD: " << checkSVD(A_h) << std::endl;

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

  return 0;
}
