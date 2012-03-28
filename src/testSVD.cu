
#include "matLib.h"

void __global__ svdKernel(const mat3& A, mat3* U, vec3* S, mat3* V){


  SVD(A, *U, *S, *V);

  

}


int main(int argc, char ** argv){


  mat3 A_h = {1,0,0, 0, 1, 0, 0, 0, 1};

  mat3 U_h, V_h;
  vec3 S_h;

  mat3 *A_d, *U_d, *V_d;
  vec3 *S_d;

  cudaMalloc(&A_d, sizeof(mat3));
  cudaMalloc(&U_d, sizeof(mat3));
  cudaMalloc(&V_d, sizeof(mat3));
  cudaMalloc(&S_d, sizeof(vec3));
  
  cudaMemcpy(A_d, &A_h, sizeof(mat3), cudaMemcpyHostToDevice);

  svdKernel<<<1,1>>>(A_d, U_d, S_d, V_d);

  cudaMemcpy(&U_h, U_d, sizeof(mat3), cudaMemcpyDeviceToHost);
  cudaMemcpy(&V_h, V_d, sizeof(mat3), cudaMemcpyDeviceToHost);
  cudaMemcpy(&S_h, S_d, sizeof(vec3), cudaMemcpyDeviceToHost);

  printVector(S_h);

  return 0;
}
