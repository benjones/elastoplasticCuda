
#ifndef _MATLIB_H
#define _MATLIB_H

#include <iostream>


#define signum(x) (( x > 0 ) - ( x < 0 ))

struct mat3{
  
  float m00, m01, m02, 
    m10, m11, m12, 
    m20, m21,m22;
  
};

struct mat2{
  float m00, m01, m10, m11;
};


__host__ __device__ bool matApproxEquals(const mat3& A, const mat3& B);

inline __host__ __device__ mat3 matDiag(const float4& v){
  mat3 ret;
  ret.m00 = v.x;
  ret.m01 = ret.m02 = 0;
  ret.m11 = v.y;
  ret.m10 = ret.m12 = 0;
  ret.m22 = v.z;
  ret.m20 = ret.m21 = 0;
  return ret;
}


inline __host__ __device__ mat3 matIdentity(){
  mat3 ret;
  ret.m00 = ret.m11 = ret.m22 = 1.0f;\
  ret.m01 = ret.m02 = ret.m10 = ret.m12 = ret.m20 = ret.m21 = 0.0f;
  return ret;
}

inline __host__ __device__ mat3 matZero(){
  mat3 ret;
  ret.m00 = ret.m01 = ret.m02 = ret.m10 = ret.m11 = ret.m12 = ret.m20 = ret.m21 = ret.m22 = 0;
  return ret;
}


inline void printVector(const float4& v ){
  std::cout << v.x << ' ' << v.y << ' ' <<v.z << std::endl;
}

inline void printMatrix(const mat3& m){
  std::cout << m.m00 << ' ' << m.m01 << ' ' << m.m02 << '\n' <<
    m.m10 << ' ' << m.m11 << ' ' << m.m12 << '\n' <<
    m.m20 << ' ' << m.m21 << ' ' << m.m22 << std::endl;
}
  

/*__device__ mat3 makeMat3(float i00, float i01, float i02,
			 float i10, float i11, float i12,
			 float i20, float i21, float i22){
  
  mat3 ret = {i00, i01, i02,
	      i10, i11, i12,
	      i20, i21, i22};

  return ret;
  
  }*/

mat3 __host__ __device__ matMult(const mat3& A, const mat3& B){
  mat3 out;
  out.m00 = A.m00*B.m00 + A.m01*B.m10 + A.m02*B.m20;
  out.m01 = A.m00*B.m01 + A.m01*B.m11 + A.m02*B.m21;
  out.m02 = A.m00*B.m02 + A.m01*B.m12 + A.m02*B.m22;

  out.m10 = A.m10*B.m00 + A.m11*B.m10 + A.m12*B.m20;
  out.m11 = A.m10*B.m01 + A.m11*B.m11 + A.m12*B.m21;
  out.m12 = A.m10*B.m02 + A.m11*B.m12 + A.m12*B.m22;

  out.m20 = A.m20*B.m00 + A.m21*B.m10 + A.m22*B.m20;
  out.m21 = A.m20*B.m01 + A.m21*B.m11 + A.m22*B.m21;
  out.m22 = A.m20*B.m02 + A.m21*B.m12 + A.m22*B.m22;

  return out;
}


__host__ __device__ mat3 matAdd(const mat3& A, const mat3& B){
  mat3 out;
  out.m00 = A.m00 + B.m00;
  out.m01 = A.m01 + B.m01;
  out.m02 = A.m02 + B.m02;
  out.m10 = A.m10 + B.m10;
  out.m11 = A.m11 + B.m11;
  out.m12 = A.m12 + B.m12;
  out.m20 = A.m20 + B.m20;
  out.m21 = A.m21 + B.m21;
  out.m22 = A.m22 + B.m22;

  return out;

}

__host__ __device__ mat3 matScale(const mat3& A, float s){
  mat3 out = A;
  out.m00 *= s;
  out.m01 *= s;
  out.m02 *= s;
  out.m10 *= s;
  out.m11 *= s;
  out.m12 *= s;
  out.m20 *= s;
  out.m21 *= s;
  out.m22 *= s;
  return out;
}



__host__ __device__ mat3 matTranspose(const mat3& in){
  mat3 out;
  out.m00 = in.m00;
  out.m01 = in.m10;
  out.m02 = in.m20;
  out.m10 = in.m01;
  out.m11 = in.m11;
  out.m12 = in.m21;
  out.m20 = in.m02;
  out.m21 = in.m12;
  out.m22 = in.m22;

  return out;
}

__host__ __device__ void SVD(const mat3& A,
		    mat3& U,
		    float4& S,
		    mat3& V){

  //see http://www.math.pitt.edu/~sussmanm/2071Spring08/lab09/index.html

  const float tol = 1.e-7;

  //mat3 checkMat;

  U = A; //copy it over to start with
  V.m00 = 1;
  V.m01 = 0;
  V.m02 = 0;
  V.m10 = 0;
  V.m11 = 1;
  V.m12 = 0;
  V.m20 = 0;
  V.m21 = 0;
  V.m22 = 1; //V starts as identity;

  /*  std::cout << "A: ";
  printMatrix(A);
  std::cout << "U: " ;
  printMatrix(U);
  std::cout << "V: ";
  printMatrix(V);*/

  float converge = 1.0f + tol;
  unsigned iterationCount = 0;
  while (converge > tol && iterationCount < 200){
    iterationCount++;
    converge = 0.0f;
    //do for (i, j) pairs:
    // (0,1), (0,2), (1,2)
    float alpha, beta, gamma;
    float zeta, t, c, s;
    //0,1 pair, i = 0, j = 1
    
    alpha = U.m00*U.m00 + U.m10*U.m10 + U.m20*U.m20;
    beta =  U.m01*U.m01 + U.m11*U.m11 + U.m21*U.m21;
    gamma = U.m00*U.m01 + U.m10*U.m11 + U.m20*U.m21;

    if(gamma){

      converge = max(converge, abs(gamma)*rsqrt(alpha*beta));

      zeta = (beta - alpha)/(2*gamma);
      t = signum(zeta)/(abs(zeta) + sqrtf(1 + zeta*zeta));
      c = rsqrtf(1 + t*t); //1/sqrt(1 + t^2)
      s = c*t;
      
      //std::cout << alpha << ' ' << beta << ' ' << gamma << ' ' << converge << ' ' << zeta << ' ' << t << ' ' << c << std::endl;
      
      //update cols i,j of U:
      t = U.m00;
      U.m00 = c*t - s*U.m01;
      U.m01 = s*t + c*U.m01;
      t = U.m10;
      U.m10 = c*t - s*U.m11;
      U.m11 = s*t + c*U.m11;
      t = U.m20;
      U.m20 = c*t - s*U.m21;
      U.m21 = s*t + c*U.m21;
      
      //update V:
      t = V.m00;
      V.m00 = c*t - s*V.m01;
      V.m01 = s*t + c*V.m01;
      t = V.m10;
      V.m10 = c*t - s*V.m11;
      V.m11 = s*t + c*V.m11;
      t = V.m20;
      V.m20 = c*t - s*V.m21;
      V.m21 = s*t + c*V.m21;
      
      //printMatrix(U);
      //std::cout << "v: ";
      //printMatrix(V);

    }

    /*matMult(U, matTranspose(V), checkMat);
    if (!matApproxEquals(A, checkMat)){
      std::cout << "error in 0, 1 pair" << std::endl;
    } else{
      std::cout << "0, 1 OK" << std::endl;
      }*/
      

    //i = 0, j = 2
    alpha = U.m00*U.m00 + U.m10*U.m10 + U.m20*U.m20;
    beta =  U.m02*U.m02 + U.m12*U.m12 + U.m22*U.m22;
    gamma = U.m00*U.m02 + U.m10*U.m12 + U.m20*U.m22;

    if(gamma){
      converge = max(converge, abs(gamma)*rsqrt(alpha*beta));
      
      zeta = (beta - alpha)/(2*gamma);
      
      t = signum(zeta)/(abs(zeta) + sqrtf(1 + zeta*zeta));
      c = rsqrtf(1 + t*t); //1/sqrt(1 + t^2)
      s = c*t;
  
      //update cols i,j of U:
      t = U.m00;
      U.m00 = c*t - s*U.m02;
      U.m02 = s*t + c*U.m02;
      t = U.m10;
      U.m10 = c*t - s*U.m12;
      U.m12 = s*t + c*U.m12;
      t = U.m20;
      U.m20 = c*t - s*U.m22;
      U.m22 = s*t + c*U.m22;
      
      //update V:
      t = V.m00;
      V.m00 = c*t - s*V.m02;
      V.m02 = s*t + c*V.m02;
      t = V.m10;
      V.m10 = c*t - s*V.m12;
      V.m12 = s*t + c*V.m12;
      t = V.m20;
      V.m20 = c*t - s*V.m22;
      V.m22 = s*t + c*V.m22;

    

    }

    /*matMult(U, matTranspose(V), checkMat);
    if (!matApproxEquals(A, checkMat)){
      std::cout << "error in 0, 2 pair" << std::endl;
    }else{
      std::cout << "0, 2 OK" << std::endl;
      }*/


    //i = 1, j = 2
    alpha = U.m01*U.m01 + U.m11*U.m11 + U.m21*U.m21;
    beta =  U.m02*U.m02 + U.m12*U.m12 + U.m22*U.m22;
    gamma = U.m01*U.m02 + U.m11*U.m12 + U.m21*U.m22;
    
    if(gamma){
      converge = max(converge, abs(gamma)*rsqrt(alpha*beta));
      
      zeta = (beta - alpha)/(2*gamma);
      t = signum(zeta)/(abs(zeta) + sqrtf(1 + zeta*zeta));
      c = rsqrtf(1 + t*t); //1/sqrt(1 + t^2)
      s = c*t;
      
      //update cols i,j of U:
      t = U.m01;
      U.m01 = c*t - s*U.m02;
      U.m02 = s*t + c*U.m02;
      t = U.m11;
      U.m11 = c*t - s*U.m12;
      U.m12 = s*t + c*U.m12;
      t = U.m21;
      U.m21 = c*t - s*U.m22;
      U.m22 = s*t + c*U.m22;
      
      //update V:
      t = V.m01;
      V.m01 = c*t - s*V.m02;
      V.m02 = s*t + c*V.m02;
      t = V.m11;
      V.m11 = c*t - s*V.m12;
      V.m12 = s*t + c*V.m12;
      t = V.m21;
      V.m21 = c*t - s*V.m22;
      V.m22 = s*t + c*V.m22;
    

    }
    /*matMult(U, matTranspose(V), checkMat);
    if (!matApproxEquals(A, checkMat)){
      std::cout << "error in 1, 2 pair" << std::endl;
    }else{
      std::cout << "1, 2 OK" << std::endl;
      }*/
    
    
  }

  //sing vals = column norms of U
  S.x = sqrtf(U.m00*U.m00 + U.m10*U.m10 + U.m20*U.m20);
  S.y = sqrtf(U.m01*U.m01 + U.m11*U.m11 + U.m21*U.m21);
  S.z = sqrtf(U.m02*U.m02 + U.m12*U.m12 + U.m22*U.m22);
  
  U.m00 /= S.x;
  U.m10 /= S.x;
  U.m20 /= S.x;

  U.m01 /= S.y;
  U.m11 /= S.y;
  U.m21 /= S.y;

  U.m02 /= S.z;
  U.m12 /= S.z;
  U.m22 /= S.z;
  
}

__host__ __device__ bool matApproxEquals(const mat3& A, const mat3& B){

  const float equalsEps = 1e-4;
  float totalError = (A.m00 - B.m00)*(A.m00 - B.m00) +
    (A.m01 - B.m01)*(A.m01 - B.m01) +
    (A.m02 - B.m02)*(A.m02 - B.m02) +
    (A.m10 - B.m10)*(A.m10 - B.m10) +
    (A.m11 - B.m11)*(A.m11 - B.m11) +
    (A.m12 - B.m12)*(A.m12 - B.m12) +
    (A.m20 - B.m20)*(A.m20 - B.m20) +
    (A.m21 - B.m21)*(A.m21 - B.m21) +
    (A.m22 - B.m22)*(A.m22 - B.m22);

  //std::cout << "total error: " << totalError << std::endl;
  return totalError <= equalsEps;

}

__host__ __device__ bool checkSVD(const mat3& A){

  mat3 U, V;
  float4 S;

  SVD(A, U, S, V);

  mat3 uut, vtv, sDiag, uSDiag, prod;
  uut = matMult(U, matTranspose(U));
  vtv = matMult(matTranspose(V), V);
  sDiag = matDiag(S);
  uSDiag = matMult(U, sDiag);
  prod = matMult(uSDiag, matTranspose(V));
  //std::cout << "A: " << std::endl;
  //printMatrix(A);
  //std::cout << "Approx A: " << std::endl;
  //printMatrix(prod);
  //std::cout << "U" << std::endl;
  //printMatrix(U);
  //std::cout << "S" << std::endl;
  //printVector(S);
  //std::cout << "V" << std::endl;
  //printMatrix(V);
  //std::cout << "u, v, product" << std::endl;

  return matApproxEquals(uut, matIdentity()) &&
    matApproxEquals(vtv, matIdentity()) &&
    matApproxEquals(prod, A);

}


__host__ __device__ mat3 pseudoInverse(const mat3& U, const float4& S, const mat3& V){

  float epsInv = 1e-4;

  mat3 ret;
  float4 Sinv;

  Sinv.x = S.x < epsInv ? 0 : 1/S.x;
  Sinv.y = S.y < epsInv ? 0 : 1/S.y;
  Sinv.z = S.z < epsInv ? 0 : 1/S.z;

  ret = matMult(V, matMult(matDiag(Sinv), matTranspose(U)));

  return ret;
}


__host__ __device__ float sphKernel(float radius, float test){
  
  float RR = radius*radius;
  float qq = test*test/RR;

  if(qq > 1)
    return 0;
  else{
    float dd = 1 - qq;
    return 315.0/(64.0*M_PI*RR*radius)*dd*dd*dd;
    
  }
}

__host__ __device__ float distance(const float4& a, const float4& b){
  
  float s = (b.x - a.x)*(b.x - a.x) + (b.y - a.y)*(b.y - a.y) + (b.z - a.z)*(b.z - a.z);
  return sqrtf(s);
}


__host__ __device__ mat3 outerProduct(const float4& a, const float4& b){
  
  mat3 ret;
  ret.m00 = a.x*b.x;
  ret.m01 = a.x*b.y;
  ret.m02 = a.x*b.z;

  ret.m10 = a.y*b.x;
  ret.m11 = a.y*b.y;
  ret.m12 = a.y*b.z;
  
  ret.m20 = a.z*b.x;
  ret.m21 = a.z*b.y;
  ret.m22 = a.z*b.z;
  
  return ret;
  
}



__host__ __device__ float vecMag(const float4& a){
  double s = a.x*a.x + a.y*a.y + a.z*a.z;
  return sqrt(s);
}

__host__ __device__ float4 vecSub(const float4& a, const float4& b){
  float4 ret = make_float4(a.x - b.x, a.y - b.y, a.z - b.z, 0);
  return ret;
}


__host__ __device__ float4 matVecMult(const mat3& m, const float4& v){
  float4 ret;
  ret.x = m.m00*v.x + m.m01*v.y + m.m02*v.z;
  ret.y = m.m10*v.x + m.m11*v.y + m.m12*v.z;
  ret.z = m.m20*v.x + m.m21*v.y + m.m22*v.z;
  return ret;

}

__host__ __device__ float4 vecScale(const float4& v, float s){
  float4 ret;
  ret.x = v.x*s;
  ret.y = v.y*s;
  ret.z = v.z*s;
  return ret;
}

#endif
