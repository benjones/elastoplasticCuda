
#ifndef _MATLIB_H
#define _MATLIB_H

#include <iostream>


#define signum(x) (( x > 0 ) - ( x < 0 ))

struct vec3{
  float x,y,z;

};


struct mat3{
  
  float m00, m01, m02, 
    m10, m11, m12, 
    m20, m21,m22;
  
};

struct mat2{
  float m00, m01, m10, m11;
};


inline void printVector(const vec3& v ){
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

__host__ /*__device__*/ void SVD(const mat3& A,
		    mat3& U,
		    vec3& S,
		    mat3& V){

  //see http://www.math.pitt.edu/~sussmanm/2071Spring08/lab09/index.html

  const float tol = 1.e-6;

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

  std::cout << "A: ";
  printMatrix(A);
  std::cout << "U: " ;
  printMatrix(U);
  std::cout << "V: ";
  printMatrix(V);

  float converge = 1.0f + tol;

  while (converge > tol){

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
      
      std::cout << alpha << ' ' << beta << ' ' << gamma << ' ' << converge << ' ' << zeta << ' ' << t << ' ' << c << std::endl;
      
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
      
      printMatrix(U);
      std::cout << "v: ";
      printMatrix(V);
    }
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
      U.m20 = c*t - s*U.m21;
      U.m21 = s*t + c*U.m21;
      
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
      U.m21 = c*t - s*U.m21;
      U.m21 = s*t + c*U.m21;
      
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




#endif
