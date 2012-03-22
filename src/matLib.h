
#ifndef _MATLIB_H
#define _MATLIB_H


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
}
  

/*__device__ mat3 makeMat3(float i00, float i01, float i02,
			 float i10, float i11, float i12,
			 float i20, float i21, float i22){
  
  mat3 ret = {i00, i01, i02,
	      i10, i11, i12,
	      i20, i21, i22};

  return ret;
  
  }*/

__device__ void SVD(const mat3& A,
		    mat3& U,
		    vec3& S,
		    Mat3& V);


#endif
