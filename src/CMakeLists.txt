PROJECT(elastoplasticCuda)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(CUDA)

FIND_PACKAGE(GLUT)
if (GLUT_FOUND)
   MESSAGE("GLUT Found: ${GLUT_LIBRARIES}")
   include_directories(${GLUT_INCLUDE_DIR})
else (GLUT_FOUND)
   MESSAGE("GLUT environment missing")
   SET(GLUT_LIBRARIES "-lglut")
endif (GLUT_FOUND)

FIND_PACKAGE(OpenGL)
if (OPENGL_FOUND)
   MESSAGE("OpenGL Found: ${OPENGL_LIBRARIES}")
else (OPENGL_FOUND)
   MESSAGE("OpenGL environment missing")   
endif (OPENGL_FOUND)
#SET(OPENGL_INCLUDE_DIR "/home/benjones/NVIDIA_GPU_Computing_SDK/C/common/inc/" 
#  "/home/benjones/NVIDIA_GPU_COMPUTING_SDK/shared/inc/")
include_directories(${OPENGL_INCLUDE_DIR})

SET(CUDA_NVCC_FLAGS "-arch=compute_20" "-code=sm_20" "-DUSE_ATOMIC_FLOAT" "-g" "-G" ) 
SET(CUDA_SDK_ROOT_DIR "/home/benjones/NVIDIA_GPU_Computing_SDK")
SET(CMAKE_CXX_FLAGS "-DUSE_ATOMIC_FLOAT" )

#SET(CUDA_SDK_ROOT_DIR "/uusoc/scratch/users/preethik/NVIDIA_GPU_Computing_SDK")
#SET(CUDA_NVCC_FLAGS "-arch=compute_13" "-code=sm_13" "-g" "-G" ) 

#LINK_LIBRARIES(
#	"-L${CUDA_SDK_ROOT_DIR}/C/lib" 
#	"-L${CUDA_SDK_ROOT_DIR}/C/common/lib/linux"
#	"-L/usr/local/cuda/lib64"
#	"-L/usr/lib64"
#)

INCLUDE(FindCUDA)
INCLUDE_DIRECTORIES(
	"/usr/local/cuda/include"
	"/usr/local/cuda/NVIDIA_GPU_Computing_SDK/C/common/inc" 
	"/home/benjones/NVIDIA_GPU_Computing_SDK/shared/inc/"
	"/home/benjones/NVIDIA_GPU_Computing_SDK/C/common/inc/"
)

CUDA_ADD_EXECUTABLE(testSVD testSVD.cu)
CUDA_ADD_EXECUTABLE(epSim 
	world.cu 
	knn_cublas_with_indexes.cu 
	main.cpp
)
TARGET_LINK_LIBRARIES(epSim ${OPENGL_LIBRARIES})
TARGET_LINK_LIBRARIES(epSim ${GLUT_LIBRARIES})
TARGET_LINK_LIBRARIES(epSim ${CUDA_SDK_ROOT_DIR}/C/common/lib/linux/libGLEW_x86_64.a)
TARGET_LINK_LIBRARIES(epSim ${CUDA_SDK_ROOT_DIR}/C/lib/libcutil_x86_64.a)
TARGET_LINK_LIBRARIES(epSim ${CUDA_SDK_ROOT_DIR}/C/lib/librendercheckgl_x86_64.a)
TARGET_LINK_LIBRARIES(epSim /usr/local/cuda/lib64/libcublas.so)


