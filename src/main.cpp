#include <stdlib.h>

#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>

#include <iostream>

//#include "world.cuh"
using namespace std;

#define MAX(a,b) ((a > b) ? a : b)

// constants
const unsigned int window_width = 512;
const unsigned int window_height = 512;


// openGL variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

float anim = 0.0;

// declarations
extern "C" 
void launch_kernel( int numParticles, float4* positions, float4* velocities, float dt);

void runCuda(struct cudaGraphicsResource **vbo_resource);

// --- openGL ---
void initGL(int *argc, char **argv);
void cleanup();
void createVBO(GLuint* vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags);
void deleteVBO(GLuint* vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
//void mouse(int button, int state, int x, int y);
//void motion(int x, int y);

// -------
// variables
// -------

// mouse controls
	int mouse_old_x, mouse_old_y;
	int mouse_buttons = 0;
	float rotate_x = 0.0, rotate_y = 0.0;
	float translate_z = -3.0;
	
	// give these global-file scope for now...
  	int numParticles = 0;
  	float4* positions_h = NULL;
  	float4* velocities_h = NULL;
	float4* positions_d, *velocities_d;
  	float dt = .001;


int main(int argc, char **argv){
  	cudaError_t res;	//cuda success or error code
  numParticles = 10;
  positions_h = (float4*) malloc(sizeof(float4)*numParticles);
  velocities_h = (float4*) malloc(sizeof(float4)*numParticles);

	cout << "---init particles..." << endl;

  // init particles
  for(int i=0; i< numParticles; i++){
	positions_h[i].x = (float)i;
	positions_h[i].y = (float)i;
	positions_h[i].z = (float)i;
	
	velocities_h[i].x = 0.0f;
	velocities_h[i].y = -1.0f;
	velocities_h[i].z = 0.0f;
  }



	cout << "---set up openGL..." << endl;
	// setup openGL
	// --- this section copies and stripped down from simpleGL example ---
	// First initialize OpenGL context, so we can properly set the GL for CUDA.
	// This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
	initGL(&argc, argv);
	
	// use device with highest Gflops/s
	cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() );

	cout << "---create VBO..." << endl;
	// create VBO
	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);
	
	cout << "---set up device memory..." << endl; 
	// IMPORTANT: apparently you need to have this stuff after cudaGLSetGLDevice...

// TODO --- cudamalloc the positions and velocities...
	res = cudaMalloc((void**)&positions_d, sizeof(float4)*numParticles);
	if (res != cudaSuccess){
		fprintf (stderr, "!!!! gpu memory allocation error (B)\n");
		fprintf(stderr, "%s\n", cudaGetErrorString(res));
        return EXIT_FAILURE;
	}
	res = cudaMalloc((void**)&velocities_d, sizeof(float4)*numParticles);
	if (res != cudaSuccess){
		fprintf (stderr, "!!!! gpu memory allocation error (B)\n");
		fprintf(stderr, "%s\n", cudaGetErrorString(res));
        return EXIT_FAILURE;
	}

  // set up device memory
  cudaMemcpy(positions_d, positions_h, sizeof(float4)*numParticles, 
	     cudaMemcpyHostToDevice);
  cudaMemcpy(velocities_d, velocities_h, sizeof(float4)*numParticles, 
	     cudaMemcpyHostToDevice);

	cout << "---run CUDA first time..." << endl;
	// TODO move animation loop into glutMainLoop (display callback)
	// run the cuda part
	runCuda(&cuda_vbo_resource);

	cout << "---enter main loop..." << endl;
	// start rendering mainloop
	atexit(cleanup);
	glutMainLoop();
	// ---

//  bool animating = true;
//  double dt = .001;
//	int frames = 100;
//	int frameCnt = 0;

//  while(animating){
//    launch_kernel(numParticles, positions_d, velocities_d, dt);
    
    
    // TODO - figure out a better exit condition
//	frameCnt++;
//	if(frameCnt > frames) animating = false;
//  }

  cudaThreadExit();		// clean up the GPU
  
}
  
// TODO --- fill in method stubs
// make all the necessary calls to get openGL rolling
void initGL(int *argc, char **argv){
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize(window_width, window_height);
    glutCreateWindow("Elastoplastic Simulation");
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
	//glutMouseFunc(mouse);
    //glutMotionFunc(motion);

	glewInit();

    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);

    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

	CUT_CHECK_ERROR_GL();
	
}

// handles switching between cuda and openGL and calling the kernel fxn
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
// TODO - set this to draw points from positions buffer...
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;

    // DEPRECATED: cutilSafeCall(cudaGLMapBufferObject((void**)&dptr, vbo));
    cutilSafeCall(cudaGraphicsMapResources(1, vbo_resource, 0));

    size_t num_bytes; 

    cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, *vbo_resource));

    //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

    
	launch_kernel(numParticles, positions_d, velocities_d, dt);

    // unmap buffer object
    // DEPRECATED: cutilSafeCall(cudaGLUnmapBufferObject(vbo));
    cutilSafeCall(cudaGraphicsUnmapResources(1, vbo_resource, 0));

}

void display(){

    // run CUDA kernel to generate vertex positions
    runCuda(&cuda_vbo_resource);


    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);


    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, numParticles);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();

    anim += 0.01;
}

// keyboard callback
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) {
    case(27) :
        exit(0);
        break;
    }
}

void cleanup()
{
    deleteVBO(&vbo, cuda_vbo_resource);
}

void createVBO(GLuint* vbo, struct cudaGraphicsResource **vbo_res, unsigned int vbo_res_flags){

    if (vbo) {
		// create buffer object
		glGenBuffers(1, vbo);

		glBindBuffer(GL_ARRAY_BUFFER, *vbo);

		// initialize buffer object
		unsigned int size = numParticles * 4 * sizeof(float);
		glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	
		// register this buffer object with CUDA
		// DEPRECATED: cutilSafeCall(cudaGLRegisterBufferObject(*vbo));
		cutilSafeCall(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

		CUT_CHECK_ERROR_GL();
    } else {
		cutilSafeCall( cudaMalloc( (void **)&d_vbo_buffer, numParticles*4*sizeof(float) ) );
    }

}

void deleteVBO(GLuint* vbo, struct cudaGraphicsResource *vbo_res)
{
    if (vbo) {
	// unregister this buffer object with CUDA
	//DEPRECATED: cutilSafeCall(cudaGLUnregisterBufferObject(*pbo));
	cudaGraphicsUnregisterResource(vbo_res);

	
	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);
	
	*vbo = 0;
    } else {
	cudaFree(d_vbo_buffer);
	d_vbo_buffer = NULL;
    }
}

