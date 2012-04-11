#include <stdlib.h>

#include <GL/glew.h>
#include <GL/glut.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>

#include <iostream>

#include "world.cuh"
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
void launch_kernel(float4* dptr, int numParticles, vec3* positions, vec3* velocities, float dt);

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
  	vec3* positions_h = NULL;
  	vec3* velocities_h = NULL;
	vec3* positions_d, *velocities_d;
  	float dt = .001;


int main(int argc, char **argv){
  	cudaError_t res;	//cuda success or error code
  numParticles = 10;
  positions_h = (vec3*) malloc(sizeof(vec3)*numParticles);
  velocities_h = (vec3*) malloc(sizeof(vec3)*numParticles);

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
	res = cudaMalloc((void**)&positions_d, sizeof(vec3)*numParticles);
	if (res != cudaSuccess){
		fprintf (stderr, "!!!! gpu memory allocation error (B)\n");
		fprintf(stderr, "%s\n", cudaGetErrorString(res));
        return EXIT_FAILURE;
	}
	res = cudaMalloc((void**)&velocities_d, sizeof(vec3)*numParticles);
	if (res != cudaSuccess){
		fprintf (stderr, "!!!! gpu memory allocation error (B)\n");
		fprintf(stderr, "%s\n", cudaGetErrorString(res));
        return EXIT_FAILURE;
	}

  // set up device memory
  cudaMemcpy(positions_d, positions_h, sizeof(vec3)*numParticles, 
	     cudaMemcpyHostToDevice);
  cudaMemcpy(velocities_d, velocities_h, sizeof(vec3)*numParticles, 
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
// ######## TODO #################
// #### Get this set up for our launch_kernel.....
cout << "A";
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
cout << "B";
    // DEPRECATED: cutilSafeCall(cudaGLMapBufferObject((void**)&dptr, vbo));
    cutilSafeCall(cudaGraphicsMapResources(1, vbo_resource, 0));
cout << "C";
    size_t num_bytes; 
cout << "D";
    cutilSafeCall(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, *vbo_resource));
cout << "E";
    printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);
cout << "F";
    
	launch_kernel(dptr, numParticles, positions_d, velocities_d, dt);
cout << "G";
    // unmap buffer object
    // DEPRECATED: cutilSafeCall(cudaGLUnmapBufferObject(vbo));
    cutilSafeCall(cudaGraphicsUnmapResources(1, vbo_resource, 0));
cout << "H" << endl;
}

void display(){
	cout << "**D: run cuda" << endl;
    // run CUDA kernel to generate vertex positions
    runCuda(&cuda_vbo_resource);

	cout << "**D: back from cuda" << endl;
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

	cout << "**D: render" << endl;
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
	cout << "1" << endl;
    if (vbo) {
	cout << "2" << endl;
	// create buffer object
	glGenBuffers(1, vbo);
	cout << "3" << endl;
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	cout << "4" << endl;

	// initialize buffer object
	unsigned int size = numParticles * 4 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
cout << "5" << endl;
	// register this buffer object with CUDA
	// DEPRECATED: cutilSafeCall(cudaGLRegisterBufferObject(*vbo));
	cutilSafeCall(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));
	cout << "5.5" << endl;

	CUT_CHECK_ERROR_GL();
    } else {
cout << "6" << endl;
	cutilSafeCall( cudaMalloc( (void **)&d_vbo_buffer, numParticles*4*sizeof(float) ) );
    }
cout << "7" << endl;
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

