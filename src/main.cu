#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <iostream>
#include <float.h>
#include <fstream>
#include <curand_kernel.h>
#include "ray.h"
#include "helper_math.h"
#include "hitable_list.h"
#include "sphere.h"
#include "hitable.h"
#include "camera.h"
#include "material.h"
#include "obj_loader.h"
#include "triangle.h"
#include "LocalRenderer/Window.h"
#include "LocalRenderer/Renderer.h"
#include "cuda_utils.h"
#include "bvh.h"


/**
 * Initializes the rendering process.
 *
 * This kernel function is responsible for initializing the rendering process by setting up the random number generator states.
 *
 * @param nx The width of the image.
 * @param ny The height of the image.
 * @param rand_state Pointer to the array of random number generator states.
 */
__global__ void render_init(int nx, int ny, curandState *rand_state) {
    //render_init doesnt have to be separate kernel, dona that way for clarity 
    //better performance to do it in the render kernel (will change later)
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= nx) || (j >= ny)) return;
    int pixel_index = j*nx + i;
    //Each thread gets diffrent seed, same sequence number, no offset
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

/**
 * @brief Renders the scene using path tracing algorithm.
 *
 * This CUDA kernel function is responsible for rendering the scene using the path tracing algorithm.
 * It takes in the framebuffer `fb`, the maximum width and height of the image `max_x` and `max_y`,
 * the number of samples per pixel `sample_per_pixel`, an array of camera pointers `cam`, an array of
 * hitable pointers `world`, and the random state for each thread `rand_state`.
 *
 * @param fb The framebuffer to store the rendered image.
 * @param max_x The maximum width of the image.
 * @param max_y The maximum height of the image.
 * @param sample_per_pixel The number of samples per pixel.
 * @param cam An array of camera pointers.
 * @param world An array of hitable pointers representing the scene.
 * @param rand_state The random state for each thread.
 */
__global__ void render(uint8_t *fb, int max_x, int max_y,int sample_per_pixel, camera **cam,bvh **world, curandState *rand_state) {
    
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y;
   if((i >= max_x) || (j >= max_y)) return;
   int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    //Antialiasing
    float3 col = make_float3(0, 0, 0);
    for (int s=0; s<sample_per_pixel; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v);
        col += (*cam)->ray_color(r, world, &local_rand_state);
    }
    int3 color = make_int3(255.99 * col/float(sample_per_pixel)); //average color of samples
    fb[3 * pixel_index] = color.x;
    fb[3 * pixel_index + 1] = color.y;
    fb[3 * pixel_index + 2] = color.z;
}

/**
 * @brief CUDA kernel to create the world, list of objects, and camera.
 *
 * This CUDA kernel is responsible for creating the world, list of objects, and camera
 * in the GPU memory. It takes in pointers to the device memory where the list of objects,
 * world, and camera will be stored.
 *
 * @param d_world Pointer to the device memory where the world will be stored.
 * @param d_camera Pointer to the device memory where the camera will be stored.
 * @param d_list Pointer to the device memory where the list of objects will be stored.
 * @param d_list_size Number of objects in objects array 
 */
__global__ void create_world(bvh **d_world, camera **d_camera, hitable **d_list, int d_list_size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {                       
        *d_world  = new bvh(d_list, d_list_size);
        *d_camera = new camera();
    }
}

__global__ void free_world(hitable **d_list, bvh **d_world, camera **d_camera, int d_list_size) {
    for (int i=0; i < d_list_size; i++) {
        delete d_list[i];
    }

    delete *d_world;
    delete *d_camera;    
}

int main()
{
    int nx = 1600;
    int ny = 900;
    
    int tx = 8; //thread amount should be a multiple of 32
    int ty = 8;
    float aspect_ratio = float(nx) / float(ny);

    int num_pixels = nx*ny;

    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    size_t fb_size = num_pixels*sizeof(uint8_t) * 3;

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    uint8_t *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    //create_world

    // Load object
    const char *file_path = "models/cube.obj";
    obj_loader loader(file_path);

    int number_of_faces = loader.get_total_number_of_faces();

    hitable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, number_of_faces * sizeof(hitable *)));

    loader.load_faces(d_list);

    bvh **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(bvh *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

    create_world<<<1,1>>>(d_world, d_camera, d_list, number_of_faces);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render<<<blocks, threads>>>(fb, nx, ny,
        100, d_camera,
        d_world,
        d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    free_world<<<1, 1>>>(d_list, d_world, d_camera, number_of_faces);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Save result to a PPM image
    std::ofstream myfile;
    myfile.open("out.ppm");
    myfile << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = 3 * (j*nx + i);
            myfile << fb[pixel_index] << " " << fb[pixel_index + 1] << " " << fb[pixel_index + 2] << "\n";
        }
    }
    myfile.close();

    Window window(nx, ny, "MultiGPU-PathTracer");
    Renderer renderer(window);

    while (!window.shouldClose()) {
        window.pollEvents();
        renderer.renderFrame(fb);
	window.swapBuffers();	

	}

    checkCudaErrors(cudaFree(fb));
    return 0;
}
