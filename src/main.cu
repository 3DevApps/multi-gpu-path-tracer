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
#include "bvh.h"







#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA ERROR = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        const char* error_string = cudaGetErrorString(result); 
        std::cerr << error_string << " " << std::endl;
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}


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
__global__ void render(float3 *fb, int max_x, int max_y,int sample_per_pixel, camera **cam,bvh **world, curandState *rand_state) {
    
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

   fb[pixel_index] = col/float(sample_per_pixel); //average color of samples
}

/**
 * @brief CUDA kernel to create the world, list of objects, and camera.
 *
 * This CUDA kernel is responsible for creating the world, list of objects, and camera
 * in the GPU memory. It takes in pointers to the device memory where the list of objects,
 * world, and camera will be stored.
 *
 * @param d_list Pointer to the device memory where the list of objects will be stored.
 * @param d_world Pointer to the device memory where the world will be stored.
 * @param d_camera Pointer to the device memory where the camera will be stored.
 */
__global__ void create_world(hitable **d_list, bvh **d_world,camera **d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(make_float3(0,0,-1), 0.5,
                               new lambertian(make_float3(0.7, 0.7, 0.5)));
        d_list[1] = new sphere(make_float3(0,-100.5,-1), 100,
                               new lambertian(make_float3(0.8, 0.8, 0.0)));
        d_list[2] = new sphere(make_float3(1,0,-1), 0.5,
                               new metal(make_float3(0.8, 0.6, 0.8),0.0));
        d_list[3] = new sphere(make_float3(-1,0,-1), 0.5, //negative radius trick makes it face inwards
                               new dielectric(1.5f));
        d_list[4] = new sphere(make_float3(-1,0,-1), -0.45,
                                 new dielectric(1.5));                    
        *d_world  = new bvh(d_list,5);
        *d_camera = new camera();
    }
}

__global__ void free_world(hitable **d_list, bvh **d_world,camera **d_camera) {
    for(int i=0; i < 5; i++) {
        delete ((sphere *)d_list[i])->mat_ptr;
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
    size_t fb_size = num_pixels*sizeof(float3);

    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
    // allocate FB
    float3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    //create_world
    hitable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 5*sizeof(hitable *)));
    bvh **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(bvh *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
    create_world<<<1,1>>>(d_list,d_world,d_camera);
    
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    
    //render
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // return;
    render<<<blocks, threads>>>(fb, nx, ny,
        100, d_camera,
        d_world,
        d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    // free_world
    free_world<<<1, 1>>>(d_list, d_world,d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //output
    std::ofstream myfile;
    myfile.open("out.ppm");
    myfile << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j*nx + i;
            int3 color = make_int3(255.99*fb[pixel_index].x, 255.99*fb[pixel_index].y, 255.99*fb[pixel_index].z);
            myfile << color.x << " " << color.y << " " << color.z << "\n";
        }
    }
    myfile.close();
    checkCudaErrors(cudaFree(fb));

    return 0;
}
