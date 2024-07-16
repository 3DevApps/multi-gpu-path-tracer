
#pragma once

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

struct RenderTask {
    int width;
    int height;
    int offset_x;
    int offset_y;
};

struct Scene {
    hitable **d_list;
    hitable_list **d_world;
    camera **d_camera;
};

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
__global__ void render(uint8_t *fb, RenderTask task, int max_x, int max_y, int sample_per_pixel, camera **cam,hitable_list **world, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= task.width) || (j >= task.height)) return;
    int pixel_index = (task.offset_y + j) * max_x + (task.offset_x + i);
    curandState local_rand_state = rand_state[pixel_index];
    //Antialiasing
    float3 col = make_float3(0, 0, 0);
    for (int s=0; s<sample_per_pixel; s++) {
        float u = float(task.offset_x + i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(task.offset_y + j + curand_uniform(&local_rand_state)) / float(max_y);
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
__global__ void create_world(hitable_list **d_world, camera **d_camera, hitable **d_list, int d_list_size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {                       
        *d_world  = new hitable_list(d_list, d_list_size);
        *d_camera = new camera();
    }
}

__global__ void free_world(hitable **d_list, hitable_list **d_world, camera **d_camera, int d_list_size) {
    for (int i=0; i < d_list_size; i++) {
        delete d_list[i];
    }

    delete *d_world;
    delete *d_camera;    
}

__global__ void setCameraFront(camera ** cam, float3 front) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {   
        (*cam)->set_camera_front(front);
    }
}

__global__ void setCameraLookFrom(camera ** cam, float3 lookFrom) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {   
        (*cam)->set_camera_look_from(lookFrom);
    }
}

class DevicePathTracer {
public:
    DevicePathTracer(int device_idx, obj_loader &loader, int view_width, int view_height) 
        : device_idx_{device_idx}, view_width_{view_width}, view_height_{view_height} {
        cudaSetDevice(device_idx);
        number_of_faces_ = loader.get_total_number_of_faces();
        int num_pixels = view_width_ * view_height_; // 

        checkCudaErrors(cudaMalloc((void **)&scene_.d_list, number_of_faces_ * sizeof(hitable *)));
        loader.load_faces(scene_.d_list);
        checkCudaErrors(cudaMalloc((void **)&scene_.d_world, sizeof(hitable_list *)));
        checkCudaErrors(cudaMalloc((void **)&scene_.d_camera, sizeof(camera *)));

        checkCudaErrors(cudaMalloc((void **)&d_rand_state_, num_pixels*sizeof(curandState))); //

        create_world<<<1,1>>>(scene_.d_world, scene_.d_camera, scene_.d_list, number_of_faces_);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        dim3 blocks(view_width_ / THREADS_PER_DIM_X_ + 1, view_height / THREADS_PER_DIM_Y_ + 1);
        dim3 threads(THREADS_PER_DIM_X_, THREADS_PER_DIM_Y_);

        render_init<<<blocks, threads>>>(view_width, view_height, d_rand_state_);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void renderTaskAsync(RenderTask &task, uint8_t *fb) {
        dim3 blocks(task.width / THREADS_PER_DIM_X_ + 1, task.height / THREADS_PER_DIM_Y_ + 1);
        dim3 threads(THREADS_PER_DIM_X_, THREADS_PER_DIM_Y_);

        cudaSetDevice(device_idx_);
        render<<<blocks, threads>>>(
            fb, task, 
            view_width_, view_height_,
            3, scene_.d_camera,
            scene_.d_world,
            d_rand_state_
        );
    }

    void waitForRenderTask() {
        cudaSetDevice(device_idx_);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void setFront(float3 front) {
        cudaSetDevice(device_idx_);
        setCameraFront<<<1,1>>>(scene_.d_camera, front);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void setLookFrom(float3 lookFrom) {
        cudaSetDevice(device_idx_);
        setCameraLookFrom<<<1,1>>>(scene_.d_camera, lookFrom);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    ~DevicePathTracer() {
        cudaSetDevice(device_idx_);
        free_world<<<1, 1>>>(scene_.d_list, scene_.d_world, scene_.d_camera, number_of_faces_);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

private:
    int device_idx_;
    int view_width_;
    int view_height_;
    dim3 blocks_;
    dim3 threads_;
    curandState *d_rand_state_;
    int number_of_faces_;
    const int THREADS_PER_DIM_X_ = 8;
    const int THREADS_PER_DIM_Y_ = 8;
    Scene scene_{};
};


