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
#include "object3d.h"
#include "triangle.h"

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA ERROR = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
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
__global__ void render(float3 *fb, int max_x, int max_y,int sample_per_pixel, camera **cam,hitable **world, curandState *rand_state) {
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
 * @param objects Array of objects loaded from .obj file
 * @param number_of_meshes Number of objects in objects array 
 */
__global__ void create_world(hitable **d_list, hitable **d_world,camera **d_camera, object3d **objects, int number_of_meshes) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // TODO: Set the materials accordingly to the object
        // For example create a property on object3d and an enum for the material type
        // We can introduce a custom texture loader to load materials for the objects (bounded to our defined materials)
        material *mat = new lambertian(make_float3(0.5, 0.5, 0.5));
        // material *mat = new metal(make_float3(0.8, 0.6, 0.2), 0.0);
        // material *mat = new dielectric(1.5);
        
        int face_counter = 0;

        for (int i = 0; i < number_of_meshes; i++) {
            for (int j = 0; j < objects[i]->num_triangles; j++) {
                d_list[face_counter] = new triangle(objects[i]->triangles[j].v0, objects[i]->triangles[j].v1, objects[i]->triangles[j].v2, mat);
                face_counter++;
            }
        }
                       
        *d_world  = new hitable_list(d_list, face_counter);
        *d_camera = new camera();
    }
}

__global__ void free_objects(object3d **objects, int num_objects) {
    for (int i = 0; i < num_objects; i++) {
        delete objects[i];
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world,camera **d_camera, int num_meshes) {
    for (int i=0; i < num_meshes; i++) {
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

    // Load object
    const char *file_path = "models/cube.obj";
    obj_loader loader(file_path);

    int number_of_meshes = loader.get_number_of_meshes();
    int *faces_per_mesh = new int[number_of_meshes];
    loader.get_number_of_faces(faces_per_mesh);

    object3d **objects;
    checkCudaErrors(cudaMallocManaged((void **)&objects, number_of_meshes * sizeof(object3d)));

    int faces_total = 0;

    for (int i = 0; i < number_of_meshes; i++) {
        object3d *object;
        checkCudaErrors(cudaMallocManaged((void **)&object, sizeof(object3d)));

        triangle *triangles;
        checkCudaErrors(cudaMallocManaged((void **)&triangles, faces_per_mesh[i] * sizeof(triangle)));
        object->triangles = triangles;

        objects[i] = object;

        faces_total += faces_per_mesh[i];
    }

    loader.load(objects);

    hitable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, faces_total * sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

    create_world<<<1,1>>>(d_list,d_world,d_camera, objects, number_of_meshes);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    free_objects<<<1,1>>>(objects, number_of_meshes);
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

    free_world<<<1, 1>>>(d_list, d_world,d_camera, faces_total);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Save result to a PPM image
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
