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

//render_init doesnt have to be separate kernel, dona that way for clarity 
//better performance to do it in the render kernel
__global__ void render_init(int nx, int ny, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= nx) || (j >= ny)) return;
    int pixel_index = j*nx + i;
    //Each thread gets diffrent seed, same sequence number, no offset
    curand_init(1984+pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(float3 *fb, int max_x, int max_y,int sample_per_pixel, camera **cam,hitable **world, curandState *rand_state) {
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y;
   if((i >= max_x) || (j >= max_y)) return;
   int pixel_index = j*max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    float3 col = make_float3(0, 0, 0);
    for (int s=0; s<sample_per_pixel; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v);
        col += (*cam)->ray_color(r, world, &local_rand_state);
    }

   fb[pixel_index] = col/float(sample_per_pixel);
}

__global__ void create_world(hitable **d_list, hitable **d_world,camera **d_camera, object3d **objects, int num_objects, int num_meshes) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        material *mat = new metal(make_float3(0.8, 0.6, 0.2),0.0);

        int g_i = 0;

        for (int i = 0; i < num_objects; i++) {
            for (int j = 0; j < objects[i]->num_triangles; j++) {
                d_list[g_i] = new triangle(objects[i]->triangles[j].v0, objects[i]->triangles[j].v1, objects[i]->triangles[j].v2, mat);
                g_i++;
            }
        }
                       
        *d_world  = new hitable_list(d_list, num_meshes);
        *d_camera = new camera();
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world,camera **d_camera, object3d **objects, int num_objects, int num_meshes) {
    for(int i=0; i < num_meshes; i++) {
        delete d_list[i];
    }

    for (int i = 0; i < num_objects; i++) {
        delete objects[i];
    }

    delete *d_world;
    delete *d_camera;
    
}

int main()
{
    int nx = 1600;
    int ny = 900;
    // int nx = 1200;
    // int ny = 600;
    
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
    const char *obj = "models/cubes.obj";

    obj_loader loader;
    int mesh_no = loader.get_number_of_meshes(obj);

    int *faces_per_mesh = new int[mesh_no];
    loader.get_number_of_faces(obj, faces_per_mesh);

    object3d **objects;
    checkCudaErrors(cudaMallocManaged((void **)&objects, mesh_no * sizeof(object3d)));

    int meshes_total = 0;

    for (int i = 0; i < mesh_no; i++) {
        object3d *object;
        checkCudaErrors(cudaMallocManaged((void **)&object, sizeof(object3d)));

        triangle *triangles;
        checkCudaErrors(cudaMallocManaged((void **)&triangles, faces_per_mesh[i] * sizeof(triangle)));
        object->triangles = triangles;

        objects[i] = object;

        meshes_total += faces_per_mesh[i];
    }


    loader.load(obj, objects);


    hitable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, meshes_total * sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

    create_world<<<1,1>>>(d_list,d_world,d_camera, objects, mesh_no, meshes_total);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //render
    render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(fb, nx, ny,
        100, d_camera,
        d_world,
        d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    //free_world
    free_world<<<1, 1>>>(d_list, d_world,d_camera, objects, mesh_no, meshes_total);
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
