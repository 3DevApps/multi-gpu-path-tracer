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

__global__ void create_world(hitable **d_list, hitable **d_world,camera **d_camera, triangle *triangles, int num_triangles) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Creating world %d\n", num_triangles);
        
        for (int i = 0; i < num_triangles; i++) {
            d_list[i] = new triangle(triangles[i].v0, triangles[i].v1, triangles[i].v2, new metal(make_float3(0.8, 0.6, 0.2),0.0));
        }

        // d_list[0] = new object3d(triangles, num_triangles, new metal(make_float3(0.8, 0.6, 0.2),0.0));

        // obj->set_material(new metal(make_float3(0.8, 0.6, 0.2),0.0));

        // object3d obj1 = obj->clone();
        // obj1.set_material(new metal(make_float3(0.8, 0.6, 0.2),0.0));

        // d_list[0] = &obj1;
        // d_list[1] = &obj1;
        // d_list[2] = &obj1;
        // d_list[3] = &obj1;
        // d_list[4] = &obj1;

        // d_list[0] = new object3d(obj->triangles, obj->num_triangles, obj->mat_ptr);
        // d_list[1] = obj->clone();
        // d_list[2] = obj->clone();
        // d_list[3] = obj->clone();
        // d_list[4] = obj->clone();

        // printf("num_triangles: %d\n", obj->num_triangles);

        // printf("triangle %f\n", obj->triangles[15].v0.x);
        // d_list[0] = new object3d(obj->triangles, obj->num_triangles, new metal(make_float3(0.8, 0.6, 0.2),0.0));
        // d_list[1] = new object3d(obj->triangles, obj->num_triangles, new metal(make_float3(0.8, 0.6, 0.2),0.0));
        // d_list[2] = new object3d(obj->triangles, obj->num_triangles, new metal(make_float3(0.8, 0.6, 0.2),0.0));
        // d_list[3] = new object3d(obj->triangles, obj->num_triangles, new metal(make_float3(0.8, 0.6, 0.2),0.0));
        // d_list[4] = new object3d(obj->triangles, obj->num_triangles, new metal(make_float3(0.8, 0.6, 0.2),0.0));


        // d_list[0]->print_gpu();


        // d_list[1] = new object3d(new metal(make_float3(0.8, 0.6, 0.2),0.0));
        // d_list[2] = new object3d( new metal(make_float3(0.8, 0.6, 0.2),0.0));
        // d_list[3] = new object3d( new metal(make_float3(0.8, 0.6, 0.2),0.0));
        // d_list[4] = new object3d( new metal(make_float3(0.8, 0.6, 0.2),0.0));

        // float3 v0 = make_float3(1,0,-1);
        // float3 v1 = make_float3(2,0,-1);
        // float3 v2 = make_float3(0,1,-1);

        // d_list[0] = new triangle(v0, v1, v2, new metal(make_float3(0.8, 0.6, 0.2),0.0));
        // d_list[1] = new sphere(make_float3(0,-100.5,-1), 100,
        //                        new lambertian(make_float3(0.8, 0.8, 0.0)));
        // d_list[2] = new sphere(make_float3(1,0,-1), 0.5,
        //                          new metal(make_float3(0.8, 0.6, 0.2),0.0));
        // d_list[3] = new sphere(make_float3(-1,0,-1), 0.5, //negative radius trick makes it face inwards
        //                             new dielectric(1.5f));
        // d_list[4] = new sphere(make_float3(-1,0,-1), -0.45,
        //                             new dielectric(1.5));
        // d_list[0] = new triangle(v0, v1, v2, new metal(make_float3(0.8, 0.6, 0.2),0.0));
        // d_list[2] = new triangle(v0, v1, v2, new metal(make_float3(0.8, 0.6, 0.2),0.0));
        // d_list[3] = new triangle(v0, v1, v2, new metal(make_float3(0.8, 0.6, 0.2),0.0));
        // d_list[4] = new triangle(v0, v1, v2, new metal(make_float3(0.8, 0.6, 0.2),0.0));


        // d_list[0]->print_gpu();

        // d_list[0] = new sphere(make_float3(0,-100.5,-1), 100,
                            //    new lambertian(make_float3(0.8, 0.8, 0.0)));
        // d_list[1] = new sphere(make_float3(0,-100.5,-1), 100,
        //                        new lambertian(make_float3(0.8, 0.8, 0.0)));
        // d_list[2] = new sphere(make_float3(1,0,-1), 0.5,
        //                        new metal(make_float3(0.8, 0.6, 0.2),0.0));
        // d_list[3] = new sphere(make_float3(-1,0,-1), 0.5, //negative radius trick makes it face inwards
        //                        new dielectric(1.5f));
        // d_list[4] = new sphere(make_float3(-1,0,-1), -0.45,
        //                          new dielectric(1.5));                    
        *d_world  = new hitable_list(d_list, num_triangles);
        *d_camera = new camera();
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world,camera **d_camera) {
    for(int i=0; i < 5; i++) {
        // delete ((sphere *)d_list[i])->mat_ptr;
        delete d_list[i];
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
    // TODO: this should be a parameter to the program
    object3d *obj;
    checkCudaErrors(cudaMallocManaged((void **)&obj, sizeof(object3d)));

    obj_loader loader;
    loader.load(obj, "models/cubes.obj");

    // Create array of triangles from object
    // TODO: this should be done in the obj_loader
    triangle *triangles;
    checkCudaErrors(cudaMallocManaged((void **)&triangles, obj->num_triangles * sizeof(triangle)));

    for (int i = 0; i < obj->num_triangles; i++) {
        triangles[i] = triangle(obj->triangles[i].v0, obj->triangles[i].v1, obj->triangles[i].v2);
    }

    hitable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, obj->num_triangles*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));

    
    // TODO: we should pass the object_3d and not triangles array
    create_world<<<1,1>>>(d_list,d_world,d_camera,triangles, obj->num_triangles);

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
