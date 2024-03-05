#include <stdio.h>
#include <iostream>
#include <float.h>
#include <fstream>
#include "ray.h"
#include "helper_math.h"
#include "hitable_list.h"
#include "sphere.h"
#include "hitable.h"

__global__ void helloCUDA()
{
    printf("Hello, CUDA!\n");
}

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}



__device__ float3 color(const ray& r, hitable **world) {
    hit_record rec;
    if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
        return 0.5f*make_float3(rec.normal.x+1, rec.normal.y+1, rec.normal.z+1);
    }
   float3 unit_direction = normalize(r.direction()); //create a unit vector
   float t = 0.5f*(unit_direction.y + 1.0f);
   return (1.0f-t)*make_float3(1.0, 1.0, 1.0) + t*make_float3(0.5, 0.7, 1.0);
}

__global__ void render_gradient(float3 *fb, int max_x, int max_y) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    fb[pixel_index] = make_float3(float(i)/max_x, float(j)/max_y, 0.2);
}

__global__ void render(float3 *fb, int max_x, int max_y, float3 lower_left_corner, float3 horizontal, float3 vertical, float3 origin,hitable **world) {
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   int j = threadIdx.y + blockIdx.y * blockDim.y;
   if((i >= max_x) || (j >= max_y)) return;
   int pixel_index = j*max_x + i;
   float u = float(i) / float(max_x);
   float v = float(j) / float(max_y);
   float3 x = lower_left_corner + u*horizontal + v*vertical;
   ray r(origin, lower_left_corner + u*horizontal + v*vertical);
   fb[pixel_index] = color(r,world);
}
__global__ void create_world(hitable **d_list, hitable **d_world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_list[0] = new sphere(make_float3(0,0,-1), 0.5);
        d_list[1] = new sphere(make_float3(0,-100.5,-1), 100);
        *d_world = new hitable_list(d_list,2);
    }
}

__global__ void free_world(hitable **d_list, hitable **d_world) {
    delete ((hitable_list *)*d_world)->list[0];
    delete ((hitable_list *)*d_world)->list[1];
    delete *d_world;
    delete d_list;
}


int main()
{
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();
    int nx = 1600;
    int ny = 900;
    float aspect_ratio = float(nx) / float(ny);

    int num_pixels = nx*ny;
    size_t fb_size = num_pixels*sizeof(float3);

    // allocate FB
    float3 *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    int tx = 8;
    int ty = 8;

    // Render our buffer
    dim3 blocks(nx/tx+1,ny/ty+1);
    dim3 threads(tx,ty);
    //create_world
    hitable **d_list;
    checkCudaErrors(cudaMalloc((void **)&d_list, 2*sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
    create_world<<<1, 1>>>(d_list, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //render
    render<<<blocks, threads>>>(fb, nx, ny,
        make_float3(-8.0, -4.5, -1.0),
        make_float3(16.0, 0.0, 0.0),//cos zjebane to zalezy od aspect ratio
        make_float3(0.0, 9, 0.0),
        make_float3(0.0, 0.0, 0.0),
        d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    //free_world
    free_world<<<1, 1>>>(d_list, d_world);
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