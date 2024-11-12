
#pragma once

#include <curand_kernel.h>
#include "ray.h"
#include "helper_math.h"
#include "hitable_list.h"
#include "sphere.h"
#include "hitable.h"
#include "camera.h"
#include "material.h"
#include "triangle.h"
#include "cuda_utils.h"
#include "bvh.h"
#include "RendererConfig.h"
#include "Framebuffer.h"
#include <thrust/device_vector.h>

struct RenderTask {
    int width;
    int height;
    int offset_x;
    int offset_y;
    int time = 0;
};

struct Scene {
    BVH **d_world = nullptr;
    camera **d_camera = nullptr;
    thrust::device_vector<BaseColorTexture> textures{}; 
    thrust::device_vector<UniversalMaterial> materials{};
    thrust::device_vector<triangle> faces{}; 
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
__global__ void render(uint8_t *fb, RenderTask task, Resolution res, int sample_per_pixel, camera **cam, BVH **world, CameraConfig cameraConfig, unsigned int recursionDepth, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= task.width) || (j >= task.height)) return;
    int pixel_index = (task.offset_y + j) * res.width + (task.offset_x + i);
    curandState local_rand_state = rand_state[pixel_index];
    //Antialiasing
    float3 col = make_float3(0, 0, 0);
    for (int s=0; s<sample_per_pixel; s++) {
        float u = float(task.offset_x + i + curand_uniform(&local_rand_state)) / float(res.width);
        float v = float(task.offset_y + j + curand_uniform(&local_rand_state)) / float(res.height);
        ray r = (*cam)->get_ray(u, v);
        col += (*cam)->ray_color(r, world, cameraConfig, recursionDepth, &local_rand_state);
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
__global__ void create_world(BVH **d_world, triangle*d_list, int d_list_size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {                                  
        *d_world  = new BVH(d_list, d_list_size);      
    }
}

__global__ void create_camera(camera **d_camera) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {                       
        *d_camera = new camera();
    }
}

__global__ void free_world(BVH **d_world) {
    if (*d_world) {
        delete *d_world; 
        *d_world = nullptr;
    }
}

class DevicePathTracer {
public:
    DevicePathTracer(
            int device_idx, 
            unsigned int samplesPerPixel,
            unsigned int recursionDepth,
            dim3 threadBlockSize,
            HostScene& hostScene,
            std::shared_ptr<Framebuffer> framebuffer,
            CameraConfig& cameraConfig) : 
            device_idx_{device_idx},
            samplesPerPixel_{samplesPerPixel},
            recursionDepth_{recursionDepth},
            hostScene_{hostScene},
            threadBlockSize_{threadBlockSize}, 
            framebuffer_{framebuffer}, 
            cameraConfig_{cameraConfig} {
        
        cudaSetDevice(device_idx_);
        checkCudaErrors(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 2000000000)); 
        checkCudaErrors(cudaDeviceSetLimit(cudaLimitStackSize, 10000)); 

        reloadWorld();
        reloadCamera();
        setFramebuffer(framebuffer_);
    }

    void renderTaskAsync(RenderTask &task, cudaStream_t stream) {
        if (task.width == 0)
            return;
        
        dim3 blocks(task.width / threadBlockSize_.x + 1, task.height / threadBlockSize_.y + 1);

        cudaSetDevice(device_idx_);
        render<<<blocks, threadBlockSize_, 0, stream>>>(
            framebuffer_->getPtr(), 
            task, 
            framebuffer_->getResolution(),
            samplesPerPixel_,
            scene_.d_camera,
            scene_.d_world,
            cameraConfig_,
            recursionDepth_,
            d_rand_state_
        );
    }

    void waitForRenderTask() {
        cudaSetDevice(device_idx_);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void synchronizeStream(cudaStream_t stream) {
        cudaSetDevice(device_idx_);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaStreamSynchronize(stream));
    }


    // to be called when camera parameters change
    void reloadCamera() {
        cudaSetDevice(device_idx_);
        if (scene_.d_camera == nullptr) {
            checkCudaErrors(cudaMalloc((void **)&scene_.d_camera, sizeof(camera *)));
        } 

        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        create_camera<<<1,1>>>(scene_.d_camera);
    }

    void loadTextures() {
        scene_.textures = {};
        for (const auto& texture : hostScene_.textures) {
            float3* d_tex;
            checkCudaErrors(cudaMalloc(
                (void **)&d_tex, 
                texture.width * texture.height * sizeof(float3)
            ));

            cudaMemcpy(
                d_tex, 
                texture.data.data(), 
                texture.width * texture.height * sizeof(float3), 
                cudaMemcpyHostToDevice
            );

            BaseColorTexture tex(
                texture.width,
                texture.height,
                d_tex
            );

            scene_.textures.push_back(tex);
        }
    }

    void loadMaterials() {
        scene_.materials = {};
        BaseColorTexture* baseColorTexture = nullptr;
        BaseColorTexture* emissiveTexture = nullptr;
        for (const auto& material : hostScene_.materials) {
            // int textureIdx = material.baseColorTextureIdx;
            if (material.baseColorTextureIdx.has_value()) {
                baseColorTexture = thrust::raw_pointer_cast(&scene_.textures[material.baseColorTextureIdx.value()]);
            }

            if (material.emissiveTextureIdx.has_value()) {
                emissiveTexture = thrust::raw_pointer_cast(&scene_.textures[material.emissiveTextureIdx.value()]);
            }

            UniversalMaterial mat(
                material.baseColor,
                // make_float3(0, 0, 0),
                baseColorTexture,
                material.emissiveFactor,
                emissiveTexture

            );
            scene_.materials.push_back(mat);
        }
    }

    void loadTrianglesWithTextures() {
        scene_.faces = {};
        for (const auto& hTriangle : hostScene_.triangles) {
            triangle t(
                hTriangle.v0,
                hTriangle.v1,
                hTriangle.v2,
                thrust::raw_pointer_cast(&scene_.materials[hTriangle.materialIdx])
            ); 
            scene_.faces.push_back(t);
        }
    }

    // To be called when scene triangles change
    void reloadWorld() {
        cudaSetDevice(device_idx_);

        if (scene_.d_world != nullptr) {
            // free previouse device pointer
            free_world<<<1, 1>>>(scene_.d_world);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());

            checkCudaErrors(cudaFree(scene_.d_world));
            scene_.d_world = nullptr;
        }

        checkCudaErrors(cudaMalloc((void **)&scene_.d_world, sizeof(BVH *)));

        loadTextures();
        loadMaterials();
        loadTrianglesWithTextures();

        create_world<<<1,1>>>(scene_.d_world, thrust::raw_pointer_cast(&scene_.faces[0]), scene_.faces.size());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void setFramebuffer(std::shared_ptr<Framebuffer> framebuffer) {
        framebuffer_ = framebuffer;

        cudaSetDevice(device_idx_);
        int num_pixels = framebuffer_->getPixelCount();
        if (d_rand_state_) {
            checkCudaErrors(cudaFree(d_rand_state_));
            d_rand_state_ = nullptr;
        }  

        checkCudaErrors(cudaMalloc((void **)&d_rand_state_, num_pixels * sizeof(curandState)));
        dim3 blocks(framebuffer_->getResolution().width / threadBlockSize_.x + 1, framebuffer_->getResolution().height / threadBlockSize_.y + 1);

        render_init<<<blocks, threadBlockSize_>>>(framebuffer_->getResolution().width, framebuffer_->getResolution().height, d_rand_state_);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    void setSamplesPerPixel(unsigned int samplesPerPixel) {
        samplesPerPixel_ = samplesPerPixel;
    }

    void setRecursionDepth(unsigned int recursionDepth) {
        recursionDepth_ = recursionDepth;
    }

    void setThreadBlockSize(dim3 threadBlockSize) {
        threadBlockSize_ = threadBlockSize;
    }

    ~DevicePathTracer() {
        cudaSetDevice(device_idx_);
        free_world<<<1, 1>>>(scene_.d_world);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

private:
    int device_idx_;
    dim3 blocks_;
    dim3 threads_;
    curandState *d_rand_state_ = nullptr;
    int number_of_faces_;
    dim3 threadBlockSize_;
    Scene scene_{};   
    HostScene& hostScene_;
    unsigned int samplesPerPixel_;
    unsigned int recursionDepth_;
    std::shared_ptr<Framebuffer> framebuffer_;
    CameraConfig& cameraConfig_;
};


