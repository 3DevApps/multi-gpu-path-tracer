#pragma once
#include "ray.h"
#include "hitable.h"
#include "curand_kernel.h"
#include "material.h"
#include "interval.h"
#include "bvh.h"
#include "HostScene.h"
#include "CameraParams.h"

/**
* @brief Represents a camera in a 3D scene.
*
* The Camera class provides functionality to define the position, orientation, and field of view of a camera in a 3D scene.
* It also allows for generating rays from the camera's position to specific points on the image plane.
*/
class camera
{
public:
    __device__ void recalculate_camera_params(CameraParams& camParams) {
        lookAt = camParams.lookFrom + camParams.front;
        focal_length = length(camParams.lookFrom - lookAt);
        float theta_v = camParams.vfov * M_PI / 180; 
        float half_height = tan(theta_v / 2);
        float theta_h = camParams.hfov * M_PI / 180;
        float half_width = tan(theta_h / 2); 
        origin = camParams.lookFrom; 
        w = normalize(camParams.lookFrom - lookAt);
        u = normalize(cross(vup, w));
        v = cross(w, u);

        lower_left_corner = origin - half_width*u - half_height*v - w;
        horizontal = 2*half_width*u;
        vertical = 2*half_height*v;   
    }
    
    /**
     * @brief Calculates the color of a ray.
     *
     * This function calculates the color of a ray by tracing it through the scene and
     * computing the interactions with the objects in the scene.
     *
     * @param r The ray to calculate the color for.
     * @param world The pointer to the array of hitable objects in the scene.
     * @param local_rand_state The pointer to the random number generator state for the current thread.
     * @return The color of the ray.
     */
    __device__ float3 ray_color(const ray& r, hitable_list **world, CameraParams& camParams, unsigned int recursionDepth, curandState* local_rand_state) {
        recalculate_camera_params(camParams);
        ray cur_ray = r;
        float3 cur_attenuation = make_float3(1.0, 1.0, 1.0);
        for(int i = 0; i < recursionDepth; i++) {
            hit_record rec;
            if ((*world)->hit(cur_ray, hit_interval, rec)) {
                ray scattered;
                float3 attenuation; //means color
                float3 color_from_emission = rec.mat_ptr->emitted();
                if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                    cur_attenuation *= attenuation + color_from_emission;
                    cur_ray = scattered;
                }
                else {
                    cur_attenuation *= color_from_emission;
                    return cur_attenuation;
                }
            }
            else {
                //background
                return make_float3(1, 1, 1) * cur_attenuation; 
                }
            }
        return make_float3(0.0,0.0,0.0); // exceeded recursion
    }   

    /**
     * @brief Calculates the ray for a given pixel coordinate.
     *
     * This function calculates the ray originating from the camera's position and passing through
     * the specified pixel coordinate on the image plane.
     *
     * @param x The x-coordinate of the pixel (ranging from 0 to 1).
     * @param y The y-coordinate of the pixel (ranging from 0 to 1).
     * @return The ray passing through the specified pixel coordinate.
     */
    __device__ ray get_ray(float u, float v) {
        return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
    }

private:
    const interval hit_interval = interval(0.001f, FLT_MAX);
    float3 origin; //camera position
    float3 lower_left_corner; //lower left corner of the viewport
    float3 horizontal; //viewport dimensions
    float3 vertical; //viewport dimensions
    unsigned int recursionDepth_;
    float focal_length; //distance between the camera and the viewport
    float verticalFieldOfView_ = 45.0; //vertical field of view
    float horizontalFieldOfView_ = 45.0;
    float3 background_color = make_float3(0.0, 0.0, 0.0);
    float3 lookAt = make_float3(1.0, 0, 0);
    float3 vup = make_float3(0.0, 1.0, 0.0);
    float3 u, v, w; //orthonormal basis for the camera
};