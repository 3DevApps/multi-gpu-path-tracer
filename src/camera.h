#pragma once
#include "ray.h"
#include "hitable.h"
#include "curand_kernel.h"
#include "material.h"
#include "interval.h"
#include "bvh.h"
class camera
{
public:
    //TODO: some of them should be private i just dont like the ideology of private variables so u can decide
    const interval hit_interval = interval(0.001f, FLT_MAX);

    float3 origin; //camera position
    float3 lower_left_corner; //lower left corner of the viewport
    float3 horizontal; //viewport dimensions
    float3 vertical; //viewport dimensions
    int image_width = 600;
    int image_height = 600;
    float aspect_ratio; //width/height
    int max_depth = 3; //Maximum nuber of bounces
    float focal_length; //distance between the camera and the viewport
    float vfov = 90.0; //vertical field of view

    float3 background_color = make_float3(0.0, 0.0, 0.0);

    float3 lookfrom = make_float3(-4.0, 2.0, 8.0);
    // float3 lookfrom = make_float3(-0.278, -0.8, 0.273);
    // float3 lookat = make_float3(-1.0, -2.0, -1.0);
    float3 lookat = make_float3(1.0, 0, 0);
    float3 vup = make_float3(0.0, 1.0, 0.0);
    
    float3 u, v, w; //orthonormal basis for the camera

    __device__ void set_camera_lookat(float3 lat) {
        lookat = lookfrom + lat;
        aspect_ratio = float(image_width) / float(image_height);
        focal_length = length(lookfrom - lookat);
        //viewport dimension calculation
        float theta = vfov * M_PI / 180; //deegres to radians
        float half_height = tan(theta/2);
        float half_width = aspect_ratio * half_height;
        origin = lookfrom;
        // lower_left_corner = origin - horizontal/2 - vertical/2 - make_float3(0.0, 0.0, focal_length);  
        w = normalize(lookfrom - lookat);
        u = normalize(cross(vup, w));
        v = cross(w, u);

        lower_left_corner = origin - half_width*u - half_height*v - w;
        horizontal = 2*half_width*u;
        vertical = 2*half_height*v;   
    }

    /**
     * @brief Represents a camera in a 3D scene.
     *
     * The Camera class provides functionality to define the position, orientation, and field of view of a camera in a 3D scene.
     * It also allows for generating rays from the camera's position to specific points on the image plane.
     */
    __device__ camera() {
        aspect_ratio = float(image_width) / float(image_height);
        focal_length = length(lookfrom - lookat);
        //viewport dimension calculation
        float theta = vfov * M_PI / 180; //deegres to radians
        float half_height = tan(theta/2);
        float half_width = aspect_ratio * half_height;
        origin = lookfrom;
        // lower_left_corner = origin - horizontal/2 - vertical/2 - make_float3(0.0, 0.0, focal_length);  
        w = normalize(lookfrom - lookat);
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
    __device__ float3 ray_color(const ray& r, hitable_list **world, curandState *local_rand_state) {
        ray cur_ray = r;
        float3 cur_attenuation = make_float3(1.0, 1.0, 1.0);
        for(int i = 0; i < max_depth; i++) {
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
                    
                    // return make_float3(0.0,0.0,0.0);
                }
            }
            else {
                //background
                float3 unit_direction = normalize(cur_ray.direction());
                float t = 0.5f*(unit_direction.y + 1.0f);
                float3 c = (1.0f-t)*make_float3(1.0, 1.0, 1.0) + t*make_float3(0.5, 0.7, 1.0);
                return cur_attenuation * c;
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
};