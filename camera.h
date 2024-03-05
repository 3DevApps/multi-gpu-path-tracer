#pragma once
#include "ray.h"
#include "hitable.h"
#include "curand_kernel.h"
class camera
{
public:
    float3 origin;
    float3 lower_left_corner;
    float3 horizontal;
    float3 vertical;
    int image_width;
    int image_height;
    float aspect_ratio;


    __device__ camera() {
        image_width = 1600;
        image_height = 900;
        aspect_ratio = float(image_width) / float(image_height);
        
        lower_left_corner = make_float3(-8.0, -4.5, -1.0);
        horizontal = make_float3(16.0, 0.0, 0.0);
        vertical = make_float3(0.0, 9.0, 0.0);
        origin = make_float3(0.0, 0.0, 0.0);
    }
    __device__ float3 ray_color(const ray& r, hitable **world, curandState *local_rand_state) {
        ray cur_ray = r;
        float cur_attenuation = 1.0f;
        for(int i = 0; i < 50; i++) {
            hit_record rec;
            if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
                float3 direction = rec.normal + random_in_unit_sphere(local_rand_state);
                cur_attenuation *= 0.5f;
                cur_ray = ray(rec.p, direction);
            }
            else {
                float3 unit_direction = normalize(cur_ray.direction());
                float t = 0.5f*(unit_direction.y + 1.0f);
                float3 c = (1.0f-t)*make_float3(1.0, 1.0, 1.0) + t*make_float3(0.5, 0.7, 1.0);
                return cur_attenuation * c;
                }
            }
        return make_float3(0.0,0.0,0.0); // exceeded recursion
    }   
    __device__ ray get_ray(float u, float v) {
        return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin);
    }
};