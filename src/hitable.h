#pragma once
#include "ray.h"
#include "aabb.h"
#include "interval.h"

class material;

struct hit_record{
    float t;
    float3 p;
    float3 normal;
    material *mat_ptr;
};



class hitable{
    public:
        __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const = 0;
        aabb bbox; //bounding box
        __device__ void debug_bbox(){
            printf("bbox min: %f %f %f\n", bbox.x.min, bbox.y.min, bbox.z.min);
            printf("bbox max: %f %f %f\n", bbox.x.max, bbox.y.max, bbox.z.max);
        }
};
