#pragma once
#include "ray.h"

class material;

struct hit_record{
    float t;
    float3 p;
    float3 normal;
    material *mat_ptr;
};

class hitable{
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;

        __device__ virtual void print_gpu() const {
             printf("debug value\n");
        }

};


