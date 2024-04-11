#pragma once
#include "hitable.h"
#include "helper_math.h"
#include "ray.h"
#include "triangle.h"

class object3d: public hitable
{
    public:
        __device__ object3d() {};
        __device__ object3d(triangle *triangles, int size) : triangles(triangles), num_triangles(size) {};
        __device__ object3d(triangle *triangles, int size, material *mat_ptr);
        __device__ object3d(material *mat_ptr) : mat_ptr(mat_ptr) {};
        __device__ ~object3d() {
            delete triangles;
            delete mat_ptr;
        };
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;

        material *mat_ptr;
        triangle *triangles;
        int num_triangles;
};

__device__ object3d::object3d(triangle *triangles, int size, material *mat_ptr){
    for (int i = 0; i < size; i++)
    {
        triangles[i].mat_ptr = mat_ptr;
    }

    this->triangles = triangles;
    this->num_triangles = size;
    this->mat_ptr = mat_ptr;
}


__device__ bool object3d::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    for (int i = 0; i < num_triangles; i++)
    {
        if (triangles[i].hit(r, t_min, closest_so_far, temp_rec))
        {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}