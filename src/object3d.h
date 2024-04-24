#pragma once
#include "hitable.h"
#include "helper_math.h"
#include "ray.h"
#include "triangle.h"

class object3d: public hitable
{
    public:
        __device__ object3d() {};
        __device__ object3d(triangle *triangles, int size) : triangles(triangles), num_triangles(size) {
            //needs aabb definition but writing it is a waste of time cus we dont even put this thing in BVH
            //we dont even call hit on this class idk what is the point of all of this
        };
        __device__ object3d(triangle *triangles, int size, material *mat_ptr);
        __device__ object3d(material *mat_ptr) : mat_ptr(mat_ptr) {};
        __device__ ~object3d() {
            delete triangles;
            delete mat_ptr;
        };
        __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const;

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


__device__ bool object3d::hit(const ray& r, interval ray_t, hit_record& rec) const
{
    hit_record temp_rec;
    bool hit_anything = false;

    for (int i = 0; i < num_triangles; i++)
    {
        if (triangles[i].hit(r, ray_t, temp_rec))
        {
            hit_anything = true;
            ray_t.max = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}