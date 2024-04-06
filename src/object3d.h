#pragma once
#include "hitable.h"
#include "helper_math.h"
#include "ray.h"
#include "triangle.h"

class object3d: public hitable
{
    public:
        object3d() {};
        object3d(triangle *triangles, int size);
        void print();
        void set_triangles(triangle *triangles, int num_triangles);
        __device__ void set_triangles_cuda(triangle *triangles, int num_triangles);
        __device__ object3d *clone() const { return new object3d(*this); }
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
        __device__ virtual void print_gpu() const;
        material *mat_ptr;
        triangle *triangles;
        int num_triangles;
};

object3d::object3d(triangle *triangles, int num_triangles)
{
    this->num_triangles = num_triangles;
    this->triangles = triangles;
}

void object3d::set_triangles(triangle *triangles, int num_triangles)
{
    this->triangles = triangles;
    this->num_triangles = num_triangles;
}

void object3d::print()
{
    printf("Object3d cpu %d\n", num_triangles);
}

__device__ void object3d::print_gpu() const
{
    printf("Object3d %d\n", num_triangles);
}

__device__ bool object3d::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    return false;
    // hit_record temp_rec;
    // bool hit_anything = false;
    // float closest_so_far = t_max;

    // for (int i = 0; i < triangles.size(); i++)
    // {
    //     if (triangles[i].hit(r, t_min, closest_so_far, temp_rec))
    //     {
    //         hit_anything = true;
    //         closest_so_far = temp_rec.t;
    //         rec = temp_rec;
    //     }
    // }

    // return hit_anything;
}