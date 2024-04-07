#pragma once
#include "helper_math.h"
#include "ray.h"
#include "triangle.h"
#include "object3d.h"

class host_object3d
{
    public:
        host_object3d() {};
        void print();
        void set_triangles(triangle *triangles, int num_triangles);
        __device__ object3d clone() const;
        triangle *triangles;
        int num_triangles;
};

void host_object3d::set_triangles(triangle *triangles, int num_triangles)
{
    this->triangles = triangles;
    this->num_triangles = num_triangles;
}


__device__ object3d host_object3d::clone() const
{
    object3d obj;
    // Copy triangles
    obj.triangles = triangles;
    obj.num_triangles = num_triangles;

    return obj;
}