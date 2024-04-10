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

        void print();
        void set_triangles(triangle *triangles, int num_triangles);
        __device__ void set_material(material *mat_ptr);
        __device__ object3d *clone() const { return new object3d(*this); }
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;
        __device__ virtual void print_gpu() const;
        material *mat_ptr;
        triangle *triangles;
        int num_triangles;
};

__device__ object3d::object3d(triangle *triangles, int size, material *mat_ptr){
    // Set material to each triangle
    for (int i = 0; i < size; i++)
    {
        triangles[i].mat_ptr = mat_ptr;
    }

    this->triangles = triangles;
    this->num_triangles = size;
    this->mat_ptr = mat_ptr;
}


void object3d::set_triangles(triangle *triangles, int num_triangles)
{
    this->triangles = triangles;
    this->num_triangles = num_triangles;
}

__device__ void object3d::set_material(material *mat_ptr)
{
    this->mat_ptr = mat_ptr;
}

void object3d::print()
{
    printf("Object3d cpu %d\n", num_triangles);
}

__device__ void object3d::print_gpu() const
{
    printf("Object3d %d\n", num_triangles);
    // Print all triangles
    // for (int i = 0; i < num_triangles; i++)
    // {
    //     printf("Triangle %d\n", triangles[i].v0.x);
    // }
    
}

__device__ bool object3d::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    // print first triangle
    // triangles[0].hit(r, t_min, t_max, rec);

    // return false;

    // return triangles[0].hit(r, t_min, t_max, rec);
    // return false;
    // float radius = 0.5;
    // float3 center = make_float3(1, 0,-1);
    // float3 oc = r.origin() - center;
    // float a = dot(r.direction(), r.direction());
    // float b = 2.0 * dot(oc, r.direction());
    // float c = dot(oc, oc) - radius*radius;
    // float discriminant = b*b - 4*a*c;
    // if (discriminant > 0)
    // {
    //     float temp = (-b - sqrt(discriminant)) / (2.0*a);
    //     if (temp < t_max && temp > t_min)
    //     {
    //         rec.t = temp;
    //         rec.p = r.point_at_parameter(rec.t);
    //         rec.normal = (rec.p - center) / radius;
    //         rec.mat_ptr = mat_ptr;
    //         return true;
    //     }
    //     temp = (-b + sqrt(discriminant)) / (2.0*a);
    //     if (temp < t_max && temp > t_min)
    //     {
    //         rec.t = temp;
    //         rec.p = r.point_at_parameter(rec.t);
    //         rec.normal = (rec.p - center) / radius;
    //         rec.mat_ptr = mat_ptr;
    //         return true;
    //     }
    // }
    // return false;
    // Print if material is present
    // if (mat_ptr != NULL)
    // {
    //     printf("Material is present\n");
    // }

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