#pragma once
#include "hitable.h"
#include "helper_math.h"
#include "ray.h"
#include "aabb.h"
#include "interval.h"

class sphere: public hitable
{
    public:
        __device__ sphere() {}
        __device__ sphere(float3 cen, float r,material *m) : center(cen), radius(r), mat_ptr(m){
            float3 rad = make_float3(radius, radius, radius);
            bbox = aabb(center - rad, center + rad);
        };
        __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const;
        float3 center;
        float radius;
        material *mat_ptr;
};
__device__ bool sphere::hit(const ray& r,interval ray_t, hit_record& rec) const
{
    float3 oc = r.origin() - center;
    float a = dot(r.direction(), r.direction());
    float b = 2.0 * dot(oc, r.direction());
    float c = dot(oc, oc) - radius*radius;
    float discriminant = b*b - 4*a*c;
    if (discriminant > 0)
    {
        float temp = (-b - sqrt(discriminant)) / (2.0*a);
        if (temp < ray_t.max && temp > ray_t.min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / (2.0*a);
        if (temp < ray_t.max && temp > ray_t.min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}
