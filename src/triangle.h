#pragma once

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "helper_math.h"
#include <unordered_map>
#include <vector>
#include "hitable.h"

class triangle: public hitable
{
public:
    triangle() {};
    triangle(float3 v0, float3 v1, float3 v2) : v0(v0), v1(v1), v2(v2) {};
    __device__ triangle(float3 v0, float3 v1, float3 v2, material *mat_ptr) : v0(v0), v1(v1), v2(v2), mat_ptr(mat_ptr) {};
    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;

    float3 v0, v1, v2;
    material *mat_ptr;
};

__device__ bool triangle::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 pvec = cross(r.direction(), e2);
    float det = dot(e1, pvec);
    if (det < 1e-8 && det > -1e-8)
        return false;
    float inv_det = 1.0 / det;
    float3 tvec = r.origin() - v0;
    float u = dot(tvec, pvec) * inv_det;
    if (u < 0 || u > 1)
        return false;
    float3 qvec = cross(tvec, e1);
    float v = dot(r.direction(), qvec) * inv_det;
    if (v < 0 || u + v > 1)
        return false;
    float t = dot(e2, qvec) * inv_det;
    if (t < t_max && t > t_min)
    {
        rec.t = t;
        rec.p = r.point_at_parameter(rec.t);
        rec.normal = normalize(cross(e1, e2));
        rec.mat_ptr = mat_ptr;
        return true;
    }

    return false;
}