#pragma once

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "helper_math.h"
#include <unordered_map>
#include <vector>

class triangle
{
public:
    triangle() {};
    triangle(float3 v0, float3 v1, float3 v2);
    bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const;

    float3 v0, v1, v2;
    material *mat_ptr;
};

triangle::triangle(float3 v0, float3 v1, float3 v2)
{
    this->v0 = v0;
    this->v1 = v1;
    this->v2 = v2;

    // mat_ptr = new lambertian(make_float3(0.5f, 0.5f, 0.5f));
}

bool triangle::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    return false;
    // float3 e1 = v1 - v0;
    // float3 e2 = v2 - v0;
    // float3 p = cross(r.direction, e2);
    // float det = dot(e1, p);

    // if (det < 1e-8)
    //     return false;

    // float3 t = r.origin - v0;
    // float u = dot(t, p);
    // if (u < 0 || u > det)
    //     return false;

    // float3 q = cross(t, e1);
    // float v = dot(r.direction, q);
    // if (v < 0 || u + v > det)
    //     return false;

    // float inv_det = 1 / det;
    // float t_hit = dot(e2, q) * inv_det;
    // if (t_hit < t_min || t_hit > t_max)
    //     return false;

    // rec.t = t_hit;
    // rec.p = r.point_at_parameter(t_hit);
    // rec.normal = normalize(cross(e1, e2));
    // rec.mat_ptr = mat_ptr;

    // return true;
}