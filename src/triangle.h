#pragma once

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "helper_math.h"
#include "hitable.h"
#include "interval.h"
#include "aabb.h"
#include "material.h"
#include "HostScene.h"

// class triangle: public hitable
class triangle {
public:
    // __device__ triangle(Vertex v0, Vertex v1, Vertex v2, UniversalMaterial *mat_ptr) : v0(v0), v1(v1), v2(v2), mat_ptr(mat_ptr) {};
    __device__ __host__ void init(Vertex v0, Vertex v1, Vertex v2, UniversalMaterial *mat_ptr) {
        v0_ = v0; 
        v1_ = v1; 
        v2_ = v2; 
        mat_ptr_ = mat_ptr;
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const;
    __device__ ~triangle() {
        delete mat_ptr_;
    };

    Vertex v0_, v1_, v2_;
    UniversalMaterial *mat_ptr_;
};

__device__ bool triangle::hit(const ray& r, interval ray_t, hit_record& rec) const {
    // Moller-Trumbore intersection algorithm
    float3 e1 = v1_.position - v0_.position;
    float3 e2 = v2_.position - v0_.position;
    float3 pvec = cross(r.direction(), e2);
    float det = dot(e1, pvec);

    if (det < 1e-8 && det > -1e-8)
        return false;

    float inv_det = 1.0 / det;
    float3 tvec = r.origin() - v0_.position;
    float u = dot(tvec, pvec) * inv_det;

    if (u < 0 || u > 1)
        return false;

    float3 qvec = cross(tvec, e1);
    float v = dot(r.direction(), qvec) * inv_det;

    if (v < 0 || u + v > 1)
        return false;

    float t = dot(e2, qvec) * inv_det;

    if (t < ray_t.max && t > ray_t.min) {
        rec.t = t;
        rec.p = r.point_at_parameter(rec.t);
        rec.normal = normalize(cross(e1, e2));
        rec.mat_ptr = mat_ptr_;

        rec.texCoord.x = (1 - u - v) * v0_.texCoords.x + u * v1_.texCoords.x + v * v2_.texCoords.x; 
        rec.texCoord.y = (1 - u - v) * v0_.texCoords.y + u * v1_.texCoords.y + v * v2_.texCoords.y;

        return true;
    }

    return false;
}