#pragma once

#include <curand_kernel.h>
#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "helper_math.h"
#include "hitable.h"
#include "interval.h"
#include "aabb.h"
#include "material.h"
#include "HostScene.h"

class triangle {
public:
    __device__ triangle() {}
    triangle(Vertex v0, Vertex v1, Vertex v2, UniversalMaterial *mat_ptr) : 
        v0_{v0}, 
        v1_{v1}, 
        v2_{v2}, 
        mat_ptr_{mat_ptr} {
    
        interval x_interval = interval(min(v0.position.x,min(v1.position.x, v2.position.x)), max(v0.position.x,max(v1.position.x,v2.position.x)));
        interval y_interval = interval(min(v0.position.y,min(v1.position.y,v2.position.y)),max(v0.position.y,max(v1.position.y,v2.position.y)));
        interval z_interval = interval(min(v0.position.z,min(v1.position.z,v2.position.z)),max(v0.position.x,max(v1.position.z,v2.position.z)));
        bbox = aabb(x_interval, y_interval, z_interval);
        centroid = (v0_.position + v1_.position + v2_.position) * 0.3333f;
        area = length(cross(v1_.position - v0_.position, v2_.position - v0_.position)) * 0.5f;
    }

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const;
    __device__ float pdf_value(const float3& o, const float3& v) const {
        hit_record rec;
        if (!this->hit(ray(o, v), interval(0.001f, FLT_MAX), rec)) {
            return 0;
        }
        float distance_squared = rec.t * rec.t * dot(v, v);
        float cosine = fabs(dot(v, rec.normal) / length(v));
        return distance_squared / (cosine * area);
    }
    __device__ float3 random(const float3& o, curandState *local_rand_state) const {
        float r1 = curand_uniform(local_rand_state);
        float r2 = curand_uniform(local_rand_state);
        float sqrt_r1 = sqrt(r1);
        float3 random_point = (1 - sqrt_r1) * v0_.position + (sqrt_r1 * (1 - r2)) * v1_.position + (sqrt_r1 * r2) * v2_.position;
        return random_point - o;
    }
    __device__ void debug_print()const{
        printf("Triangle: \n");
        printf("v0: %f %f %f\n", v0_.position.x, v0_.position.y, v0_.position.z);
        printf("v1: %f %f %f\n", v1_.position.x, v1_.position.y, v1_.position.z);
        printf("v2: %f %f %f\n", v2_.position.x, v2_.position.y, v2_.position.z);
        mat_ptr_->debug_print();
    }

    Vertex v0_, v1_, v2_;
    float area;
    UniversalMaterial *mat_ptr_;
    aabb bbox;
    float3 centroid;
};

__device__ bool triangle::hit(const ray& r, interval ray_t, hit_record& rec) const {
    // return true;
    // Moller-Trumbore intersection algorithm

    float3 e1 = v1_.position - v0_.position;
    float3 e2 = v2_.position - v0_.position;
    // float3 e1 = v1_.position;
    // float3 e2 = v2_.position;
    float3 pvec = cross(r.direction(), e2);
    float det = dot(e1, pvec);

    // rec.mat_ptr = mat_ptr_;
    

    // return true; 
    if (det < 1e-8 && det > -1e-8)
        return false;

    // return true;
    
    // return true;
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