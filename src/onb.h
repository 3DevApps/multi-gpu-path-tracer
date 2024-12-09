#pragma once

#include "helper_math.h"

class onb{
    public:
    __device__ onb() {}
    __device__ onb(const float3& n) {
        axis[2] = normalize(n);
        float3 a = fabsf(axis[2].x) > 0.9f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
        axis[1] = normalize(cross(axis[2], a));
        axis[0] = cross(axis[2], axis[1]);
    }
    __device__ float3 operator[](int i) const { return axis[i]; }
    __device__ float3 u() const { return axis[0]; }
    __device__ float3 v() const { return axis[1]; }
    __device__ float3 w() const { return axis[2]; }

    __device__ float3 local(float3 v) const {
        return v.x * axis[0] + v.y * axis[1] + v.z * axis[2];
    }
    private:
    float3 axis[3];

};