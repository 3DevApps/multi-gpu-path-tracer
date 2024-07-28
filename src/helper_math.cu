
#include <curand_kernel.h>
#include <iostream>
#include "helper_math.h"


__device__ bool refract(const float3& v, const float3& n, float ni_over_nt, float3& refracted) {
    float3 uv = normalize(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

__host__ std::ostream& operator<<(std::ostream& os, const float3& vec) {
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}