#pragma once
#include "helper_math.h"
class ray
{
   public:
      __device__ ray() {}

      /**
       * @brief Constructs a ray from two points.
       *
       * This constructor creates a ray with a starting point `a` and a direction `b`.
       *
       * @param a The starting point of the ray.
       * @param b The direction of the ray.
       */
      __device__ ray(const float3& a, const float3& b) { A = a; B = b; }
      __device__ float3 origin() const       { return A; }
      __device__ float3 direction() const    { return B; }
      __device__ float3 point_at_parameter(float t) const { return A + t*B; }

      float3 A;
      float3 B;
};