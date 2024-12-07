#pragma once
#include "ray.h"
#include "hitable.h"
#include "helper_math.h"
#include "interval.h"

struct hit_record;
class aabb{
public:
    interval x, y, z;

    __device__ __host__ aabb() {
        pad_to_minimums();
    }

    __device__ __host__ aabb(const interval& ix, const interval& iy, const interval& iz)
      : x(ix), y(iy), z(iz) {
        pad_to_minimums();
      }

    __device__ __host__ aabb(const float3& a, const float3& b) {
        // Treat the two points a and b as extrema for the bounding box, so we don't require a
        // particular minimum/maximum coordinate order.
        x = interval(fmin(a.x,b.x), fmax(a.x,b.x));
        y = interval(fmin(a.y,b.y), fmax(a.y,b.y));
        z = interval(fmin(a.z,b.z), fmax(a.z,b.z));
        pad_to_minimums();
    }

    __device__ __host__ aabb(const aabb& box0, const aabb& box1) {
        x = interval(box0.x, box1.x);
        y = interval(box0.y, box1.y);
        z = interval(box0.z, box1.z);
        pad_to_minimums();
    }

    //TODO: check diffrent aproach from the book (speed)
    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const {
        // r.dir is unit direction vector of ray
        float3 dirfrac = make_float3(0, 0, 0); 
        dirfrac.x = 1.0f / r.direction().x;
        dirfrac.y = 1.0f / r.direction().y;
        dirfrac.z = 1.0f / r.direction().z;

        // lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
        // r.org is origin of ray
        float t1 = (x.min - r.origin().x) * dirfrac.x;
        float t2 = (x.max - r.origin().x) * dirfrac.x;
        float t3 = (y.min - r.origin().y) * dirfrac.y;
        float t4 = (y.max - r.origin().y) * dirfrac.y;
        float t5 = (z.min - r.origin().z) * dirfrac.z;
        float t6 = (z.max - r.origin().z) * dirfrac.z;

        float tmin = fmax(fmax(fmin(t1, t2), fmin(t3, t4)), fmin(t5, t6));
        float tmax = fmin(fmin(fmax(t1, t2), fmax(t3, t4)), fmax(t5, t6));

        // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
        if (tmax < 0) {
            return false;
        }

        if (tmin > tmax) {
            return false;
        }
        return true;
    }

    __device__ float3 center() const {
        return make_float3((x.min + x.max) / 2, (y.min + y.max) / 2, (z.min + z.max) / 2);
    }
    __device__ float3 size() const {
        return make_float3(x.max - x.min, y.max - y.min, z.max - z.min);
    }

    __device__ float area() const {
        float3 e = make_float3(x.max - x.min, y.max - y.min, z.max - z.min); // box extent
        return e.x * e.y + e.y * e.z + e.z * e.x; 
    }

    __device__ __host__ void extend(float3 v) {
        x.min = fmin(v.x, x.min);
        y.min = fmin(v.y, y.min);
        z.min = fmin(v.z, z.min);

        x.max = fmax(v.x, x.max);
        y.max = fmax(v.y, y.max);
        z.max = fmax(v.z, z.max);
    }
 
    __device__ __host__ void pad_to_minimums() {
        // Adjust the AABB so that no side is narrower than some delta, padding if necessary.
        float delta = 0.00001;
        if (x.size() < delta) x = x.expand(delta);
        if (y.size() < delta) y = y.expand(delta);
        if (z.size() < delta) z = z.expand(delta);
    }
};