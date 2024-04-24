#pragma once
#include "ray.h"
#include "hitable.h"
#include "helper_math.h"
#include "interval.h"

struct hit_record;
class aabb{
public:
    interval x, y, z;

    __device__ aabb() {} // The default AABB is empty, since intervals are empty by default.

    __device__ aabb(const interval& ix, const interval& iy, const interval& iz)
      : x(ix), y(iy), z(iz) { }

    __device__ aabb(const float3& a, const float3& b) {
        // Treat the two points a and b as extrema for the bounding box, so we don't require a
        // particular minimum/maximum coordinate order.
        x = interval(fmin(a.x,b.x), fmax(a.x,b.x));
        y = interval(fmin(a.y,b.y), fmax(a.y,b.y));
        z = interval(fmin(a.z,b.z), fmax(a.z,b.z));
    }

    __device__ aabb(const aabb& box0, const aabb& box1) {
        x = interval(box0.x, box1.x);
        y = interval(box0.y, box1.y);
        z = interval(box0.z, box1.z);
    }

    //TODO: check diffrent aproach from the book (speed)
    __device__ bool hit(const ray& r,interval ray_t, hit_record& rec) const {
        auto t0 = fmin((x.min - r.origin().x) / r.direction().x,
                        (x.max - r.origin().x) / r.direction().x);
        auto t1 = fmax((x.min - r.origin().x) / r.direction().x,
                        (x.max - r.origin().x) / r.direction().x);
        ray_t.min = fmax(t0, ray_t.min);
        ray_t.max = fmin(t1, ray_t.max);
        if (ray_t.max <= ray_t.min)
            return false;

        t0 = fmin((y.min - r.origin().y) / r.direction().y,
                  (y.max - r.origin().y) / r.direction().y);
        t1 = fmax((y.min - r.origin().y) / r.direction().y,
                    (y.max - r.origin().y) / r.direction().y);
        ray_t.min = fmax(t0, ray_t.min);
        ray_t.max = fmin(t1, ray_t.max);
        if (ray_t.max <= ray_t.min)
            return false;

        t0 = fmin((z.min - r.origin().z) / r.direction().z,
                  (z.max - r.origin().z) / r.direction().z);
        t1 = fmax((z.min - r.origin().z) / r.direction().z,
                    (z.max - r.origin().z) / r.direction().z);
        ray_t.min = fmax(t0, ray_t.min);
        ray_t.max = fmin(t1, ray_t.max);
        if (ray_t.max <= ray_t.min)
            return false;

        return true;
    }
    __device__ float3 center() const {
        return make_float3((x.min + x.max) / 2, (y.min + y.max) / 2, (z.min + z.max) / 2);
    }
    __device__ float3 size() const {
        return make_float3(x.max - x.min, y.max - y.min, z.max - z.min);
    }
};