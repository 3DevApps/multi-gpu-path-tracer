#pragma once
#include "hitable.h"
#include "helper_math.h"
#include "ray.h"
#include "aabb.h"
#include "interval.h"
#include "triangle.h"

class hitable_list {
    public:
        __device__ hitable_list() {}
        __device__ hitable_list(triangle **l, int n) {list = l; list_size = n;}

        __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const;

        __device__ float pdf_value(const float3& o, const float3& v) const {
            float weight = 1.0f/list_size;
            float sum = 0.0f;
            for (int i = 0; i < list_size; i++)
                sum += weight * list[i]->pdf_value(o, v);
            return sum;
        }
        __device__ float3 random(const float3& o, curandState *local_rand_state) const {
            int index = int(truncf(curand_uniform(local_rand_state)*((list_size - 1) + 0.999999)));
            return list[index]->random(o, local_rand_state);
        }
        triangle **list;
        int list_size;
};


/**
 * Determines if a ray intersects with the objects in the hitable list.
 *
 * @param r The ray to test for intersection.
 * @param ray_t The interval of valid t values for the ray.
 * @param rec The hit record that stores information about the intersection point.
 * @return True if the ray intersects with any object in the hitable list, false otherwise.
 */
__device__ bool hitable_list::hit(const ray& r, interval ray_t, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false; 
    for (int i = 0; i < list_size; i++)
    {
        if (list[i]->hit(r, ray_t, temp_rec))
        {
            hit_anything = true;
            ray_t.max = temp_rec.t;
            rec = temp_rec;
        }
    }
    return hit_anything;
}