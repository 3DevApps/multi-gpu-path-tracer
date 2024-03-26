#pragma once
#include "hitable.h"
#include "helper_math.h"
#include "ray.h"
#include "aabb.h"
#include "interval.h"


class hitable_list: public hitable
{
    public:
        __device__ hitable_list() {}
        __device__ hitable_list(hitable **l, int n) {list = l; list_size = n; }
        __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const;
        hitable **list;
        int list_size;
        aabb bbox;
};
/**
 * Determines if a ray intersects with the objects in the hitable list.
 *
 * @param r The ray to test for intersection.
 * @param t_min The minimum value of the parameter t for valid intersections.
 * @param t_max The maximum value of the parameter t for valid intersections.
 * @param rec The hit record that stores information about the intersection point.
 * @return True if the ray intersects with any object in the hitable list, false otherwise.
 */
__device__ bool hitable_list::hit(const ray& r, interval ray_t, hit_record& rec) const
{
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