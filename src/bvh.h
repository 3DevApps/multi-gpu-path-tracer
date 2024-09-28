#pragma once

#include "hitable.h"
#include "ray.h"
#include "helper_math.h"
#include <thrust/sort.h>
#include "interval.h"

#include <thrust/sort.h>



class bvh_node : public hitable {
    public:
        __device__ bvh_node(hitable **list, int start, int end, curandState* local_rand_state) {
            int axis = int(curand_uniform(local_rand_state) * (2+0.999999));
            int size = end - start;
            printf("start: %d, mid: %d, end: %d\n", start, start + size/2, end);
            if (size == 1) {
                left = right = list[start];
            } else if (size == 2) {
                left = list[start];
                right = list[start + 1];
            } else {
                // Sort the list of hitables based on the axis
                // sort_hitables(list, start, end, axis);
                sort_hitables_thrust(list, start, end, axis);
                printf("sorted\n");
                left = new bvh_node(list, start, (start + size/2), local_rand_state);
                printf("left\n");
                right = new bvh_node(list, start + size/2, end, local_rand_state);
                printf("right\n");
            }
            bbox = aabb(left->bbox, right->bbox);
        }
        __device__ bool hit (const ray& r, interval ray_t, hit_record& rec) const override {
            if (!bbox.hit(r, ray_t, rec)) {
                return false;
            }
            bool hit_left = left->hit(r, ray_t, rec);
            bool hit_right = right->hit(r, interval(ray_t.min, hit_left ? rec.t : ray_t.max), rec);
            return hit_left || hit_right;
        }
        __device__ ~bvh_node() {
            delete left;
            delete right;
        }

    private:
    __device__ void sort_hitables_thrust(hitable **list, int start, int end, int axis){
        int size = end - start;
        float *keys = new float[size];
        for (int i = 0; i < size; i++){
            if (axis == 0){
                keys[i] = list[start + i]->bbox.x.min;
            }else if (axis == 1){
                keys[i] = list[start + i]->bbox.y.min;
            }else{
                keys[i] = list[start + i]->bbox.z.min;
            }
        }
        thrust::sort_by_key(thrust::device, keys, keys + size, list+start);
        delete[] keys;
    }
        hitable* left;
        hitable* right;       
};