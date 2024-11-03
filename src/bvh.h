#pragma once

#include "hitable.h"
#include "ray.h"
#include "helper_math.h"
#include <thrust/sort.h>
#include "interval.h"

#include <thrust/sort.h>



class bvh {
    public:
        __device__ bvh(triangle *list, int start, int end, curandState* local_rand_state) {
            int axis = int(curand_uniform(local_rand_state) * (2+0.999999));
            int size = end - start;
            list_ = list;
            if (size == 1) {
                index = start;
                bbox = aabb(list[index].bbox, list[index].bbox);
            } 
            else {
                sort_hitables_thrust(list, start, end, axis);
                left = new bvh(list, start, (start + size/2), local_rand_state);
                right = new bvh(list, start + size/2, end, local_rand_state);
                bbox = aabb(left->bbox, right->bbox);
            }
        }

        __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const {
            bvh* stack[64];
            bvh** stackPtr = stack;
            *stackPtr++ = nullptr; // push
            bool found = false;
            interval closest = ray_t;
            interval rt = ray_t;

            const bvh* node = this;
            do {
                bvh* childL = node->left;
                bvh* childR = node->right;

                bool overlapL = childL->bbox.hit(r, rt, rec);
                bool overlapR = childR->bbox.hit(r, rt, rec);

                // Query overlaps a leaf node => report collision.
                if (overlapL && childL->index != -1) {
                    if (list_[childL->index].hit(r, ray_t, rec)) {
                        ray_t = interval(ray_t.min, rec.t);
                        found = true;
                    }
                }

                if (overlapR && childR->index != -1){
                    if (list_[childR->index].hit(r, ray_t, rec)) {
                        ray_t = interval(ray_t.min, rec.t);
                        found = true;
                    }
                }   

                // Query overlaps an internal node => traverse.
                bool traverseL = (overlapL && childL->index == -1);
                bool traverseR = (overlapR && childR->index == -1);

                if (!traverseL && !traverseR)
                    node = *--stackPtr; 
                else {
                    node = (traverseL) ? childL : childR;
                    if (traverseL && traverseR)
                        *stackPtr++ = childR;
                }
            } while(node != nullptr);
            return found;
        }

        __device__ ~bvh() {
            delete left;
            delete right;
        }

    private:

    __device__ void sort_hitables_thrust(triangle *list, int start, int end, int axis){
        int size = end - start;
        float *keys = new float[size];
        for (int i = 0; i < size; i++){
            if (axis == 0){
                keys[i] = list[start + i].bbox.x.min;
            }else if (axis == 1){
                keys[i] = list[start + i].bbox.y.min;
            }else{
                keys[i] = list[start + i].bbox.z.min;
            }
        }
        thrust::sort_by_key(thrust::seq, keys, keys + size, list+start);
        delete[] keys;
    }

    bvh* left;
    bvh* right;      
    int index = -1;
    triangle* list_;
    aabb bbox; 
};