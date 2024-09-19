#pragma once

#include "hitable.h"
#include "ray.h"
#include "helper_math.h"
#include <thrust/sort.h>
#include <cub/device/device_radix_sort.cuh>
#include "interval.h"

class bvh_node : public hitable {
    public:
        __device__ bvh_node(hitable **list, int start, int end, curandState* local_rand_state) {
            printf("start: %d, end: %d\n", start, end);
            int axis = int(curand_uniform(local_rand_state) * (2+0.999999));
            int size = end - start;
            if (size == 1) {
                left = right = list[start];
            } else if (size == 2) {
                left = list[start];
                right = list[start + 1];
            } else {
                // Sort the list of hitables based on the axis
                // sort_hitables(list, start, end, axis);
                quicksort(list, start, end, axis);
                printf("sorted\n");
                printf("start: %d, end: %d\n", start, start + size/2);
                left = new bvh_node(list, start, start + size/2, local_rand_state);
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
    __device__ void quicksort(hitable **list, int start, int end, int axis){
        if (start >= end) return;
        int pivot = start + (end - start) / 2;
        int left = start;
        int right = end-1;
        while (left <= right){
            if (axis == 0){
                while (list[left]->bbox.x.min < list[pivot]->bbox.x.min) left++;
                while (list[right]->bbox.x.min > list[pivot]->bbox.x.min) right--;
            }else if (axis == 1){
                while (list[left]->bbox.y.min < list[pivot]->bbox.y.min) left++;
                while (list[right]->bbox.y.min > list[pivot]->bbox.y.min) right--;
            }else{
                while (list[left]->bbox.z.min < list[pivot]->bbox.z.min) left++;
                while (list[right]->bbox.z.min > list[pivot]->bbox.z.min) right--;
            }
            if (left <= right){
                hitable *temp = list[left];
                list[left] = list[right];
                list[right] = temp;
                left++;
                right--;
            }
        }
        quicksort(list, start, right, axis);
        quicksort(list, left, end, axis);
    
    }
    __device__ void sort_hitables(hitable **list, int start,int end,int axis){
        int size = end - start;
        int *indexes = new int[size];
        int *sorted_indexes = new int[size];
        float *temp_IDs = new float[size];
        float *sorted_objects_IDs = new float[size];
        for (int i = 0; i < size; i++){
            indexes[i] = i+start;
            if (axis == 0){
                temp_IDs[i] = list[start + i]->bbox.x.min;
            }else if (axis == 1){
                temp_IDs[i] = list[start + i]->bbox.y.min;
            }else{
                temp_IDs[i] = list[start + i]->bbox.z.min;
            }
        }
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;

        //Get needed temporary storage
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
            indexes, sorted_indexes, temp_IDs, sorted_objects_IDs, size);
            
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        // Run sorting operationc
        // cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        //     indexes, sorted_indexes, temp_IDs, sorted_objects_IDs, size);
        __syncthreads();

        cudaFree(d_temp_storage);
        hitable **temp_list = new hitable*[size];
        for (int i = 0; i < size; i++){
            temp_list[i] = list[sorted_indexes[i]];
        }
        for (int i = 0; i < size; i++){
            list[start + i] = temp_list[i];
        }

        delete[] indexes;
        delete[] sorted_indexes;
        // delete[] temp_list;
        delete[] temp_IDs;
        delete[] sorted_objects_IDs;
    }
        hitable* left;
        hitable* right;       
};