#pragma once

#include "hitable.h"
#include "ray.h"
#include "helper_math.h"
#include <thrust/sort.h>
#include <cub/device/device_radix_sort.cuh>
#include "interval.h"
//axis aligned bounding box

//WORK IN PROGRESS
class internal_node : public hitable{
    public:
        __device__ internal_node(){
            bbox = aabb();
        }
        __device__ internal_node(hitable* l, hitable* r){
            left = l;
            right = r;
            bbox = aabb(l->bbox, r->bbox);
        }
        __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const override{
            if(!bbox.hit(r, ray_t,rec)){
                return false;
            }
            bool hit_left = left->hit(r, ray_t, rec);
            bool hit_right = right->hit(r, interval(ray_t.min,hit_left ? rec.t:ray_t.max), rec);
            return hit_left || hit_right;
        }
        hitable* left = nullptr;
        hitable* right = nullptr;
        internal_node* parent = nullptr;
        int flag = 0;
};
class leaf_node : public hitable{
    public:
        __device__ leaf_node(){}
        __device__ leaf_node(int objectID){
            this->objectID = objectID;
        }
        __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const override{
            //TODO
        }
        // if start == end then it is a leaf node
        int objectID;
        internal_node* parent;
};


__global__ void build_internal_nodes(int n, internal_node *internal_nodes,leaf_node *leaf_nodes,unsigned int *morton_codes,hitable** object_list);
class bvh : public hitable{
    public:
        hitable **list;
        int list_size;
        unsigned int *morton_codes;
        unsigned int *sorted_morton_codes;
        int *sorted_objects_IDs;
        leaf_node *leaf_nodes;
        internal_node *internal_nodes;
        __device__ bvh() {}
        __device__ bvh(hitable **l, int n) {
            list = l;
            list_size = n; 
            morton_codes = new unsigned int[n];
            sorted_objects_IDs = new int[n];
            float3 bbox_center_normlized;
            for (int i = 0; i < n; i++)
            {
                bbox = aabb(bbox, l[i]->bbox);
                morton_codes[i] = i;
            }
            //could be parallelized
            float3 bbox_min = make_float3(bbox.x.min, bbox.y.min, bbox.z.min);
            float3 bbox_size = bbox.size();
            for (int i = 0; i < n; i++)
            {
                //more reaserch needed xd
                bbox_center_normlized = (l[i]->bbox.center() - bbox_min)/bbox_size;
                morton_codes[i] = morton3D(bbox_center_normlized.x, bbox_center_normlized.y, bbox_center_normlized.z);
            }
            sort_morton_codes();
            build_bvh();


        }
        __device__ ~bvh(){
            delete[] sorted_morton_codes;
            delete[] sorted_objects_IDs;
            // delete[] leaf_nodes;
            // delete[] internal_nodes;
        }
        __device__ virtual bool hit(const ray& r, interval ray_t, hit_record& rec) const override{
            return internal_nodes[0].hit(r, ray_t, rec);
        }
        // Expands a 10-bit integer into 30 bits
        // by inserting 2 zeros after each bit.
        __device__ unsigned int expandBits(unsigned int v){
            v = (v * 0x00010001u) & 0xFF0000FFu;
            v = (v * 0x00000101u) & 0x0F00F00Fu;
            v = (v * 0x00000011u) & 0xC30C30C3u;
            v = (v * 0x00000005u) & 0x49249249u;
            return v;
        }

        // Calculates a 30-bit Morton code for the
        // given 3D point located within the unit cube [0,1].
        __device__ unsigned int morton3D(float x, float y, float z){
            x = min(max(x * 1024.0f, 0.0f), 1023.0f);
            y = min(max(y * 1024.0f, 0.0f), 1023.0f);
            z = min(max(z * 1024.0f, 0.0f), 1023.0f);
            unsigned int xx = expandBits((unsigned int)x);
            unsigned int yy = expandBits((unsigned int)y);
            unsigned int zz = expandBits((unsigned int)z);
            return xx * 4 + yy * 2 + zz;
        }


        __device__ void print_morton_codes(){  
            //debug print yoooo
            printf("printing SORTED morton codes\n");
            for (int i = 0; i < list_size; i++)
            {
                printf("morton code %d\n",morton_codes[i]);
                printf("sorted_objects_IDs %d\n",sorted_objects_IDs[i]);
            }
        }
        __device__ void sort_morton_codes(){
            sorted_morton_codes = new unsigned int[list_size];
            int *temp_IDs = new int[list_size];
            for (int i = 0; i < list_size; i++)
            {
                temp_IDs[i] = i;
            }
            void     *d_temp_storage = NULL;
            size_t   temp_storage_bytes = 0;

            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                morton_codes, sorted_morton_codes, temp_IDs, sorted_objects_IDs, list_size);
                
            // Allocate temporary storage
            cudaMalloc(&d_temp_storage, temp_storage_bytes);

            // Run sorting operationc
            cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                morton_codes, sorted_morton_codes, temp_IDs, sorted_objects_IDs, list_size);
            __syncthreads();
            delete[] temp_IDs;
            delete[] morton_codes;
        }
        __device__ int find_index_from_pointer(hitable* ptr, hitable* list, int n){
            for (int i = 0; i < n; i++)
            {
                if (&list[i] == ptr)
                {
                    return i;
                }
            }
            return -1;
        }
        __device__ void build_bvh(){
            leaf_nodes = new leaf_node[list_size];
            internal_nodes = new internal_node[list_size-1];
            // Construct leaf nodes.
            // Note: This step can be avoided by storing
            // the tree in a slightly different way.
            //  NO IDEA HOW
            for (int i = 0; i < list_size; i++)
            {
                leaf_nodes[i] = leaf_node(sorted_objects_IDs[i]);
            }
            build_internal_nodes<<<(list_size+255)/256, 256>>>(list_size, internal_nodes, leaf_nodes, sorted_morton_codes,list);
            __syncthreads();
        }
};

__device__ void determine_range(unsigned int *morton_codes,int n,int index,int *first,int *last){ //morton codes need to be sorted so
    if (index == 0){
        *first = 0;
        *last = n-1;
        return;
    }
    if (index == n-1){
        *first = 0;
        *last = n-1;
        return;
    }
    unsigned int common_prefix_left = __clz(morton_codes[index-1] ^ morton_codes[index]);
    unsigned int common_prefix_right = __clz(morton_codes[index+1] ^ morton_codes[index]);
    int direction = (common_prefix_left > common_prefix_right) ? -1 : 1;
    unsigned int common_prefix = (direction == -1) ? common_prefix_right : common_prefix_left;

    int lmax = 2;
    while (index + (lmax*direction) >= 0 && index + (lmax*direction) < n && 
    __clz(morton_codes[index] ^ morton_codes[index + (lmax*direction)]) > common_prefix)//out of bounds check
    {
        lmax *= 2;
    }
    int l = 0;
    for (int t = lmax/2; t >= 1; t /= 2)
    {
        if (index + ((l + t)*direction) >= 0 && index + ((l + t)*direction) < n && __clz(morton_codes[index] ^ morton_codes[index + ((l + t)*direction)]) > common_prefix)
        {
            l += t;
        }
    }
    int end = index + l*direction;
    *first = min(index, end);
    *last = max(index, end);
    return;
}

__device__ int find_split(unsigned int *morton_codes,int first,int last){
    unsigned int firstCode = morton_codes[first];
    unsigned int lastCode = morton_codes[last];
    if (firstCode == lastCode)
    {
        return (first + last) / 2;
    }
    int commonPrefix = __clz(firstCode ^ lastCode);
    int split = first;
    int step = last - first;
    do
    {
        step = (step + 1) / 2;
        int newSplit = split + step;
        if (newSplit < last)
        {
            unsigned int splitCode = morton_codes[newSplit];
            int splitPrefix = __clz(firstCode ^ splitCode);
            if (splitPrefix > commonPrefix)
            {
                split = newSplit;
            }
        }
    } while (step > 1);
    return split;
}
__device__ void calculate_bbox(internal_node* node){
    
    if(atomicAdd(&node->flag,1) == 0){
        return;
    }
    node->bbox = aabb(node->left->bbox, node->right->bbox);    
    if (node->parent != nullptr){
        calculate_bbox(node->parent);
    }
    return;
}
__global__ void build_internal_nodes(int n, internal_node *internal_nodes,leaf_node *leaf_nodes,unsigned int *morton_codes,hitable **object_list){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= n){
        return;
    }
    if (index != n-1){
        int first, last;
        determine_range(morton_codes,n,index, &first, &last);
        int split = find_split(morton_codes, first, last);
        internal_nodes[index] = internal_node();
        if (split == first || split ==last){
            internal_nodes[index].left = object_list[split];
            leaf_nodes[split].parent = &internal_nodes[index];
        }else{
            internal_nodes[index].left = &internal_nodes[split];
            internal_nodes[split].parent = &internal_nodes[index];
        }
        if (split+1 == first || split+1 == last){
            internal_nodes[index].right = object_list[split+1];
            leaf_nodes[split+1].parent = &internal_nodes[index];
        }else{
            internal_nodes[index].right = &internal_nodes[split+1];
            internal_nodes[split+1].parent = &internal_nodes[index];
        }
    }
    __syncthreads();
    calculate_bbox(leaf_nodes[index].parent);
    __syncthreads();
}




