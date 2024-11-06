#pragma once

#include "hitable.h"
#include "ray.h"
#include "helper_math.h"
#include <thrust/sort.h>
#include "interval.h"

#include <thrust/sort.h>

struct BVHNode {
    aabb bbox;
    uint leftNode, rightNode; 
    bool isLeaf;           
    uint firstTriIdx, triCount;   
};  

class BVH {
    public:
        __device__ BVH(triangle *list, int size) {
            list_ = list;
            indices = new int[size * 2 - 1];
            nodes = new BVHNode[size * 2 - 1];
            for (int i = 0; i < size; i++) 
                indices[i] = i;

            BVHNode& root = nodes[rootNodeIdx];
            root.leftNode = root.rightNode = 0;
            root.firstTriIdx = 0;
            root.triCount = size;
            updateNodeBounds(rootNodeIdx);
            subdivide(rootNodeIdx);
        }

        __device__ void subdivide(uint nodeIdx) {
            BVHNode& node = nodes[nodeIdx];

            // if node contains 5 or less primitives - termianate
            // makes build faster and does not have impact on query performance
            if (node.triCount <= 5) 
                return;

            int axis;
            float splitPos;

            // find best split axis and position using Surface Area Heuristic cost
            float splitCost = findBestSplitPlane(node, axis, splitPos);
            float nosplitCost = calculateNodeCost(node);

            //split node into subnodes only if it results in better tree 
            if (splitCost >= nosplitCost) 
                return;

            // in-place partition similar to quicksort
            int i = node.firstTriIdx;
            int j = i + node.triCount - 1;
            float pos;
            while (i <= j) {
                if (axis == 0) {
                    pos = list_[indices[i]].centroid.x;
                }
                else if (axis == 1) {
                    pos = list_[indices[i]].centroid.y;             
                }
                else {
                    pos = list_[indices[i]].centroid.z;
                }

                if (pos < splitPos)
                    i++;
                else
                    thrust::swap(indices[i], indices[j--]);
            }

            int leftCount = i - node.firstTriIdx;
            if (leftCount == 0 || leftCount == node.triCount) {
                // printf("leaf node kids: %d\n", node.triCount);
                return;
            }
                
            int leftChildIdx = nodesUsed++;
            int rightChildIdx = nodesUsed++;
            nodes[leftChildIdx].firstTriIdx = node.firstTriIdx;
            nodes[leftChildIdx].triCount = leftCount;
            nodes[rightChildIdx].firstTriIdx = i;
            nodes[rightChildIdx].triCount = node.triCount - leftCount;
            node.leftNode = leftChildIdx;
            node.rightNode = rightChildIdx;
            node.triCount = 0;
            updateNodeBounds(leftChildIdx);
            updateNodeBounds(rightChildIdx);
            subdivide(leftChildIdx);
            subdivide(rightChildIdx);
        }

        __device__ void updateNodeBounds(uint nodeIdx) {
            BVHNode& node = nodes[nodeIdx];
            node.bbox = {};
            for (uint first = node.firstTriIdx, i = 0; i < node.triCount; i++) {
                uint leafTriIdx = indices[first + i];
                triangle& leafTri = list_[leafTriIdx];
                node.bbox.extend(leafTri.v0_.position);
                node.bbox.extend(leafTri.v1_.position);
                node.bbox.extend(leafTri.v2_.position);
            }
        }

        __device__ float findBestSplitPlane(BVHNode& node, int& axis, float& splitPos) {
            float bestCost = 1e30f;
            for (int a = 0; a < 3; a++) {
                float boundsMin;
                float boundsMax;
                if (a == 0) {
                    boundsMin = node.bbox.x.min;
                    boundsMax = node.bbox.x.max;
                }
                else if (a == 1) {
                    boundsMin = node.bbox.y.min;
                    boundsMax = node.bbox.y.max;
                }
                else {
                    boundsMin = node.bbox.z.min;
                    boundsMax = node.bbox.z.max;
                }

                if (boundsMin == boundsMax) continue;
                float scale = (boundsMax - boundsMin) / 4;
                for (uint i = 1; i < 4; i++)
                {
                    float candidatePos = boundsMin + i * scale;
                    float cost = evaluateSAH( node, a, candidatePos );
                    if (cost < bestCost)
                        splitPos = candidatePos, axis = a, bestCost = cost;
                }
            }
            return bestCost;
        }

        __device__ float calculateNodeCost(BVHNode& node) {
            float surfaceArea = node.bbox.area();
            return node.triCount * surfaceArea;
        }

        __device__ float evaluateSAH( BVHNode& node, int axis, float pos ) {
            aabb leftBox, rightBox;
            int leftCount = 0, rightCount = 0;
            float centVal;
            for( uint i = 0; i < node.triCount; i++ )
            {
                triangle& triangle = list_[indices[node.firstTriIdx + i]];
                if (axis == 0) {
                    centVal = triangle.centroid.x;
                }
                else if (axis == 1) {
                    centVal = triangle.centroid.y;
                }
                else {
                    centVal = triangle.centroid.z;
                }
                if (centVal < pos) {
                    leftCount++;
                    leftBox.extend(triangle.v0_.position);
                    leftBox.extend(triangle.v1_.position);
                    leftBox.extend(triangle.v2_.position);
                }
                else {
                    rightCount++;
                    rightBox.extend(triangle.v0_.position);
                    rightBox.extend(triangle.v1_.position);
                    rightBox.extend(triangle.v2_.position);
                }
            }

            float cost = leftCount * leftBox.area() + rightCount * rightBox.area();
            return cost > 0 ? cost : 1e30f;
        }

        __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const {
            const BVHNode* stack[64];
            const BVHNode** stackPtr = stack;

            *stackPtr++ = nullptr; 
            bool found = false;
            interval closest = ray_t;
            interval rt = ray_t;
            const BVHNode* node = &nodes[rootNodeIdx];
            float distL;
            float distR;

            do {
                const BVHNode* childL = &nodes[node->leftNode];
                const BVHNode* childR = &nodes[node->rightNode];

                bool overlapL = childL->bbox.hit(r, rt, rec);
                distL = rec.t;
                bool overlapR = childR->bbox.hit(r, rt, rec);
                distR = rec.t;

                if (overlapL && childL->triCount != 0) {
                    for (uint first = childL->firstTriIdx, i = 0; i < childL->triCount; i++) {
                        uint leafTriIdx = indices[first + i];
                        triangle& leafTri = list_[leafTriIdx];

                        if (leafTri.hit(r, ray_t, rec)) {
                            ray_t = interval(ray_t.min, rec.t);
                            found = true;
                        }
                    }
                }

                if (overlapR && childR->triCount != 0){
                    for (uint first = childR->firstTriIdx, i = 0; i < childR->triCount; i++) {
                        uint leafTriIdx = indices[first + i];
                        triangle& leafTri = list_[leafTriIdx];

                        if (leafTri.hit(r, ray_t, rec)) {
                            ray_t = interval(ray_t.min, rec.t);
                            found = true;
                        }
                    }
                }   

                bool traverseL = (overlapL && childL->triCount == 0);
                bool traverseR = (overlapR && childR->triCount == 0);

                if (!traverseL && !traverseR) {
                    node = *--stackPtr; 
                }
                else {
                    if (traverseL && traverseR) {
                        if(distR < distL) {
                            node = childR;
                            *stackPtr++ = childL;
                        }
                        else {
                            node = childL;
                            *stackPtr++ = childR;
                        }
                        
                    }
                    else {
                        node = (traverseL) ? childL : childR;
                    }
                }
            } while(node != nullptr);
            return found;
        }

        __device__ ~BVH() {
            delete[] indices;
            delete[] nodes;
        }

private:

    triangle* list_;
    int* indices;
    BVHNode* nodes;
    uint rootNodeIdx = 0;
    uint nodesUsed = 1;
 };