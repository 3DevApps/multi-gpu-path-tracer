#pragma once
#include "nvVector.h"
#include "hitable.h"

// Class representing a 3D object
// This is a wrapper for all triangles available from objfile
class scene3d : public hitable {
public:
    // Array of triangles
    hitable **list;
    int list_size;

    // Constructor
    scene3d(hitable **l, int n) : list(l), list_size(n) {}

    // Function to check if a ray intersects with the object
    virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        float closest_so_far = t_max;

        for (int i = 0; i < list_size; i++) {
            if (list[i]->hit(r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
};
