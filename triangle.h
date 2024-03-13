#pragma once
#include "helper_math.h"
#include "material.h"
#include "hitable.h"

// Class representing a 3D triangle
class triangle : public hitable {
public:
    float3 v0, v1, v2;
    material mat;

    triangle(float3 v0, float3 v1, float3 v2, material mat) : v0(v0), v1(v1), v2(v2), mat(mat) {}

    // Function to check if a ray intersects with the triangle
    virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        float3 e1 = v1 - v0;
        float3 e2 = v2 - v0;
        float3 h = cross(r.direction, e2);
        float a = dot(e1, h);

        if (a > -0.00001 && a < 0.00001) {
            return false;
        }

        float f = 1.0 / a;
        float3 s = r.origin - v0;
        float u = f * dot(s, h);

        if (u < 0.0 || u > 1.0) {
            return false;
        }

        float3 q = cross(s, e1);
        float v = f * dot(r.direction, q);

        if (v < 0.0 || u + v > 1.0) {
            return false;
        }

        float t = f * dot(e2, q);

        if (t > t_min && t < t_max) {
            rec.t = t;
            rec.p = r.point_at_parameter(t);
            rec.normal = normalize(cross(e1, e2));
            rec.mat = mat;
            return true;
        }

        return false;
    }
};

