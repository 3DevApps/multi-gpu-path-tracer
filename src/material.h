#pragma once
#include "hitable.h"
#include "ray.h"
#include "helper_math.h"
class hit_record;

class material {
  public:
    __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, float3& attenuation, ray& scattered, curandState  *local_rand_state) const = 0;
};
//matte material
//many diffrent approaches to this
//it can either always scatter and attenuate by its reflectance R,
//or it can sometimes scatter (with probabilty 1−R) with no attenuation 
//(where a ray that isn't scattered is just absorbed into the material). 
// It could also be a mixture of both those strategies
class lambertian : public material {
    public:
        __device__ lambertian(const float3& color) : albedo(color) {}
        __device__ virtual bool scatter(
            const ray& r_in, const hit_record& rec, float3& attenuation, ray& scattered, curandState *local_rand_state) 
            const override {
            float3 scatter_direction = rec.normal + random_in_unit_sphere(local_rand_state);
            // Catch degenerate scatter direction
            if (near_zero(scatter_direction)) {
                scatter_direction = rec.normal;
            }
            scattered = ray(rec.p, scatter_direction);
            attenuation = albedo;
            return true;
        }

        float3 albedo;
};
//metal material
class metal : public material {
    public:
        __device__ metal(const float3& color,float f ) : albedo(color),fuzz(f<1 ? f : 1) {}
        __device__ virtual bool scatter(
            const ray& r_in, const hit_record& rec, float3& attenuation, ray& scattered, curandState *local_rand_state) 
            const override {
            float3 reflected = reflect(normalize(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected+fuzz*random_in_unit_sphere(local_rand_state));
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }

        float3 albedo;
        float fuzz;
};


