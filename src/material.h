#pragma once
#include "hitable.h"
#include "ray.h"
#include "helper_math.h"
class hit_record;

__device__ float schlick_approx(float cosine, float ref_idx) {
    float r0 = (1-ref_idx) / (1+ref_idx);
    r0 = r0*r0;
    return r0 + (1-r0)*pow((1 - cosine),5);
}

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

class dielectric : public material {
    public:
        __device__ dielectric(float index_of_refraction) : ir(index_of_refraction) {}
        __device__ virtual bool scatter(
        const ray& r_in, const hit_record& rec, float3& attenuation, ray& scattered, curandState *local_rand_state) 
        const override {
            float3 outward_normal;
            float3 reflected = reflect(r_in.direction(), rec.normal);
            float ni_over_nt;
            attenuation = make_float3(1.0, 1.0, 1.0);
            float3 refracted;
            float reflect_prob;
            float cosine;
            if (dot(r_in.direction(), rec.normal) > 0.0f) {
                outward_normal = -rec.normal;
                ni_over_nt = ir;
                cosine = dot(r_in.direction(), rec.normal) / length(r_in.direction());
                cosine = sqrt(1.0f - ir*ir*(1-cosine*cosine));
            }
            else {
                outward_normal = rec.normal;
                ni_over_nt = 1.0f / ir;
                cosine = -dot(r_in.direction(), rec.normal) / length(r_in.direction());
            }
            if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
                reflect_prob = schlick_approx(cosine, ir);
            else
                reflect_prob = 1.0f;
            if (curand_uniform(local_rand_state) < reflect_prob)
                scattered = ray(rec.p, reflected);
            else
                scattered = ray(rec.p, refracted);
            return true;
        }       
        // __device__ virtual bool scatter(
        // const ray& r_in, const hit_record& rec, float3& attenuation, ray& scattered, curandState *local_rand_state) 
        // const override {
        //     attenuation = make_float3(1.0, 1.0, 1.0);
        //     float3 outward_normal;
        //     float ni_over_nt;
        //     float3 unit_direction = normalize(r_in.direction());
        //     if (dot(r_in.direction(), rec.normal) > 0.0f) {
        //         outward_normal = -rec.normal;
        //         ni_over_nt = ir;
        //     }else {
        //         outward_normal = rec.normal;
        //         ni_over_nt = 1.0f / ir;
        //     }
        //     float cos_theta = fminf(dot(-unit_direction, outward_normal), 1.0);
        //     float sin_theta = sqrt(1.0 - cos_theta*cos_theta);
        //     bool cannot_refract = ni_over_nt * sin_theta > 1.0;
        //     float3 direction;
        //     if (cannot_refract || schlick_approx(cos_theta, ni_over_nt) > curand_uniform(local_rand_state)) {
        //         direction = reflect(unit_direction, outward_normal);
        //     } else {
        //         direction = refract(unit_direction, outward_normal, ni_over_nt);
        //     }
        //     scattered = ray(rec.p, direction);
        //     return true;
        // }

        float ir; // Index of Refraction
};
