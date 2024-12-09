#pragma once

#include "helper_math.h"
#include "onb.h"
#include "hitable_list.h"

class pdf {
    public:
    __device__ pdf() {}
    __device__ virtual float value(const float3& direction) const = 0;
    __device__ virtual float3 generate(curandState *local_rand_state) const = 0;
};

class cosine_pdf : public pdf {
    public:
    __device__ cosine_pdf(const float3& w) { 
        uvw = onb(w);
        }
    __device__ virtual float value(const float3& direction) const override {
        float cosine = dot(normalize(direction), uvw.w());
        return (cosine <= 0) ? 0 : cosine / M_PI;
    }
    __device__ virtual float3 generate(curandState *local_rand_state) const override {
        return uvw.local(random_cosine_direction(local_rand_state));
    }

    onb uvw;
};

class triangle_pdf : public pdf {
    public:
    __device__ triangle_pdf(const triangle *p, const float3& origin) : ptr(p), o(origin) {}
    __device__ virtual float value(const float3& direction) const override {
        return ptr->pdf_value(o, direction);
    }
    __device__ virtual float3 generate(curandState *local_rand_state) const override {
        return ptr->random(o, local_rand_state);
    }

    float3 o;
    const triangle *ptr;
};
class hitable_list_pdf:public pdf{
    public:
    __device__ hitable_list_pdf(const hitable_list *l, const float3& o) : list(l), origin(o) {}
    __device__ virtual float value(const float3& direction) const override {
        return list->pdf_value(origin, direction);
    }
    __device__ virtual float3 generate(curandState *local_rand_state) const override {
        return list->random(origin, local_rand_state);
    }

    const hitable_list *list;
    float3 origin;

};
class mixture_pdf : public pdf {
    public:
    __device__ mixture_pdf(const pdf *p0, const pdf *p1) { 
        p[0] = p0; 
        p[1] = p1;
    }
    __device__ virtual float value(const float3& direction) const override {
        return 0.5f * p[0]->value(direction) + 0.5f * p[1]->value(direction);
    }
    __device__ virtual float3 generate(curandState *local_rand_state) const override {
        if (curand_uniform(local_rand_state) < 0.5f) {
            return p[0]->generate(local_rand_state);
        } else {
            return p[1]->generate(local_rand_state);
        }
    }

    const pdf *p[2];
};