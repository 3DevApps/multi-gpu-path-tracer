
#pragma once

#include <curand_kernel.h>
#include "helper_math.h"

struct CameraParams {
    float3 front;
    float3 lookFrom;
};

// Materials supported by the obj loader
enum material_type {
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    DIFFUSE_LIGHT
};

struct m_ai_material {
    material_type type ;
    float3 color_ambient;
    float3 color_diffuse;
    float index_of_refraction;
    float shininess;
};

struct Triangle {
    float3 v0;
    float3 v1;
    float3 v2;
    m_ai_material material_params;
};

struct HostScene {
    std::vector<Triangle> triangles;
    CameraParams cameraParams;
};