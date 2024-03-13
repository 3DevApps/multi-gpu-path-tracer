#pragma once
#include "helper_math.h"

// Class representing a material
class material {
public:
    float3 ambientColor = make_float3(0.1f, 0.1f, 0.1f);
    float3 diffuseColor = make_float3(0.5f, 0.5f, 0.5f);
    float3 specularColor = make_float3(0.5f, 0.5f, 0.5f);

    float specularExponent = 32.0f;
    float transparency = 1.0f;

    float illuminationModel = 2.0f;

    float3 color = make_float3(20/255.0, 117/255.0, 83/255.0);    
};