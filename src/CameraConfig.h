#pragma once

#include "helper_math.h"

struct CameraConfig {
    CameraConfig(float3 lookFrom, float3 front, float vfov = 45.0f, float hfov = 45.0f) :
        front{front},
        lookFrom{lookFrom},
        vfov{vfov},
        hfov{hfov} {}
    float3 front;
    float3 lookFrom;
    float vfov = 45.0f;
    float hfov = 45.0f;
    float pitch;
    float yaw;
};
