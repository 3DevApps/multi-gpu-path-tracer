
#pragma once

#include <curand_kernel.h>
#include "helper_math.h"

struct CameraParams {
    float3 front;
    float3 lookFrom;
};