#pragma once

#include <curand_kernel.h>
#include "helper_math.h"

struct Resolution {
    unsigned int width;
    unsigned int height;
};

enum SchedulingAlgorithmType {
    BASIC_SCHEDULING,
    FTDL,
    DTDL
};

struct RendererConfig {
    std::string jobId = "0";
    unsigned int samplesPerPixel = 10; 
    unsigned int recursionDepth = 3; 
    std::string modelPath{};
    unsigned int gpuNumber = 4;
    unsigned int streamsPerGpu = 1;
    Resolution resolution{400, 400}; 
    SchedulingAlgorithmType algorithmType = FTDL; // TODO: add support for different algorithms
    dim3 threadBlockSize{8, 8}; 
    float vfov = 45.0f; 
    float hfov = 45.0f; 
    float3 cameraLookFromVec{0.0f, 0.0f, 0.0f};
    float3 cameraFrontVec{1.0f, 0.0f, 0.0f};
    unsigned int maxTasksInRow = 1; // 
    bool showTasks = true;
};
