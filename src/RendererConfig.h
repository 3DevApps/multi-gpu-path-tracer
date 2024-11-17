#pragma once

#include <curand_kernel.h>
#include "helper_math.h"

struct Resolution {
    unsigned int width;
    unsigned int height;
};

enum SchedulingAlgorithmType {
    FST, // Fixed size tasks
    DTFL, // Dynamic tasks with fixed layout
    DT  // Dynamic layout tasks
};

struct RendererConfig {
    std::string jobId = "0";
    unsigned int samplesPerPixel = 10;
    unsigned int recursionDepth = 3;
    std::string modelPath{};
    unsigned int gpuNumber = 2;
    unsigned int streamsPerGpu = 1;
    Resolution resolution{400, 400};
    SchedulingAlgorithmType algorithmType = FST; // TODO: add support for different algorithms
    dim3 threadBlockSize{8, 8};
    float vfov = 45.0f;
    float hfov = 45.0f;
    float3 cameraLookFromVec{0.0f, 0.0f, 0.0f};
    float3 cameraFrontVec{1.0f, 0.0f, 0.0f};
    unsigned int maxTasksInRow = 2; //
    bool showTasks = true;
};
