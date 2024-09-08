#pragma once

struct Resolution {
    unsigned int width;
    unsigned int height;
};

enum SchedulingAlgorithmType {
    BASIC_SCHEDULING
};

struct RendererConfig {
    unsigned int samplesPerPixel = 20; 
    unsigned int recursionDepth = 3; 
    std::string objPath = "models/cornell-box.obj"; 
    unsigned int gpuNumber = 2;   
    unsigned int streamsPerGpu = 5;
    Resolution resolution{400, 400}; 
    SchedulingAlgorithmType algorithmType = BASIC_SCHEDULING;
    dim3 threadBlockSize{8, 8};
    float verticalFieldOfView = 45.0f; 
    float horizontalFieldOfView = 45.0f; 
    float3 cameraLookFromVec{0.0f, 0.0f, 0.0f};
    float3 cameraFrontVec{1.0f, 0.0f, 0.0f};
};
