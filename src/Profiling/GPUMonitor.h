#pragma once

#include <nvml.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include "../Renderer/Renderer.h"

#ifndef NVML_RT_CALL
#define NVML_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<nvmlReturn_t>( call );                                                               \
        if ( status != NVML_SUCCESS )                                                                                  \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA NVML call \"%s\" in line %d of file %s failed "                                      \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     nvmlErrorString( status ),                                                                        \
                     status );                                                                                         \
    }
#endif

struct DeviceInfo {
    nvmlDevice_t device_handle;
    char name[NVML_DEVICE_NAME_V2_BUFFER_SIZE];
    nvmlMemory_t memory_info;
    nvmlUtilization_t utilization;
};

class GPUMonitor {
public:
    GPUMonitor();
    void queryStats();
    void logLatestStats();
    ~GPUMonitor();
    std::string getLatestStats();
    void updateFps();
    void updateTimeOfRendering(int gpuIdx, float ms);
    float avgTimeOfRendering(int gpuIdx);
    void updateImbalance(float im);
    float avgImbalance();

private:
    unsigned int device_count_;
    std::vector<DeviceInfo> device_infos_;
    size_t free_byte;
    size_t total_byte;
    int average_fps_ = 0;
    int frame_count_ = 0;
    int fps_ = 0;
    int current_fps_ = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_fps_update_;
    std::vector<std::vector<float>> timesOfRendering{8};
    std::vector<float> loadImbalances{};
};

class MonitorThread {
public:
    MonitorThread(Renderer &renderer);
    void operator()();
    void safeTerminate();
    void updateFps();
    void updateTimeOfRendering(int gpuIdx, float ms);
    void updateImbalance(float im);

private:
    std::atomic_bool shouldTerminate = false;
    Renderer &renderer;
    GPUMonitor monitor_;
};
