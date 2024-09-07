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

private:
    unsigned int device_count_;
    std::vector<DeviceInfo> device_infos_;
};

class MonitorThread {
public:
    MonitorThread(Renderer &renderer) : renderer(renderer) {}
    void operator()();
    void safeTerminate();

private:
    std::atomic_bool shouldTerminate = false;
    Renderer &renderer;
};


