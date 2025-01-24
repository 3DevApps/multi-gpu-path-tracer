#include <sstream>
#include "GPUMonitor.h"


#include <iostream>
#include <curand_kernel.h>

#define checkCudaErrors(val) check_cuda_in( (val), #val, __FILE__, __LINE__ )
void check_cuda_in(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA ERROR = " << static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

GPUMonitor::GPUMonitor() {
    nvmlInit();
    NVML_RT_CALL(nvmlDeviceGetCount(&device_count_));
    for (int i = 0; i < device_count_; i++) {
        device_infos_.push_back({});
        NVML_RT_CALL(nvmlDeviceGetHandleByIndex(i, &device_infos_[i].device_handle));
        NVML_RT_CALL(nvmlDeviceGetName(device_infos_[i].device_handle,
                                        device_infos_[i].name,
                                        NVML_DEVICE_NAME_V2_BUFFER_SIZE));
    }
}

void GPUMonitor::queryStats() {
    for (int i = 0; i < device_count_; i++) {
        NVML_RT_CALL( nvmlDeviceGetMemoryInfo(device_infos_[i].device_handle,
                                                &device_infos_[i].memory_info));
        NVML_RT_CALL(nvmlDeviceGetUtilizationRates(device_infos_[i].device_handle,
                                                    &device_infos_[i].utilization));
        checkCudaErrors(cudaMemGetInfo( &free_byte, &total_byte ));
    }
}

void GPUMonitor::logLatestStats() {
    for (int i = 0; i < device_count_; i++) {
        printf("ID: %d | Name: %s | Mem Total: %ld MB | Mem Free: %ld MB | GPU Util: %d | Mem Util: %d | TOR: %f | Imbalance: %f\n",
            i,
            device_infos_[i].name,
            device_infos_[i].memory_info.total / 1000000,
            device_infos_[i].memory_info.free / 1000000,
            device_infos_[i].utilization.gpu,
            device_infos_[i].utilization.memory,
            avgTimeOfRendering(i),
            avgImbalance());
            double free_db = (double)free_byte;
            double total_db = (double)total_byte;
            double used_db = total_db - free_db;
            printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
                used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
    }
    std::cout << "-------------------------------------------------------------------------" << std::endl;
}

void GPUMonitor::updateFps() {
    frame_count_++;
}

void GPUMonitor::updateTimeOfRendering(int gpuIdx, float ms) {
    timesOfRendering[gpuIdx].push_back(ms);
}

float calcAvg(std::vector<float> &v) {
    if (v.size() == 0)
        return 0;

    float sum = 0;
    for (int i = 0; i < v.size(); i++) {
        sum += v[i];
    }
    return sum / v.size();
}

float GPUMonitor::avgTimeOfRendering(int gpuIdx) {
    return calcAvg(timesOfRendering[gpuIdx]);
}

void GPUMonitor::updateImbalance(float im) {
    loadImbalances.push_back(im);
}

float GPUMonitor::avgImbalance() {
    return calcAvg(loadImbalances);
}

std::string GPUMonitor::getLatestStats() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - last_fps_update_);

    if (duration.count() > 0) {
        fps_ = frame_count_ / duration.count();
        frame_count_ = 0;
        last_fps_update_ = now;
        average_fps_ = (average_fps_ + fps_) / 2;
    }

    std::ostringstream stats;
    stats << "FPS|FPS|" << fps_ << "|";
    stats << "FPS|Average FPS|" << average_fps_ << "|";

    auto avgTimeOfRenderingTotal = 0.0;
    for (int i = 0; i < device_count_; i++) {
        avgTimeOfRenderingTotal += avgTimeOfRendering(i);
    }
    avgTimeOfRenderingTotal /= device_count_;

    for (int i = 0; i < device_count_; i++) {
        stats << "Mem Total (MB)|Mem Total GPU " << i << "|" << device_infos_[i].memory_info.total / 1000000 << "|";
        stats << "Mem Free (MB)|Mem Free GPU " << i << "|" << device_infos_[i].memory_info.free / 1000000 << "|";
        stats << "GPU Util (%)|GPU " << i << "|" << device_infos_[i].utilization.gpu << "|";
        stats << "Mem Util (%)|GPU " << i << "|" << device_infos_[i].utilization.memory << "|";
        stats << "Rendering Time (ms)|Rendering Time GPU " << i << "|" << avgTimeOfRendering(i) << "|";
        timesOfRendering[i] = {};
    }
    stats << "Rendering Time (ms)|Rendering Time Total|" << avgTimeOfRenderingTotal << "|";
    stats << "Load Imbalance|Load Imbalance|" << avgImbalance() << "|";
    loadImbalances = {};
    return stats.str();
}

GPUMonitor::~GPUMonitor() {
    NVML_RT_CALL(nvmlShutdown());
}

MonitorThread::MonitorThread(Renderer &renderer) : renderer(renderer) {
    GPUMonitor monitor;
    monitor_ = monitor;
}

void MonitorThread::operator()() {
    while (!shouldTerminate) {
        monitor_.queryStats();
        monitor_.logLatestStats();
        std::string statsMessage = "RENDER_STATS#" + monitor_.getLatestStats();
        renderer.send(statsMessage);
        std::this_thread::sleep_for( std::chrono::milliseconds(500));
    }
}


void MonitorThread::updateFps() {
    monitor_.updateFps();
}

void MonitorThread::updateTimeOfRendering(int gpuIdx, float ms) {
    monitor_.updateTimeOfRendering(gpuIdx, ms);
}

void MonitorThread::updateImbalance(float im) {
    monitor_.updateImbalance(im);
}

void MonitorThread::safeTerminate() {
    shouldTerminate = true;
}
