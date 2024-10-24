#include <sstream>
#include "GPUMonitor.h"

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
    }
}

void GPUMonitor::logLatestStats() {
    for (int i = 0; i < device_count_; i++) {
        printf("ID: %d | Name: %s | Mem Total: %ld MB | Mem Free: %ld MB | GPU Util: %d% | Mem Util: %d%\n",
            i,
            device_infos_[i].name,
            device_infos_[i].memory_info.total / 1000000,
            device_infos_[i].memory_info.free / 1000000,
            device_infos_[i].utilization.gpu,
            device_infos_[i].utilization.memory);
    }
    std::cout << "-------------------------------------------------------------------------" << std::endl; 
}

std::string GPUMonitor::getLatestStats() {
    std::ostringstream stats;
    stats << "ID|Name|Mem Total (MB)|Mem Free (MB)|GPU Util (%)|Mem Util (%)";
    for (int i = 0; i < device_count_; i++) {
        stats << "#" << i << "|" << device_infos_[i].name << "|" <<
            device_infos_[i].memory_info.total / 1000000 << "|" << 
            device_infos_[i].memory_info.free / 1000000 << "|" << 
            device_infos_[i].utilization.gpu << "|" <<
            device_infos_[i].utilization.memory;
    }
    return stats.str();
}

GPUMonitor::~GPUMonitor() {
    NVML_RT_CALL(nvmlShutdown());
}

void MonitorThread::operator()() {
    GPUMonitor monitor;
    while (!shouldTerminate) {
        monitor.queryStats();
        monitor.logLatestStats();
        std::string statsMessage = "JOB_MESSAGE#RENDER_STATS#" + monitor.getLatestStats();
        renderer.send(statsMessage);
        std::this_thread::sleep_for( std::chrono::milliseconds(1000));
    }
}

void MonitorThread::safeTerminate() {
    shouldTerminate = true;
}