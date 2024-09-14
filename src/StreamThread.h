
#pragma once

#include <nvml.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include "semaphore.h"
#include "DevicePathTracer.h"
#include "SafeQueue.h"
#include <thread>


class StreamThread {
public:
    StreamThread(int device_idx, SafeQueue<RenderTask> &queue, semaphore* thread_semaphore, std::condition_variable* thread_cv, std::atomic_int* completed_streams, std::shared_ptr<DevicePathTracer> devicePathTracer):
        deviceIdx{device_idx},
        devicePathTracer{devicePathTracer},
        queue{queue},
        thread_semaphore{thread_semaphore},
        thread_cv{thread_cv},
        completed_streams{completed_streams} {
        
        cudaSetDevice(device_idx);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        cudaStreamCreate(&stream);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        cudaEventCreate(&event);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());   
    }

    ~StreamThread() {
        shouldTerminate = true;
        if(thread_.joinable()) {
            thread_.join();
        } 
    }

    void start() {
        thread_ = std::thread(&StreamThread::threadMain, this);
    }

    void finish() {
        shouldTerminate = true;
    }

    void join() {
        if(thread_.joinable()) {
            thread_.join();
        } 
    }

    void threadMain() {
        RenderTask task;
        while(!shouldTerminate){
            while(!shouldTerminate && queue.ConsumeSync(task)) {
                devicePathTracer->renderTaskAsync(task, stream);
                devicePathTracer->synchronizeStream(stream);
                completed_streams->fetch_add(1);
                thread_cv->notify_all();
                if (shouldTerminate) {
                    return;
                }
            }
        }
    }

private:
    int deviceIdx;
    std::atomic_bool shouldTerminate = false;
    std::shared_ptr<DevicePathTracer> devicePathTracer;
    SafeQueue<RenderTask> &queue;
    semaphore* thread_semaphore;
    std::condition_variable* thread_cv;
    std::atomic_int* completed_streams;
    cudaEvent_t event;
    cudaStream_t stream;
    std::thread thread_{};
};
