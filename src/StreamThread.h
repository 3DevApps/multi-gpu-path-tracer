
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
#include "barrier.h"


class StreamThread {
public:
    StreamThread(int device_idx, std::shared_ptr<DevicePathTracer> devicePathTracer, std::vector<RenderTask> &tasks, Barrier& barrier):
        deviceIdx{device_idx},
        devicePathTracer{devicePathTracer},
        tasks_{tasks},
        barrier_{barrier} {

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

    void detach() {
        finish();
        thread_.detach();
    }

    void threadMain() {
        RenderTask task;

        while(!shouldTerminate){
            // std::cout << "before lock..." << std::endl;

            if (shouldTerminate) {
                return;
            }

            barrier_.wait(); //wair for all threads before start

            if (shouldTerminate) {
                return;
            }

            auto start = std::chrono::high_resolution_clock::now();

            // devicePathTracer->renderTaskAsync(task, stream);
            devicePathTracer->renderTaskAsync(tasks_[deviceIdx], stream);
            devicePathTracer->synchronizeStream(stream);

            //set time for task
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

            tasks_[deviceIdx].time = duration.count();

            // std::cout << "waiting on barrier..." << std::endl;

            if (shouldTerminate) {
                return;
            }

            barrier_.wait();

            if (shouldTerminate) {
                return;
            }
        }
    }

private:
    int deviceIdx;
    std::atomic_bool shouldTerminate = false;
    std::shared_ptr<DevicePathTracer> devicePathTracer;
    cudaEvent_t event;
    cudaStream_t stream;
    std::thread thread_{};
    std::vector<RenderTask> &tasks_;
    Barrier& barrier_;
};
