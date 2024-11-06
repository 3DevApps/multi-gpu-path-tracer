
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
    StreamThread(int device_idx, SafeQueue<RenderTask> &queue, semaphore* thread_semaphore, std::condition_variable* thread_cv, std::atomic_int* completed_streams, std::shared_ptr<DevicePathTracer> devicePathTracer, std::vector<RenderTask> &tasks):
        deviceIdx{device_idx},
        devicePathTracer{devicePathTracer},
        queue{queue},
        thread_semaphore{thread_semaphore},
        thread_cv{thread_cv},
        completed_streams{completed_streams}, 
        tasks_{tasks} {
        
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

        //todo queue is not important
        while(!shouldTerminate){
            while(!shouldTerminate && queue.ConsumeSync(task)) {
                auto start = std::chrono::high_resolution_clock::now();

                // devicePathTracer->renderTaskAsync(task, stream);
                devicePathTracer->renderTaskAsync(tasks_[deviceIdx], stream);
                devicePathTracer->synchronizeStream(stream);

                //set time for task 

                
                // manager.renderFrame();
                auto stop = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
                // std::cout << "path tracing for id: " << deviceIdx << " took : " << duration.count() << "ms" << std::endl;

                tasks_[deviceIdx].time = duration.count();

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
    std::vector<RenderTask> &tasks_;
};
