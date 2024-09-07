
#include <nvml.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include "semaphore.h"
#include "DevicePathTracer.h"
#include "SafeQueue.h"



class StreamThread {
public:
    StreamThread(int device_idx, SafeQueue<RenderTask> &queue, uint8_t *fb, semaphore* thread_semaphore, std::condition_variable* thread_cv, std::atomic_int* completed_streams, HostScene& hostScene, DevicePathTracer& devicePathTracer):
        devicePathTracer{devicePathTracer},
        queue{queue},
        fb{fb},
        thread_semaphore{thread_semaphore},
        thread_cv{thread_cv},
        completed_streams{completed_streams} {
        
        cudaSetDevice(device_idx);
        cudaStreamCreate(&stream);
        cudaEventCreate(&event);
    }

    void operator()() {
        RenderTask task;
        thread_semaphore->acquire();
        while(true){
            while(queue.Consume(task)) {
                // Process the message
                devicePathTracer.renderTaskAsync(task, fb, stream);
                cudaStreamSynchronize(stream);
            }
            completed_streams->fetch_add(1);
            thread_cv->notify_all();
            thread_semaphore->acquire();
        }
    }


    void safeTerminate() {
        shouldTerminate = true;
    }

private:
    std::atomic_bool shouldTerminate = false;
    DevicePathTracer devicePathTracer;
    SafeQueue<RenderTask> &queue;
    uint8_t *fb;
    semaphore* thread_semaphore;
    std::condition_variable* thread_cv;
    std::atomic_int* completed_streams;
    cudaEvent_t event;
    cudaStream_t stream;
};
