
#include <nvml.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include "semaphore.h"
#include "DevicePathTracer.h"
#include "SafeQueue.h"



class GPUThread {
public:
    GPUThread(int device_idx,cudaStream_t stream_num,obj_loader &loader, int view_width, int view_height, SafeQueue<RenderTask> &queue, uint8_t *fb, semaphore* thread_semaphore, std::condition_variable* thread_cv, std::atomic_int* completed_streams, CameraParams& camParams):
        devicePathTracer{device_idx, loader, view_width, view_height},
        queue{queue},
        fb{fb},
        stream_num{stream_num},
        thread_semaphore{thread_semaphore},
        thread_cv{thread_cv},
        completed_streams{completed_streams},
        camParams{camParams}
        {}

    void operator()() {
        RenderTask task;
        thread_semaphore->acquire();
        while(true){
            while(queue.Consume(task)) {
                // Process the message
                devicePathTracer.setFront(camParams.front);
                devicePathTracer.setLookFrom(camParams.lookFrom);
                devicePathTracer.renderTaskAsync(task, fb, stream_num);
                cudaStreamSynchronize(stream_num);
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
    cudaStream_t stream_num;
    semaphore* thread_semaphore;
    std::condition_variable* thread_cv;
    std::atomic_int* completed_streams;
    CameraParams& camParams;
};
