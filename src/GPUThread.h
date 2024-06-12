
#include <nvml.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include "DevicePathTracer.h"
#include "SafeQueue.h"


class GPUThread {
public:
    GPUThread(int device_idx, obj_loader &loader, int view_width, int view_height, SafeQueue<RenderTask> &queue, uint8_t *fb) :
        devicePathTracer{device_idx, loader, view_width, view_height},
        queue{queue},
        fb{fb} {}

    void operator()() {
        RenderTask task;
        while(queue.ConsumeSync(task)) {
		    // Process the message
            devicePathTracer.renderTaskAsync(task, fb);
            devicePathTracer.waitForRenderTask();
	    }
    }

    // void sync() {
    //     devicePathTracer.waitForRenderTask();
    // }

    void safeTerminate() {
        shouldTerminate = true;
    }

private:
    std::atomic_bool shouldTerminate = false;
    DevicePathTracer devicePathTracer;
    SafeQueue<RenderTask> &queue;
    uint8_t *fb;
};
