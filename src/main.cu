#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <iostream>
#include <float.h>
#include <fstream>
#include <curand_kernel.h>
#include "obj_loader.h"
#include "LocalRenderer/Window.h"
#include "LocalRenderer/Renderer.h"
#include "cuda_utils.h"
#include "Profiling/GPUMonitor.h"
#include "DevicePathTracer.h"
#include <chrono>
#include <cmath>
#include "SafeQueue.h"
#include "GPUThread.h"
#include "helper_math.h"
#include "CameraParams.h"

double getRadians(double value) {
    return M_PI * value / 180.0;
}

int main() {
    int view_width = 600;
    int view_height = 600;
    int num_pixels = view_width * view_height;
    size_t fb_size = num_pixels*sizeof(uint8_t) * 3;
    uint8_t *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // Load object
    const char *file_path = "models/cornell-box.obj";
    obj_loader loader(file_path);

    DevicePathTracer pt0(0, loader, view_width, view_height);
    DevicePathTracer pt1(1, loader, view_width, view_height);
    CameraParams camParams;
    camParams.lookFrom = make_float3(-277.676, 157.279, 545.674);
    camParams.front = make_float3(-0.26, 0.121, -0.9922);

    Window window(view_width, view_height, "MultiGPU-PathTracer", camParams);
    Renderer renderer(window);

    MonitorThread monitor_thread_obj;
    std::thread monitor_thread(std::ref(monitor_thread_obj));

    int x, y;
    bool firstMouse;
    double lastX, lastY;
    double yaw = 0, pitch = 0;

    // ----------------------------------------------------------------- //
    // SafeQueue<RenderTask> queue;
    // RenderTask task;
    // GPUThread t0(0, loader, view_width, view_height, queue, fb);
    // GPUThread t1(1, loader, view_width, view_height, queue, fb);
    // std::thread gpu_0_thread(std::ref(t0));
    // std::thread gpu_1_thread(std::ref(t1));
    // ----------------------------------------------------------------- //

    while (!window.shouldClose()) {
        window.pollEvents();

        RenderTask task_0{300, 600, 0, 0};
        RenderTask task_1{300, 600, 300, 0};        

        pt0.setFront(camParams.front);
        pt0.setLookFrom(camParams.lookFrom);

        pt1.setFront(camParams.front);
        pt1.setLookFrom(camParams.lookFrom);

        auto start = std::chrono::high_resolution_clock::now();

        pt0.renderTaskAsync(task_0, fb);
        pt1.renderTaskAsync(task_1, fb);

        pt0.waitForRenderTask();
        pt1.waitForRenderTask();

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        // std::cout << "path tracing took: " << duration.count() << "ms" << std::endl;

        renderer.renderFrame(fb);
	    window.swapBuffers();	
	}

    monitor_thread_obj.safeTerminate();
    monitor_thread.join();

    checkCudaErrors(cudaFree(fb));
    return 0;
}
