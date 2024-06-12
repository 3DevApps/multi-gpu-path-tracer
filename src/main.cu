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

    // RenderTask task_0{800, 900, 0, 0};
    // RenderTask task_1{800, 900, 800, 0};

    // Load object
    const char *file_path = "models/cornell-box.obj";
    obj_loader loader(file_path);

    DevicePathTracer pt0(0, loader, view_width, view_height);
    DevicePathTracer pt1(1, loader, view_width, view_height);

    Window window(view_width, view_height, "MultiGPU-PathTracer");
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

        // SafeQueue<RenderTask> queue;

        // insert elements
        
        window.getMousePos(x, y);

        if (firstMouse)
        {
            lastX = (double)x;
            lastY = (double)y;
            firstMouse = false;
        }

        double xoffset = (double)x - lastX;
        double yoffset = lastY - (double)y; 
        lastX = x;
        lastY = y;

        double sensitivity = 0.5f;
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        yaw += xoffset;
        pitch += yoffset;

        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;

        float3 lookat = make_float3(cos(getRadians(yaw)) * cos(getRadians(pitch)), 
                                   sin(getRadians(pitch)), 
                                   sin(getRadians(yaw)) * cos(getRadians(pitch)));

        pt0.setLookAt(lookat);
        pt1.setLookAt(lookat);

        auto start = std::chrono::high_resolution_clock::now();

        pt0.renderTaskAsync(task_0, fb);
        pt1.renderTaskAsync(task_1, fb);

        pt0.waitForRenderTask();
        pt1.waitForRenderTask();

        // queue.Produce(std::move(task_0));
        // queue.Produce(std::move(task_1));

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "path tracing took: " << duration.count() << "ms" << std::endl;

        renderer.renderFrame(fb);
	    window.swapBuffers();	
	}

    monitor_thread_obj.safeTerminate();
    monitor_thread.join();

    checkCudaErrors(cudaFree(fb));
    return 0;
}
