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

int main() {
    int view_width = 1600;
    int view_height = 900;
    int num_pixels = view_width * view_height;
    size_t fb_size = num_pixels*sizeof(uint8_t) * 3;
    uint8_t *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    RenderTask task_0{800, 900, 0, 0};
    RenderTask task_1{800, 900, 800, 0};

    // Load object
    const char *file_path = "models/cubes.obj";
    obj_loader loader(file_path);

    DevicePathTracer pt0(0, loader, view_width, view_height);
    DevicePathTracer pt1(1, loader, view_width, view_height);

    Window window(view_width, view_height, "MultiGPU-PathTracer");
    Renderer renderer(window);

    MonitorThread monitor_thread_obj;
    std::thread monitor_thread(std::ref(monitor_thread_obj));

    while (!window.shouldClose()) {
        window.pollEvents();

        auto start = std::chrono::high_resolution_clock::now();

        pt0.renderTaskAsync(task_0, fb);
        pt1.renderTaskAsync(task_1, fb);
        
        pt0.waitForRenderTask();
        pt1.waitForRenderTask();

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
