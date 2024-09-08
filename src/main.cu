#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <iostream>
#include <float.h>
#include <fstream>
#include <curand_kernel.h>
#include "semaphore.h"
#include <mutex>
#include "obj_loader.h"
#include "LocalRenderer/Window.h"
#include "LocalRenderer/Renderer.h"
#include "cuda_utils.h"
#include "Profiling/GPUMonitor.h"
#include "DevicePathTracer.h"
#include <chrono>
#include <cmath>
#include "SafeQueue.h"
#include "StreamThread.h"
#include "helper_math.h"
#include "HostScene.h"
#include "Scheduling/TaskGenerator.h"
#include <vector>
#include "RendererConfig.h"
#include "Framebuffer.h"
#include "RenderManager.h"

int main() {
    RendererConfig config;

    CameraParams camParams;
    camParams.lookFrom = make_float3(0, 0, 0);
    camParams.front = make_float3(-0.26, 0.121, -0.9922);
    Window window(config.resolution.width, config.resolution.height, "MultiGPU-PathTracer", camParams);
    Renderer renderer(window);

    MonitorThread monitor_thread_obj;
    std::thread monitor_thread(std::ref(monitor_thread_obj));

    auto framebuffer = std::make_shared<Framebuffer>(config.resolution);
    RenderManager manager(config, camParams);


    int samples = 0;
    while (!window.shouldClose()) {
        window.pollEvents();

        switch (window.newEvent_) {
            case 1: {
                manager.setResolution({900, 900});
                window.newEvent_ = false;
                break;
            }
            case 2: {
                manager.setGpuAndStreamNumber(1, 6);
                window.newEvent_ = false;
                break;
            }
            case 3: {
                manager.setSamplesPerPixel(samples += 10);
                window.newEvent_ = false;
                break;
            }
            case 4: {
                manager.setRecursionDepth(100);
                std::cout << "recursion depth: 100" << std::endl;
                // devicePathTracerIdx0.setRecursionDepth(100);
                // devicePathTracerIdx1.setRecursionDepth(100);
                break;
            }
            case 5: {
                // std::cout << "obj changes to 2 cubes" << std::endl;
                // file_path = "models/cubes2.obj";
                // hScene.triangles = loader.load_triangles(file_path);
                // devicePathTracerIdx0.reloadWorld();
                // devicePathTracerIdx1.reloadWorld();
                break;
            }
            case 6: {
                // std::cout << "verical field of view changed to 80" << std::endl;
                // hScene.cameraParams.verticalFieldOfView = 80.0f;
                // devicePathTracerIdx0.reloadCamera();
                // devicePathTracerIdx1.reloadCamera();
                // break;
            }
            // case 7: {
            //     std::cout << "thread block size changed to 12 x 12" << std::endl;
            //     devicePathTracerIdx0.setThreadBlockSize({12, 12});
            //     devicePathTracerIdx0.setThreadBlockSize({12, 12});
            //     break;
            // }
            default: {}
        }
        window.newEvent_ = 0;

        auto start = std::chrono::high_resolution_clock::now();

        manager.renderFrame();

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "path tracing took: " << duration.count() << "ms" << std::endl;

        renderer.renderFrame(manager.getCurrentFrame(), manager.getCurrentFrameWidth(), manager.getCurrentFrameHeight());
	    window.swapBuffers(); 	
	}

    manager.reset();
    monitor_thread_obj.safeTerminate();
    monitor_thread.join();
    return 0;
}