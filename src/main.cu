#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <iostream>
#include <float.h>
#include <fstream>
// #include <curand_kernel.h>
#include "semaphore.h"
#include <mutex>
#include "Renderer/LocalRenderer/Window.h"
#include "Renderer/LocalRenderer/LocalRenderer.h"
// #include "cuda_utils.h"
#include "Profiling/GPUMonitor.h"
#include "DevicePathTracer.h"
#include <chrono>
#include <cmath>
#include "SafeQueue.h"
#include "StreamThread.h"
// #include "helper_math.h"
#include "HostScene.h"
#include "Scheduling/TaskGenerator.h"
#include <vector>
#include "PixelDataEncoder/PixelDataEncoder.h"
#include "PixelDataEncoder/JPEGEncoder.h"
#include "PixelDataEncoder/PNGEncoder.h"
#include "ArgumentLoader.h"
#include "Renderer/RemoteRenderer/RemoteRenderer.h"
#include "Renderer/Renderer.h"
#include "Renderer/RemoteRenderer/RemoteEventHandlers/RemoteEventHandlers.h"
#include "RendererConfig.h"
#include "Framebuffer.h"
#include "RenderManager.h"

int main(int argc, char** argv) {
    ArgumentLoader argLoader(argc, argv);
    auto args = argLoader.loadAndGetArguments();

    RendererConfig config; 
    HostScene hScene(config.objPath, make_float3(0.05, 0.05, 0.05), make_float3(-0.05, -0.05, -0.05));
    Window window(config.resolution.width, config.resolution.height, "MultiGPU-PathTracer", hScene.cameraParams);
    RenderManager manager(config, hScene);
    /*
    changing parameters:
    manager.setSamplesPerPixel(30);
    manager.setRecursionDepth(5);
    manager.setGpuAndStreamNumber(1, 6);
    manager.setResolution({900, 900}); // TODO: make rendered frame resolution independent from window size
    manager.setThreadBlockSize({16, 16});

    hScene.loadTriangles("path/to/obj");
    hScene.setVFOV(60.0f);
    hScene.setHFOV(60.0f);
    hScene.setCameraLookFrom(make_float3(1, 1, 1));
    hScene.setCameraFront(make_float3(1, 1, 1));
    */

    LocalRenderer localRenderer(window);
    RemoteRenderer remoteRenderer(args.jobId, config.resolution.width, config.resolution.height);
    RemoteEventHandlers remoteEventHandlers(remoteRenderer, hScene.cameraParams);
    Renderer &renderer = localRenderer;

    MonitorThread monitor_thread_obj(renderer);
    std::thread monitor_thread(std::ref(monitor_thread_obj));

    while (!renderer.shouldStopRendering()) {
        
        auto start = std::chrono::high_resolution_clock::now();
        manager.renderFrame();
        auto stop = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "path tracing took: " << duration.count() << "ms" << std::endl;

        renderer.renderFrame(manager.getCurrentFrame());
	    window.swapBuffers(); 	        
	}

    manager.reset();
    monitor_thread_obj.safeTerminate();
    monitor_thread.join();
    return 0;
}