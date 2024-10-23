#include <stdio.h>
#include <iostream>
#include <float.h>
#include <fstream>
#include "semaphore.h"
#include <mutex>
#include "Profiling/GPUMonitor.h"
#include "DevicePathTracer.h"
#include <chrono>
#include <cmath>
#include "SafeQueue.h"
#include "StreamThread.h"
#include "HostScene.h"
#include "Scheduling/TaskGenerator.h"
#include <vector>
#include "PixelDataEncoder/PixelDataEncoder.h"
#include "PixelDataEncoder/JPEGEncoder.h"
#include "PixelDataEncoder/PNGEncoder.h"
#include "ArgumentLoader.h"
#include "RendererConfig.h"
#include "Framebuffer.h"
#include "RenderManager.h"
#include "Renderer/Renderer.h"
#include "CameraParams.h"
#ifdef USE_LOCAL_RENDERER
#include "Renderer/LocalRenderer/Window.h"
#include "Renderer/LocalRenderer/LocalRenderer.h"
#else
#include "Renderer/RemoteRenderer/RemoteRenderer.h"
#include "Renderer/RemoteRenderer/RemoteEventHandlers/RemoteEventHandlers.h"
#endif

int main(int argc, char** argv) {
    RendererConfig config; 
    SceneLoader sceneLoader;
    ArgumentLoader argLoader(argc, argv);
    argLoader.loadArguments(config);


    CameraParams cameraParams(make_float3(6, 6, 6), make_float3(-6, -6, -6));
    HostScene hScene = sceneLoader.load(config.path);
    
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

    auto start_init = std::chrono::high_resolution_clock::now();

    RenderManager manager(config, hScene, cameraParams);

    auto stop_init = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop_init - start_init);
    std::cout << "initializing in: " << duration.count() << "ms" << std::endl;

    #ifdef USE_LOCAL_RENDERER
    Window window(config.resolution.width, config.resolution.height, "MultiGPU-PathTracer", cameraParams);
    LocalRenderer localRenderer(window);
    Renderer &renderer = localRenderer;
    #else
    RemoteRenderer remoteRenderer(config.jobId, config);
    RemoteEventHandlers remoteEventHandlers(remoteRenderer, manager, hScene, cameraParams);
    Renderer &renderer = remoteRenderer;
    #endif

    MonitorThread monitor_thread_obj(renderer);
    std::thread monitor_thread(std::ref(monitor_thread_obj));

    while (!renderer.shouldStopRendering()) {
        auto start = std::chrono::high_resolution_clock::now();
        manager.renderFrame();
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "path tracing took: " << duration.count() << "ms" << std::endl;
        renderer.renderFrame(manager.getCurrentFrame());
	}

    manager.reset();
    monitor_thread_obj.safeTerminate();
    monitor_thread.join();
    return 0;
}