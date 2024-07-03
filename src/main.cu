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
#include "rtc/rtc.hpp"
#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXUserAgent.h>

int main(int argc, char **argv) {
    int view_width = 1600;
    int view_height = 900;
    int num_pixels = view_width * view_height;
    size_t fb_size = num_pixels*sizeof(uint8_t) * 3;
    uint8_t *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    RenderTask task_0{800, 900, 0, 0};
    RenderTask task_1{800, 900, 800, 0};

    // Load object
    char *file_path;
    if (argc > 2) {
        file_path = argv[2];
    } else {
        file_path = "models/cubes.obj";
    }
    obj_loader loader(file_path);

    // Load job id
    char *job_id;
    if (argc > 1) {
        job_id = argv[1];
    } else {
        job_id = "0";
    }

    ix::WebSocket webSocket;
    std::string url("wss://pathtracing-relay-server.klatka.it");
    webSocket.setUrl(url);

    // // Setup a callback to be fired (in a background thread, watch out for race conditions !)
    // // when a message or an event (open, close, error) is received
    // webSocket.setOnMessageCallback([](const ix::WebSocketMessagePtr& msg)
    //     {
    //         if (msg->type == ix::WebSocketMessageType::Message)
    //         {
    //             std::cout << "received message: " << msg->str << std::endl;
    //             std::cout << "> " << std::flush;
    //         }
    //         else if (msg->type == ix::WebSocketMessageType::Open)
    //         {
    //             std::cout << "Connection established" << std::endl;
    //             std::cout << "> " << std::flush;
    //         }
    //         else if (msg->type == ix::WebSocketMessageType::Error)
    //         {
    //             // Maybe SSL is not configured properly
    //             std::cout << "Connection error: " << msg->errorInfo.reason << std::endl;
    //             std::cout << "> " << std::flush;
    //         }
    //     }
    // );

    // Now that our callback is setup, we can start our background thread and receive messages
    webSocket.start();

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

        std::string pixel_data = "";
        for (int j = view_height-1; j >= 0; j--) {
            for (int i = 0; i < view_width; i++) {
                size_t pixel_index = j*view_width + i;
                int3 color = make_int3(255.99*fb[pixel_index].x, 255.99*fb[pixel_index].y, 255.99*fb[pixel_index].z);
                pixel_data += std::to_string(color.x) + "," + std::to_string(color.y) + "," + std::to_string(color.z) + ",";
            }
        }

        // Send pixel data to server
        webSocket.send("JOB_MESSAGE#" + std::string(job_id) + "#" + pixel_data);

        renderer.renderFrame(fb);
	    window.swapBuffers();	
	}

    monitor_thread_obj.safeTerminate();
    monitor_thread.join();

    checkCudaErrors(cudaFree(fb));
    return 0;
}
