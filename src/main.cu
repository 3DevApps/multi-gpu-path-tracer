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
#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXUserAgent.h>
#include <cmath>
#include "SafeQueue.h"
#include "GPUThread.h"
#include "helper_math.h"
#include "CameraParams.h"
#include "Scheduling/TaskGenerator.h"
#include <vector>

double getRadians(double value) {
    return M_PI * value / 180.0;
}

int main(int argc, char **argv) {
    int view_width = 600;
    int view_height = 600;
    int num_pixels = view_width * view_height;
    size_t fb_size = num_pixels*sizeof(uint8_t) * 3;
    uint8_t *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

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
    webSocket.setOnMessageCallback([](const ix::WebSocketMessagePtr& msg)
        {
            if (msg->type == ix::WebSocketMessageType::Message)
            {
                std::cout << "received message: " << msg->str << std::endl;
            }
            else if (msg->type == ix::WebSocketMessageType::Open)
            {
                std::cout << "Connection established" << std::endl;
            }
            else if (msg->type == ix::WebSocketMessageType::Error)
            {
                // Maybe SSL is not configured properly
                std::cout << "Connection error: " << msg->errorInfo.reason << std::endl;
            }
        }
    );

    // Now that our callback is setup, we can start our background thread and receive messages
    webSocket.start();

    // DevicePathTracer pt0(0, loader, view_width, view_height);
    // DevicePathTracer pt1(1, loader, view_width, view_height);
    CameraParams camParams;
    camParams.lookFrom = make_float3(-277.676, 157.279, 545.674);
    camParams.front = make_float3(-0.26, 0.121, -0.9922);

    Window window(view_width, view_height, "MultiGPU-PathTracer", camParams);
    Renderer renderer(window);

    MonitorThread monitor_thread_obj;
    std::thread monitor_thread(std::ref(monitor_thread_obj));

    // ----------------------------------------------------------------- //
    // SafeQueue<RenderTask> queue;
    // RenderTask task;
    // GPUThread t0(0, loader, view_width, view_height, queue, fb);
    // GPUThread t1(1, loader, view_width, view_height, queue, fb);
    // std::thread gpu_0_thread(std::ref(t0));
    // std::thread gpu_1_thread(std::ref(t1));
    // ----------------------------------------------------------------- //
    int num_streams_per_gpu = 4;
    TaskGenerator task_gen(view_width, view_height);

    std::vector<RenderTask> render_tasks;

    task_gen.generateTasks(32,32,render_tasks);
    SafeQueue<RenderTask> queue;
    
    std::condition_variable thread_cv;
    semaphore thread_semaphore(0);
    std::atomic_int completed_streams = 0;



    cudaStream_t stream_0[num_streams_per_gpu];
    cudaStream_t stream_1[num_streams_per_gpu];

    cudaEvent_t event_0[num_streams_per_gpu];
    cudaEvent_t event_1[num_streams_per_gpu];
    for (int i = 0; i < num_streams_per_gpu; i++) {
        cudaSetDevice(0);
        cudaStreamCreate(&stream_0[i]);
        cudaEventCreate(&event_0[i]);

        cudaSetDevice(1);
        cudaStreamCreate(&stream_1[i]);
        cudaEventCreate(&event_1[i]);
    }
    GPUThread t0_0(0,stream_0[0], loader, view_width, view_height, queue, fb, &thread_semaphore, &thread_cv, &completed_streams, camParams);
    GPUThread t0_1(0,stream_0[1], loader, view_width, view_height, queue, fb, &thread_semaphore, &thread_cv, &completed_streams, camParams);
    GPUThread t0_2(0,stream_0[2], loader, view_width, view_height, queue, fb, &thread_semaphore, &thread_cv, &completed_streams, camParams);
    GPUThread t0_3(0,stream_0[3], loader, view_width, view_height, queue, fb, &thread_semaphore, &thread_cv, &completed_streams, camParams);
    GPUThread t1_0(1,stream_1[0], loader, view_width, view_height, queue, fb, &thread_semaphore, &thread_cv, &completed_streams, camParams);
    GPUThread t1_1(1,stream_1[1], loader, view_width, view_height, queue, fb, &thread_semaphore, &thread_cv, &completed_streams, camParams);
    GPUThread t1_2(1,stream_1[2], loader, view_width, view_height, queue, fb, &thread_semaphore, &thread_cv, &completed_streams, camParams);
    GPUThread t1_3(1,stream_1[3], loader, view_width, view_height, queue, fb, &thread_semaphore, &thread_cv, &completed_streams, camParams);
    std::thread gpu_0_thread_0(std::ref(t0_0));
    std::thread gpu_0_thread_1(std::ref(t0_1));
    std::thread gpu_0_thread_2(std::ref(t0_2));
    std::thread gpu_0_thread_3(std::ref(t0_3));
    std::thread gpu_1_thread_0(std::ref(t1_0));
    std::thread gpu_1_thread_1(std::ref(t1_1));
    std::thread gpu_1_thread_2(std::ref(t1_2));
    std::thread gpu_1_thread_3(std::ref(t1_3));

    std::mutex m;
    std::unique_lock<std::mutex> lk(m);

    

    while (!window.shouldClose()) {
        window.pollEvents();
        // pt0.setFront(camParams.front);
        // pt0.setLookFrom(camParams.lookFrom);

        // pt1.setFront(camParams.front);
        // pt1.setLookFrom(camParams.lookFrom);

         // insert elements
        for (int i = 0; i < render_tasks.size(); i++) {
            queue.Produce(std::move(render_tasks[i]));
        }

        auto start = std::chrono::high_resolution_clock::now();

        thread_semaphore.release(2*num_streams_per_gpu);
        while(completed_streams != num_streams_per_gpu * 2) {
            thread_cv.wait(lk);
        }
        completed_streams = 0;

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        // std::cout << "path tracing took: " << duration.count() << "ms" << std::endl;

        std::string pixel_data = "";
        for (int j = view_height-1; j >= 0; j--) {
            for (int i = 0; i < view_width; i++) {
                size_t pixel_index = j*view_width + i;
                auto x = fb[3*pixel_index];
                auto y = fb[3*pixel_index+1];
                auto z = fb[3*pixel_index+2];
                pixel_data += std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ",";
            }
        }

        // Send pixel data to server
        webSocket.send("JOB_MESSAGE#" + std::string(job_id) + "#" + pixel_data);

        // renderer.renderFrame(fb);
	    // window.swapBuffers();	
	}

    monitor_thread_obj.safeTerminate();
    monitor_thread.join();

    checkCudaErrors(cudaFree(fb));
    return 0;
}
