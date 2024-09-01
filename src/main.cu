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
#include <ixwebsocket/IXWebSocketSendData.h>
#include <png.h>

// Custom user write function to store PNG data into a vector
void write_png_data_to_vector(png_structp png_ptr, png_bytep data, png_size_t length) {
    std::vector<uint8_t>* p = (std::vector<uint8_t>*)png_get_io_ptr(png_ptr);
    p->insert(p->end(), data, data + length);
}

// Function to create PNG image from pixel data vector
bool create_png(const std::vector<uint8_t>& pixels, int width, int height, std::vector<uint8_t>& png_output) {
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        std::cerr << "Failed to create png write struct" << std::endl;
        return false;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, nullptr);
        std::cerr << "Failed to create png info struct" << std::endl;
        return false;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        std::cerr << "Failed during png creation" << std::endl;
        return false;
    }

    // Set custom write function
    png_set_write_fn(png, &png_output, write_png_data_to_vector, nullptr);

    // Set the header
    png_set_IHDR(
        png,
        info,
        width, height,
        8,
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    // Allocate memory for rows of pointers to each row's data
    std::vector<uint8_t*> row_pointers(height);
    for (int y = 0; y < height; ++y) {
        row_pointers[y] = (uint8_t*)&pixels[y * width * 3];
    }

    png_write_image(png, row_pointers.data());
    png_write_end(png, nullptr);

    png_destroy_write_struct(&png, &info);

    return true;
}


int main(int argc, char **argv) {
    int view_width = 600;
    int view_height = 600;
    int num_pixels = view_width * view_height;
    size_t fb_size = num_pixels*sizeof(uint8_t) * 3;
    uint8_t *fb;
    checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

    // Load object
    std::string file_path;
    if (argc > 2) {
        file_path = argv[2];
    } else {
        file_path = "models/cubes.obj";
    }
    obj_loader loader(file_path.c_str());

    // Load job id
    std::string job_id;
    if (argc > 1) {
        job_id = argv[1];
    } else {
        job_id = "123";
    }

    ix::WebSocket webSocket;
    std::string url = "wss://pathtracing-relay-server.klatka.it/?path-tracing-job=true&jobId=";
    // std::string url = "ws://localhost:8080/?path-tracing-job=true&jobId=";
    url += job_id;
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

    std::vector<uint8_t> pixelData(num_pixels * 3, 0);

    while (true) {
        // window.pollEvents();

        auto start = std::chrono::high_resolution_clock::now();

        thread_semaphore.release(2*num_streams_per_gpu);
        while(completed_streams != num_streams_per_gpu * 2) {
            thread_cv.wait(lk);
        }
        completed_streams = 0;

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        // std::cout << "path tracing took: " << duration.count() << "ms" << std::endl;

        start = std::chrono::high_resolution_clock::now();

        for (int y = view_height - 1; y >= 0; --y) {
            for (int x = 0; x < view_width; ++x) {
                int fbi = (y * view_width + x) * 3;
                int pdi = ((view_height - y - 1) * view_width + x) * 3;
                pixelData[pdi] = fb[fbi];
                pixelData[pdi + 1] = fb[fbi + 1];
                pixelData[pdi + 2] = fb[fbi + 2];
            }
        }

        std::vector<uint8_t> png_output;
        if (create_png(pixelData, view_width, view_height, png_output)) {
            std::string messagePrefix = "JOB_MESSAGE#RENDER#";
            std::vector<uint8_t> messagePrefixVec(messagePrefix.begin(), messagePrefix.end());
            png_output.insert(png_output.begin(), messagePrefixVec.begin(), messagePrefixVec.end());
            ix::IXWebSocketSendData IXPixelData(png_output);
            stop = std::chrono::high_resolution_clock::now();

    // Calculate the duration in milliseconds
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

        // Print the duration in milliseconds
        std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;

            webSocket.sendBinary(IXPixelData);
        } 

        // renderer.renderFrame(fb);
	    // window.swapBuffers();	
	}

    monitor_thread_obj.safeTerminate();
    monitor_thread.join();

    checkCudaErrors(cudaFree(fb));
    return 0;
}
