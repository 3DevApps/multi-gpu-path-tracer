#pragma once

#include <cmath>
#include "EventHandler.h"
#include "../../../RenderManager.h"
#include "../../../PixelDataEncoder/PNGEncoder.h"

class RenderManagerEventHander: EventHandler {
    public:
        RenderManagerEventHander(RenderManager& manager, HostScene &hScene) : manager(manager), hScene(hScene) {};

        void handleEvent(const std::string& message) override {
            auto sepPos = message.find("#");
            auto command = message.substr(0, sepPos);
            auto parsedRawData = message.substr(sepPos+1);

            if (command == "GPU_NUMBER") {
                auto gpuNumber = std::stoi(parsedRawData);
                manager.setGpuNumber(gpuNumber);
            } else if (command == "STREAMS_PER_GPU") {
                auto streamsPerGpu = std::stoi(parsedRawData);
                manager.setStreamsPerGpu(streamsPerGpu);
            } else if (command == "SAMPLES_PER_PIXEL") {
                auto samplesPerPixel = std::stoi(parsedRawData);
                manager.setSamplesPerPixel(samplesPerPixel);
            } else if (command == "RECURSION_DEPTH") {
                auto recursionDepth = std::stoi(parsedRawData);
                manager.setRecursionDepth(recursionDepth);
            } else if (command == "THREAD_BLOCK_SIZE") {
                auto sepPos = parsedRawData.find("#");
                auto threadBlockSizeX = std::stoi(parsedRawData.substr(0, sepPos));
                auto threadBlockSizeY = std::stoi(parsedRawData.substr(sepPos+1));
                dim3 threadBlockSize(threadBlockSizeX, threadBlockSizeY);
                manager.setThreadBlockSize(threadBlockSize);
            } else if (command == "IMAGE_RESOLUTION") {
                auto sepPos = parsedRawData.find("#");
                unsigned int width = std::stoi(parsedRawData.substr(0, sepPos));
                unsigned int height = std::stoi(parsedRawData.substr(sepPos+1));
                manager.setResolution({width, height});
            } else if (command == "LOAD_UPLOADED_SCENE") {
                hScene.loadUploadedScene();
            } else if (command == "DOWNLOAD_SCENE_SNAPSHOT") {
                // TODO: use processFrame from RemoteRenderer and send the data
            }
        };
        
        std::string getEventName() override {
            return "RENDERER_PARAMETER";
        };

    private:
        RenderManager& manager;
        HostScene& hScene;
        PNGEncoder pngEncoder;
};