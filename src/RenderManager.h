#pragma once

#include <stdio.h>
#include <iostream>
#include <float.h>
#include <fstream>
#include <curand_kernel.h>
#include "semaphore.h"
#include <mutex>
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

class RenderManager { 
public:
        RenderManager(RendererConfig &config, HostScene &hScene, CameraConfig& cameraConfig, SceneLoader& sceneLoader) : config_{config}, hScene_{hScene}, lk_{m_}, cameraConfig_{cameraConfig}, sceneLoader_{sceneLoader} {
        newConfig_ = config_;
        renderTasks_ = taskGen_.generateEqualTasks(config.gpuNumber * config.streamsPerGpu, config.resolution.width, config.resolution.height);
        framebuffer_ = std::make_shared<Framebuffer>(config.resolution);
        setup();
    }

    void reset() {
        for (int i = 0; i < config_.gpuNumber; i++) {
            for (int j = 0; j < config_.streamsPerGpu; j++) { 
                streamThreads_[i][j]->finish();
            }
        }
        queue_.Finish();
        for (int i = 0; i < config_.gpuNumber; i++) {
            for (int j = 0; j < config_.streamsPerGpu; j++) { 
                streamThreads_[i][j]->join();
            }
        }

        streamThreads_ = {};
        devicePathTracers_ = {};
    }

    void setup() {
        for (int i = 0; i < config_.gpuNumber; i++) {
            devicePathTracers_.push_back(std::make_shared<DevicePathTracer>(
                i, 
                config_.samplesPerPixel, 
                config_.recursionDepth, 
                config_.threadBlockSize, 
                hScene_, 
                framebuffer_,
                cameraConfig_
            ));

            streamThreads_.push_back({}); 
            streamThreads_[i].reserve(config_.streamsPerGpu);
            for (int j = 0; j < config_.streamsPerGpu; j++) {
                streamThreads_[i].push_back(std::make_shared<StreamThread>(
                    i, 
                    queue_, 
                    &threadSemaphore_, 
                    &threadCv_, 
                    &completedStreams_, 
                    devicePathTracers_[i]
                ));
                streamThreads_[i][j]->start();
            }
        }

        renderTasks_ = taskGen_.generateEqualTasks(
            config_.gpuNumber * config_.streamsPerGpu,
            framebuffer_->getResolution().width, 
            framebuffer_->getResolution().height);
    }

    void updatePathTracingParamsIfNeeded() {
        if (!shouldUpdatePathTracerParams) {
            return;
        }
        shouldUpdatePathTracerParams = false;

        if (config_.gpuNumber != newConfig_.gpuNumber || config_.streamsPerGpu != newConfig_.streamsPerGpu) {
            reset();
            config_.gpuNumber = newConfig_.gpuNumber;
            config_.streamsPerGpu = newConfig_.streamsPerGpu;
            if (shouldReloadScene) {
                hScene_ = sceneLoader_.load(config_.modelPath);
                shouldReloadScene = false;
            }
            setup();
        } 

        if (config_.resolution.width != newConfig_.resolution.width || config_.resolution.height != newConfig_.resolution.height) {
            config_.resolution = newConfig_.resolution;
            framebuffer_->setResolution(config_.resolution);
            for (const auto & dpt : devicePathTracers_) {
                dpt->setFramebuffer(framebuffer_);
            }
            renderTasks_ = taskGen_.generateEqualTasks(config_.gpuNumber * config_.streamsPerGpu, config_.resolution.width, config_.resolution.height);
        }

        if (config_.samplesPerPixel != newConfig_.samplesPerPixel) {
            config_.samplesPerPixel = newConfig_.samplesPerPixel;
            for (const auto & dpt : devicePathTracers_) {
                dpt->setSamplesPerPixel(config_.samplesPerPixel);
            }
        }

        if (config_.recursionDepth != newConfig_.recursionDepth) {
            config_.recursionDepth = newConfig_.recursionDepth;
            for (const auto & dpt : devicePathTracers_) {
                dpt->setRecursionDepth(config_.recursionDepth);
            }
        }

        if (config_.threadBlockSize.x != newConfig_.threadBlockSize.x || config_.threadBlockSize.y != newConfig_.threadBlockSize.y) {
            config_.threadBlockSize = newConfig_.threadBlockSize;
            for (const auto & dpt : devicePathTracers_) {
                dpt->setThreadBlockSize(config_.threadBlockSize);
            }
        }
    }

    void setGpuNumber(int gpuNumber) {
        newConfig_.gpuNumber = gpuNumber;
        shouldUpdatePathTracerParams = true;
    }

    void setStreamsPerGpu(int streamsPerGpu) {
        newConfig_.streamsPerGpu = streamsPerGpu;
        shouldUpdatePathTracerParams = true;
    }

    void setGpuAndStreamNumber(int gpuNumber, int streamsPerGpu) {
        newConfig_.gpuNumber = gpuNumber;
        newConfig_.streamsPerGpu = streamsPerGpu;
        shouldUpdatePathTracerParams = true;
    }

    void setResolution(Resolution res) {
        newConfig_.resolution = res;
        shouldUpdatePathTracerParams = true;
    }

    void setSamplesPerPixel(unsigned int samples) {
        newConfig_.samplesPerPixel = samples;
        shouldUpdatePathTracerParams = true;
    }

    void setRecursionDepth(unsigned int depth) {
        newConfig_.recursionDepth = depth;
        shouldUpdatePathTracerParams = true;
    }

    void setThreadBlockSize(dim3 threadBlockSize) {
        newConfig_.threadBlockSize = threadBlockSize;
        shouldUpdatePathTracerParams = true;
    }

    void reloadWorldIfNeeded() {
        if (!shouldReloadWorld) {
            return;
        }
        shouldReloadWorld = false;

        for (const auto & dpt : devicePathTracers_) {
            dpt->reloadWorld();
        }
    }

    void renderFrame() {
        updatePathTracingParamsIfNeeded();
        reloadWorldIfNeeded();

        for (int i = 0; i < renderTasks_.size(); i++) {
            queue_.Produce(std::move(renderTasks_[i]));
        }

        while(completedStreams_ != config_.streamsPerGpu * config_.gpuNumber) {
            threadCv_.wait(lk_);
        }
        completedStreams_ = 0;
    }

    std::shared_ptr<Framebuffer> getFramebuffer() {
        return framebuffer_;
    }

    uint8_t* getCurrentFrame() {
        return framebuffer_->getRGBPtr();
    }

    unsigned int getCurrentFrameWidth() {
        return framebuffer_->getResolution().width;
    }

    unsigned int getCurrentFrameHeight() {
        return framebuffer_->getResolution().height;
    }

    void reloadScene() {
        shouldReloadScene = true;
    }

    void updatePrimitives() {
        shouldReloadWorld = true;
    }

private:
    std::vector<std::shared_ptr<DevicePathTracer>> devicePathTracers_{};
    TaskGenerator taskGen_{};
    SafeQueue<RenderTask> queue_{};
    std::condition_variable threadCv_{};
    semaphore threadSemaphore_{0};
    std::atomic_int completedStreams_ = 0;
    std::shared_ptr<Framebuffer> framebuffer_;
    HostScene& hScene_; 
    std::mutex m_;
    std::unique_lock<std::mutex> lk_;
    std::vector<RenderTask> renderTasks_{};
    std::vector<std::vector<std::shared_ptr<StreamThread>>> streamThreads_{};    
    RendererConfig& config_;
    RendererConfig newConfig_{};
    bool shouldUpdatePathTracerParams = false;
    bool shouldReloadWorld = false;
    bool shouldReloadScene = false;
    CameraConfig& cameraConfig_;
    SceneLoader& sceneLoader_;
};
