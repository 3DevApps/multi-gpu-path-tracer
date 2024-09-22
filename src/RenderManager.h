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


class RenderManager : PrimitivesObserver { 
public:
        RenderManager(RendererConfig &config, HostScene &hScene) : hScene_{hScene}, lk_{m_} {
        hScene_.registerPrimitivesObserver(this);

        streamsPerGpu_ = config.streamsPerGpu;
        gpuNumber_ = config.gpuNumber;
        
        samplesPerPixel_ = config.samplesPerPixel;
        recursionDepth_ = config.recursionDepth;
        threadBlockSize_ = config.threadBlockSize;

        renderTasks_ = taskGen_.generateEqualTasks(config.gpuNumber * config.streamsPerGpu, config.resolution.width, config.resolution.height);
        framebuffer_ = std::make_shared<Framebuffer>(config.resolution);
        setup();
    }

    ~RenderManager() {
        hScene_.removePrimitivesObserver(this);
    }

    void reset() {
        for (int i = 0; i < gpuNumber_; i++) {
            for (int j = 0; j < streamsPerGpu_; j++) { 
                streamThreads_[i][j]->finish();
            }
        }
        queue_.Finish();
        for (int i = 0; i < gpuNumber_; i++) {
            for (int j = 0; j < streamsPerGpu_; j++) { 
                streamThreads_[i][j]->join();
            }
        }

        devicePathTracers_ = {};
        streamThreads_ = {};
    }

    void setup() {
        for (int i = 0; i < gpuNumber_; i++) {
            devicePathTracers_.push_back(std::make_shared<DevicePathTracer>(
                i, 
                samplesPerPixel_, 
                recursionDepth_, 
                threadBlockSize_, 
                hScene_, 
                framebuffer_
            ));

            streamThreads_.push_back({}); 
            streamThreads_[i].reserve(streamsPerGpu_);
            for (int j = 0; j < streamsPerGpu_; j++) {
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
            gpuNumber_ * streamsPerGpu_,
            framebuffer_->getResolution().width, 
            framebuffer_->getResolution().height);
    }

    void setGpuNumber(int gpuNumber) {
        reset();
        gpuNumber_ = gpuNumber;
        setup();
    }

    void setStreamsPerGpu(int streamsPerGpu) {
        reset();
        streamsPerGpu_ = streamsPerGpu;
        setup();
    }

    void setGpuAndStreamNumber(int gpuNumber, int streamsPerGpu) {
        reset();
        gpuNumber_ = gpuNumber;
        streamsPerGpu_ = streamsPerGpu;
        setup();
    }

    void setResolution(Resolution res) {
        framebuffer_->setResolution(res);
        for (const auto & dpt : devicePathTracers_) {
            dpt->setFramebuffer(framebuffer_);
        }
        renderTasks_ = taskGen_.generateEqualTasks(gpuNumber_ * streamsPerGpu_, res.width, res.height);
    }

    void setSamplesPerPixel(unsigned int samples) {
        samplesPerPixel_ = samples;
        for (const auto & dpt : devicePathTracers_) {
            dpt->setSamplesPerPixel(samples);
        }
    }

    void setRecursionDepth(unsigned int depth) {
        recursionDepth_ = depth;
        for (const auto & dpt : devicePathTracers_) {
            dpt->setRecursionDepth(depth); 
        }
    }

    void setThreadBlockSize(dim3 threadBlockSize) {
        for (const auto & dpt : devicePathTracers_) {
            dpt->setThreadBlockSize(threadBlockSize); 
        }
    }

    void reloadWorld() {
        for (const auto & dpt : devicePathTracers_) {
            dpt->reloadWorld();
        }
    }

    void renderFrame() {
        for (int i = 0; i < renderTasks_.size(); i++) {
            queue_.Produce(std::move(renderTasks_[i]));
        }

        while(completedStreams_ != streamsPerGpu_ * gpuNumber_) {
            threadCv_.wait(lk_);
        }
        completedStreams_ = 0;
    }

    uint8_t* getCurrentFrame() {
        return framebuffer_->getPtr();
    }

    unsigned int getCurrentFrameWidth() {
        return framebuffer_->getResolution().width;
    }

    unsigned int getCurrentFrameHeight() {
        return framebuffer_->getResolution().height;
    }

    void updatePrimitives() {
        reloadWorld();
    }

private:
    std::vector<std::shared_ptr<DevicePathTracer>> devicePathTracers_{};
    TaskGenerator taskGen_{};
    SafeQueue<RenderTask> queue_{};
    std::condition_variable threadCv_{};
    semaphore threadSemaphore_{0};
    std::atomic_int completedStreams_ = 0;
    std::shared_ptr<Framebuffer> framebuffer_;
    int gpuNumber_;
    int streamsPerGpu_;
    HostScene& hScene_; 
    std::mutex m_;
    std::unique_lock<std::mutex> lk_;
    std::vector<RenderTask> renderTasks_{};
    std::vector<std::vector<std::shared_ptr<StreamThread>>> streamThreads_{};    
    unsigned int recursionDepth_;
    unsigned int samplesPerPixel_;
    dim3 threadBlockSize_;
};
