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
#include "barrier.h"

class RenderManager { 
public:
        RenderManager(RendererConfig &config, HostScene &hScene, CameraConfig& cameraConfig) : config_{config}, hScene_{hScene}, cameraConfig_{cameraConfig}, barrier_{config.gpuNumber * config.streamsPerGpu + 1} {
        newConfig_ = config_;
        renderTasks_ = taskGen_.generateEqualTasks(config.gpuNumber * config.streamsPerGpu, config.resolution.width, config.resolution.height);
        framebuffer_ = std::make_shared<Framebuffer>(config.resolution);
        setup();

        std::cout << "number of threads: " << config.gpuNumber * config.streamsPerGpu << std::endl;

        int maxRowTask = 2;
        int task = 0;
        while (task < config.gpuNumber * config.streamsPerGpu) {
            taskLayout.push_back({});
            int rowTask = 0;
            while (rowTask < maxRowTask && task < config.gpuNumber * config.streamsPerGpu) {
                taskLayout[taskLayout.size() - 1].push_back(task);
                task++;
                rowTask++;
            }
        }

        printf("layout: \n");
        for (int i = 0; i < taskLayout.size(); i++) {
            for (int j = 0; j < taskLayout[i].size(); j++) {
                printf("%d ", taskLayout[i][j]);
            }
            printf("\n");
        }

        offsets = std::vector<int>(taskLayout.size());
        heights = std::vector<int>(taskLayout.size());
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
                    devicePathTracers_[i],
                    renderTasks_,
                    &shouldThreadStartCV_,
                    threadStartMutex,
                    shouldThreadStart,
                    barrier_
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

    void adjustTasks() {
        std::vector<std::vector<int>> hDivs{};
        std::vector<int> vDivs = getDivPointsVert();
        for (int i = 0; i < taskLayout.size(); i++) {
            hDivs.push_back(getDivPointsHoriz(i));
        }

        // adjust widths and x offsets of tasks
        for (int rowIdx = 0; rowIdx < taskLayout.size(); rowIdx++) {
            renderTasks_[taskLayout[rowIdx][0]].offset_x = 0;
            // renderTasks_[taskLayout[rowIdx][0]].width = framebuffer_->getResolution().width;
            int i = 0;
            while (i < hDivs[rowIdx].size()) {
                renderTasks_[taskLayout[rowIdx][i]].width = hDivs[rowIdx][i] - renderTasks_[taskLayout[rowIdx][i]].offset_x;
                renderTasks_[taskLayout[rowIdx][i + 1]].offset_x = hDivs[rowIdx][i];
                i++;
            }
            renderTasks_[taskLayout[rowIdx][i]].width = framebuffer_->getResolution().width - renderTasks_[taskLayout[rowIdx][i]].offset_x;
        }


        // adjust heights and y offset
        offsets[0] = 0;
        heights[0] = framebuffer_->getResolution().height;
        int i = 0;
        while (i < vDivs.size()) {
            heights[i] = vDivs[i] - offsets[i];
            offsets[i + 1] = vDivs[i];
            i++;
        }
        heights[i] = framebuffer_->getResolution().height - offsets[i];


        for (int i = 0; i < taskLayout.size(); i++) {
            for (int j = 0; j < taskLayout[i].size(); j++) {
                renderTasks_[taskLayout[i][j]].offset_y = offsets[i];
                renderTasks_[taskLayout[i][j]].height = heights[i];
            }
        }
    }

    bool first = true;

    void renderFrame() {
        updatePathTracingParamsIfNeeded();
        reloadWorldIfNeeded();

        if (first) {
            first = false;
        }
        else {
            adjustTasks();
        }

        printf("after adjust\n");
        

        barrier_.wait();

        //path tracer works here...

        barrier_.wait(); // wair for rendering end

        printf("after rendering\n");
            
        std::cout << "all finished." << std::endl;
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
        shouldReloadWorld = true;
    }


    std::vector<float> getBlockTimes(std::vector<int> &taskTimes, std::vector<int> &taskLengths, int length) { // return vector with block times
        int blockSize = 8;
        int blockCount = length / blockSize;//
        std::vector<float> blockTimes(blockCount);

        int it = 0;
        int renderTaskBlocks;
        for(int i = 0; i < taskTimes.size(); i++) {
            renderTaskBlocks = taskLengths[i] / blockSize;
            for (int j = 0; j < renderTaskBlocks; j++) {
                blockTimes[it++] = taskTimes[i] / (float)renderTaskBlocks;
            }
        }
        return blockTimes;
    }

    std::vector<int> getDivPoints(std::vector<float>&blockTimes, int taskCount, float targetTime) {
        float current = 0;
        int blockSize = 8;
        std::vector<int> divPoints;
        for (int i = 1; i < blockTimes.size(); i++) {
            if (blockTimes.size() - i == taskCount - 1 - divPoints.size()) {
                divPoints.push_back((i - 1) * blockSize);
                current = 0;
            }
            else if (current + blockTimes[i] > targetTime) {
                divPoints.push_back((i - 1) * blockSize);
                current = 0;
            }
            else {
                current += blockTimes[i];
            }
        }
        return divPoints;
    }

    std::vector<int> getDivPointsVert() {
        float sum = 0;
        int blockSize = 8;
        std::vector<int> rowTimes; 

        for(int i = 0; i < taskLayout.size(); i++) {
            int s = 0;
            for (int j = 0; j < taskLayout[i].size(); j++) {
                s += renderTasks_[taskLayout[i][j]].time;
            }

            rowTimes.push_back(s); // adjust horizontal and get time
            printf("rowTimes: %d\n", rowTimes[i]);
            sum += rowTimes[i];
        }

        float targetTime = sum / (float)taskLayout.size();
        //new block times count
        std::vector<int> currentTimes;
        std::vector<int> currentLengths;
        for (int i = 0; i < taskLayout.size(); i++) {
            currentTimes.push_back(rowTimes[i]);
            currentLengths.push_back(renderTasks_[taskLayout[i][0]].height);
        }
        std::vector<float> blockTimes = getBlockTimes(currentTimes, currentLengths, framebuffer_->getResolution().height);
        std::vector<int> vertDiv = getDivPoints(blockTimes, taskLayout.size(), targetTime);

        printf("vert_size size %d, vals:\n", vertDiv.size());

        for (int i = 0; i < vertDiv.size(); i++) {
            printf("v: %d\n", vertDiv[i]);
        }

        return vertDiv;
    }

    //adjust tasks in specific row
    std::vector<int> getDivPointsHoriz(int rowIdx) {
        float sum = 0;
        int blockSize = 8;

        for(int i = 0; i < taskLayout[rowIdx].size(); i++) {
            sum += renderTasks_[taskLayout[rowIdx][i]].time;
        }

        float targetTime = sum / (float)taskLayout[rowIdx].size();

        int offset = 0;
        int currentIdx = 0;
        int prevOffset = 0;
        int taskLeft = 0;

        int blockCount = framebuffer_->getResolution().width / blockSize;

        //new block times count
        std::vector<int> currentTaskTimes;
        std::vector<int> currentTaskLengths;
        for (int i = 0; i < taskLayout[rowIdx].size(); i++) {
            currentTaskTimes.push_back(renderTasks_[taskLayout[rowIdx][i]].time);
            currentTaskLengths.push_back(renderTasks_[taskLayout[rowIdx][i]].width);
        }
        std::vector<float> blockTimes = getBlockTimes(currentTaskTimes, currentTaskLengths, framebuffer_->getResolution().width);
        std::vector<int> horizontalDiv = getDivPoints(blockTimes, taskLayout[rowIdx].size(), targetTime);

        // return horizontalDiv;
        printf("horizontal size %d\n", horizontalDiv.size());
        for (int i = 0; i < horizontalDiv.size(); i++) {
            printf("h: %d\n", horizontalDiv[i]);
        }

        return horizontalDiv;
    }

private:
    std::vector<std::shared_ptr<DevicePathTracer>> devicePathTracers_{};
    TaskGenerator taskGen_{};
    SafeQueue<RenderTask> queue_{};
    std::condition_variable threadCv_{};
    std::condition_variable shouldThreadStartCV_{};
    bool shouldThreadStart;
    std::mutex threadStartMutex;
    semaphore threadSemaphore_{0};
    std::atomic_int completedStreams_ = 0;
    std::shared_ptr<Framebuffer> framebuffer_;
    HostScene& hScene_; 
    std::mutex m_;
    // std::unique_lock<std::mutex> lk_;
    std::vector<RenderTask> renderTasks_{};
    std::vector<std::vector<std::shared_ptr<StreamThread>>> streamThreads_{};    
    RendererConfig& config_;
    RendererConfig newConfig_{};
    bool shouldUpdatePathTracerParams = false;
    bool shouldReloadWorld = false;
    CameraConfig& cameraConfig_;
    Barrier barrier_;
    std::vector<std::vector<int>> taskLayout{};
    std::vector<int> offsets;
    std::vector<int> heights;
};
