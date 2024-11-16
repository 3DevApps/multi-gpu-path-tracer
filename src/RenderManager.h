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
#include <algorithm>

class RenderManager {
public:
        RenderManager(RendererConfig &config, HostScene &hScene, CameraConfig& cameraConfig, SceneLoader& sceneLoader) :
        config_{config},
        hScene_{hScene},
        cameraConfig_{cameraConfig},
        barrier_{config.gpuNumber * config.streamsPerGpu + 1},
        sceneLoader_{sceneLoader} {

        newConfig_ = config_;
        setup();
    }

    std::vector<std::vector<int>> getTaskLayout(unsigned int maxTasksInRow) {
        std::vector<std::vector<int>> taskLayout;
        int task = 0;
        while (task < config_.gpuNumber * config_.streamsPerGpu) {
            taskLayout.push_back({});
            int rowTask = 0;
            while (rowTask < maxTasksInRow && task < config_.gpuNumber * config_.streamsPerGpu) {
                taskLayout[taskLayout.size() - 1].push_back(task);
                task++;
                rowTask++;
            }
        }

        return taskLayout;
    }

    void reset() {
        barrier_.wait();
        for (int i = 0; i < config_.gpuNumber; i++) {
            for (int j = 0; j < config_.streamsPerGpu; j++) {
                streamThreads_[i][j]->detach();
            }
        }

        streamThreads_ = {};
        devicePathTracers_ = {};
    }

    void setup() {
        framebuffer_ = std::make_shared<Framebuffer>(config_.resolution);
        threadCount_ = config_.gpuNumber * config_.streamsPerGpu;
        barrier_.reset(threadCount_ + 1);
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
                    devicePathTracers_[i],
                    renderTasks_,
                    barrier_
                ));
                streamThreads_[i][j]->start();
            }
        }

        taskLayout_ = getTaskLayout(config_.maxTasksInRow);
        renderTasks_ = taskGen_.generateEqualTasks(
            threadCount_,
            taskLayout_,
            config_.resolution.width,
            config_.resolution.height
        );
    }

    void updatePathTracingParamsIfNeeded() {
        if (!shouldUpdatePathTracerParams) {
            return;
        }
        shouldUpdatePathTracerParams = false;

        if (config_.algorithmType != newConfig_.algorithmType) {
            config_.algorithmType = newConfig_.algorithmType;
        }

        if (config_.showTasks != newConfig_.showTasks) {
            config_.showTasks = newConfig_.showTasks;
        }

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

    void setSchedulingAlgorithm(SchedulingAlgorithmType alg) {
        newConfig_.algorithmType = alg;
        shouldUpdatePathTracerParams = true;
    }

    void setShowTasks(bool val) {
        newConfig_.showTasks = val;
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

    //adjust tasks for dynamic task size and fixed layout variant
    void adjustTasksDTFL() {
        //calculate horizontal and vertical divison points
	    auto horizDivPoints = getHorizDivPoints();
	    auto vertDivPoints = getVertDivPoints();

        //adjust widths and offset_x for tasks
        for (int rowIdx = 0; rowIdx < taskLayout_.size(); rowIdx++) {
            int i = 0;
		    renderTasks_[taskLayout_[rowIdx][0]].offset_x = 0;
            for (int i = 0; i < horizDivPoints[rowIdx].size(); i++) {
                auto &task = renderTasks_[taskLayout_[rowIdx][i]];
                auto &nextTask = renderTasks_[taskLayout_[rowIdx][i + 1]];
                int newWidth = std::max(horizDivPoints[rowIdx][i] - task.offset_x, 1);
                if (newWidth > task.width) {
                    task.width = std::min(newWidth, static_cast<int>(task.width + config_.threadBlockSize.x));
                    if (task.offset_x + task.width > framebuffer_->getResolution().width) {
                        task.width = framebuffer_->getResolution().height - task.offset_x;
                    }
                    nextTask.offset_x = task.offset_x + task.width;
		        }
		        else {
                    task.width = std::max(newWidth, static_cast<int>(task.width - config_.threadBlockSize.x));
                    if (task.offset_x + task.width > framebuffer_->getResolution().width) {
                        task.width = framebuffer_->getResolution().height - task.offset_x;
                    }
			        nextTask.offset_x = task.offset_x + task.width;
		        }
            }
            auto &lastTask = renderTasks_[taskLayout_[rowIdx][taskLayout_[rowIdx].size() - 1]];
            lastTask.width = framebuffer_->getResolution().width - lastTask.offset_x;
        }


        //adjust heights and offset_y for tasks
        auto offsets = std::vector<int>(taskLayout_.size());
    	auto heights = std::vector<int>(taskLayout_.size());
        offsets[0] = 0;
        for (int i = 0; i < vertDivPoints.size(); i++) {
            int taskHeight = renderTasks_[taskLayout_[i][0]].height;
            if (taskHeight > vertDivPoints[i] - offsets[i]) {
                heights[i] = std::max(std::max(vertDivPoints[i] - offsets[i], 1), static_cast<int>(taskHeight - config_.threadBlockSize.x));
                if (offsets[i] + heights[i] > framebuffer_->getResolution().height) {
                    heights[i] = framebuffer_->getResolution().height - offsets[i];
                }
                offsets[i + 1] = offsets[i] + heights[i];
            }
            else {
                heights[i] = std::min(std::max(vertDivPoints[i] - offsets[i], 1), static_cast<int>(taskHeight + config_.threadBlockSize.x));
                if (offsets[i] + heights[i] > framebuffer_->getResolution().height) {
                    heights[i] = framebuffer_->getResolution().height - offsets[i];
                }
                offsets[i + 1] = offsets[i] + heights[i];
            }
        }
        heights[vertDivPoints.size()] = framebuffer_->getResolution().height - offsets[vertDivPoints.size()];

        for (int i = 0; i < taskLayout_.size(); i++) {
                for (int j = 0; j < taskLayout_[i].size(); j++) {
                    renderTasks_[taskLayout_[i][j]].offset_y = offsets[i];
                    renderTasks_[taskLayout_[i][j]].height = heights[i];
                }
        }
    }

    void renderFrame() {
        updatePathTracingParamsIfNeeded();
        reloadWorldIfNeeded();

        if (config_.algorithmType == DTFL) {
            adjustTasksDTFL();
        }

        barrier_.wait(); // wait for render threads

        //path tracer works here...

        barrier_.wait(); // wair for rendering end

        if (config_.showTasks) {
            markTasks();
        }
    }


    void markTasks() {
	    for (int i = 0; i < renderTasks_.size(); i++) {
		    for (int x = 0; x < renderTasks_[i].width; x++) {
			    int pixel_index = renderTasks_[i].offset_y * framebuffer_->getResolution().width + renderTasks_[i].offset_x + x;
				framebuffer_->updatePixel(pixel_index, 0, 0, 0);
		    }

		    for (int x = 0; x < renderTasks_[i].width; x++) {
			    int pixel_index = (renderTasks_[i].offset_y + renderTasks_[i].height) * framebuffer_->getResolution().width + renderTasks_[i].offset_x + x;
				framebuffer_->updatePixel(pixel_index, 0, 0, 0);
		    }

		    for (int y = 0; y < renderTasks_[i].height; y++) {
			    int pixel_index = (renderTasks_[i].offset_y + y) * framebuffer_->getResolution().width + renderTasks_[i].offset_x;
				framebuffer_->updatePixel(pixel_index, 0, 0, 0);
		    }

		    for (int y = 0; y < renderTasks_[i].height; y++) {
			    int pixel_index = (renderTasks_[i].offset_y + y) * framebuffer_->getResolution().width + renderTasks_[i].offset_x + renderTasks_[i].width;
				framebuffer_->updatePixel(pixel_index, 0, 0, 0);
		    }
	    }
    }

    std::shared_ptr<Framebuffer>& getFramebuffer() {
        return framebuffer_;
    }

    uint8_t* getCurrentFrame() {
        return framebuffer_->getRGBPtr();
    }

    uint8_t* getYUVFrame() {
        return framebuffer_->getYUVPtr();
    }

    unsigned int getCurrentFrameWidth() {
        return framebuffer_->getResolution().width;
    }

    unsigned int getCurrentFrameHeight() {
        return framebuffer_->getResolution().height;
    }

    void reloadScene() {
        std::string objPath = "../files/f" + config_.jobId + ".glb";
        hScene_ = sceneLoader_.load(objPath);
        shouldReloadWorld = true;
    }

    void updatePrimitives() {
        shouldReloadWorld = true;
    }

    std::vector<int> getRowDivPoints(std::vector<int> &taskTimes, std::vector<int> &taskLengths, int blockLength) {
        if (taskTimes.size() == 1) {
            return {};
        }

        std::vector<int> divPoints;
        float sum = 0;
        for (int i = 0; i < taskTimes.size(); i++) {
            sum += taskTimes[i];
        }
        float targetTime = sum / (float)taskTimes.size();
        int currentBlock = 0;
        float currentTime = 0;
        int divCount = taskTimes.size() - 1;

        //iterate over tasks in a row and block widths inside task,
        //if current time is exceedes target time, division point is found
        int blocks = framebuffer_->getResolution().width / blockLength;
        for (int i = 0; i < taskTimes.size(); i++) {
            int blockCount = taskLengths[i] / blockLength;
            float currentBlockTime = taskTimes[i] /  (float)blockCount;
            for (int k = 0; k < blockCount; k++) {
                int remainingDivs = divCount - divPoints.size();
                //we add divion point if current time exceeds target time
                //or number of blocks to the end of row equals number of divion points we need to add
                if (currentTime + currentBlockTime > targetTime ||
                    blocks - currentBlock == remainingDivs) {
                    divPoints.push_back(currentBlock * blockLength);
                    currentTime = 0;
                }
                else {
                    currentTime += currentBlockTime;
                }
                currentBlock++;

                // if we found all division points (number of tasks in a row - 1), return
                if (divPoints.size() == divCount) {
                    return divPoints;
                }
            }
        }

        //we add divion points util we have the right number.
        while (divCount > divPoints.size()) {
            divPoints.push_back(currentBlock * blockLength);
            currentBlock++;
        }
        return divPoints;
    }

    std::vector<std::vector<int>> getHorizDivPoints() {
	    std::vector<std::vector<int>> divPoints{};
        for(int i = 0; i < taskLayout_.size(); i++) {
            std::vector<int> lengths;
            std::vector<int> times;
            for (int j = 0; j < taskLayout_[i].size(); j++) {
                lengths.push_back(renderTasks_[taskLayout_[i][j]].width);
                times.push_back(renderTasks_[taskLayout_[i][j]].time);
            }
            divPoints.push_back(getRowDivPoints(times, lengths, config_.threadBlockSize.x));
        }
	    return divPoints;
    }


    std::vector<int> getVertDivPoints() {
        float sum = 0;
        std::vector<int> horizTimes(taskLayout_.size());
        std::vector<int> lengths;
        for (int i = 0; i < taskLayout_.size(); i++) {
            horizTimes[i] = 0;
            for (int j = 0; j < taskLayout_[i].size(); j++) {
                sum += renderTasks_[taskLayout_[i][j]].time;
                horizTimes[i] += renderTasks_[taskLayout_[i][j]].time;
            }
            lengths.push_back(renderTasks_[taskLayout_[i][0]].height);
        }

        return getRowDivPoints(horizTimes, lengths, config_.threadBlockSize.y);
    }

private:
    std::vector<std::shared_ptr<DevicePathTracer>> devicePathTracers_{};
    TaskGenerator taskGen_{};
    std::shared_ptr<Framebuffer> framebuffer_;
    HostScene& hScene_;
    std::vector<RenderTask> renderTasks_{};
    std::vector<std::vector<std::shared_ptr<StreamThread>>> streamThreads_{};
    RendererConfig& config_;
    RendererConfig newConfig_{};
    bool shouldUpdatePathTracerParams = false;
    bool shouldReloadWorld = false;
    CameraConfig& cameraConfig_;
    Barrier barrier_;
    std::vector<std::vector<int>> taskLayout_;
    int threadCount_;
    SceneLoader& sceneLoader_;
};
