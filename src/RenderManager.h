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
#include <cmath>

class RenderManager
{
public:
    RenderManager(RendererConfig &config, HostScene &hScene, CameraConfig &cameraConfig, SceneLoader &sceneLoader) : config_{config},
                                                                                                                     hScene_{hScene},
                                                                                                                     cameraConfig_{cameraConfig},
                                                                                                                     barrier_{config.gpuNumber * config.streamsPerGpu + 1},
                                                                                                                     sceneLoader_{sceneLoader}
    {
        // hardcoded size of vector with block times to avoid mallocs
        blockTimes.resize(500, std::vector<float>(500, 0));
        newConfig_ = config_;
        setup();
    }

    std::vector<std::vector<int>> getTaskLayout(unsigned int maxTasksInRow)
    {
        std::vector<std::vector<int>> taskLayout;
        int task = 0;
        while (task < config_.gpuNumber * config_.streamsPerGpu)
        {
            taskLayout.push_back({});
            int rowTask = 0;
            while (rowTask < maxTasksInRow && task < config_.gpuNumber * config_.streamsPerGpu)
            {
                taskLayout[taskLayout.size() - 1].push_back(task);
                task++;
                rowTask++;
            }
        }

        return taskLayout;
    }

    void reset()
    {
        barrier_.wait();
        for (int i = 0; i < config_.gpuNumber; i++)
        {
            for (int j = 0; j < config_.streamsPerGpu; j++)
            {
                streamThreads_[i][j]->detach();
            }
        }

        streamThreads_ = {};
        devicePathTracers_ = {};
    }

    void setup()
    {
        framebuffer_ = std::make_shared<Framebuffer>(config_.resolution);
        threadCount_ = config_.gpuNumber * config_.streamsPerGpu;
        barrier_.reset(threadCount_ + 1);
        for (int i = 0; i < config_.gpuNumber; i++)
        {
            devicePathTracers_.push_back(std::make_shared<DevicePathTracer>(
                i,
                config_.samplesPerPixel,
                config_.recursionDepth,
                config_.threadBlockSize,
                hScene_,
                framebuffer_,
                cameraConfig_));

            streamThreads_.push_back({});
            streamThreads_[i].reserve(config_.streamsPerGpu);
            for (int j = 0; j < config_.streamsPerGpu; j++)
            {
                streamThreads_[i].push_back(std::make_shared<StreamThread>(
                    i,
                    devicePathTracers_[i],
                    renderTasks_,
                    barrier_));
                streamThreads_[i][j]->start();
            }
        }

        taskLayout_ = getTaskLayout(config_.maxTasksInRow);
        renderTasks_ = taskGen_.generateEqualTasks(
            threadCount_,
            taskLayout_,
            config_.resolution.width,
            config_.resolution.height);
        depth = log2(threadCount_);
    }

    void setKParameter(int val)
    {
        newConfig_.maxTasksInRow = val;
        shouldUpdatePathTracerParams = true;
    }

    void updatePathTracingParamsIfNeeded()
    {
        if (!shouldUpdatePathTracerParams)
        {
            return;
        }
        shouldUpdatePathTracerParams = false;

        if (config_.algorithmType != newConfig_.algorithmType)
        {
            config_.algorithmType = newConfig_.algorithmType;
        }

        if (config_.showTasks != newConfig_.showTasks)
        {
            config_.showTasks = newConfig_.showTasks;
        }

        if (config_.gpuNumber != newConfig_.gpuNumber || config_.streamsPerGpu != newConfig_.streamsPerGpu)
        {
            reset();
            config_.gpuNumber = newConfig_.gpuNumber;
            config_.streamsPerGpu = newConfig_.streamsPerGpu;
            setup();
        }

        if (config_.resolution.width != newConfig_.resolution.width || config_.resolution.height != newConfig_.resolution.height || config_.maxTasksInRow != newConfig_.maxTasksInRow)
        {
            config_.maxTasksInRow = newConfig_.maxTasksInRow;
            config_.resolution = newConfig_.resolution;
            framebuffer_->setResolution(config_.resolution);
            for (const auto &dpt : devicePathTracers_)
            {
                dpt->setFramebuffer(framebuffer_);
            }
            renderTasks_ = taskGen_.generateEqualTasks(config_.gpuNumber * config_.streamsPerGpu, config_.resolution.width, config_.resolution.height);
            depth = log2(threadCount_);
        }

        if (config_.samplesPerPixel != newConfig_.samplesPerPixel)
        {
            config_.samplesPerPixel = newConfig_.samplesPerPixel;
            for (const auto &dpt : devicePathTracers_)
            {
                dpt->setSamplesPerPixel(config_.samplesPerPixel);
            }
        }

        if (config_.recursionDepth != newConfig_.recursionDepth)
        {
            config_.recursionDepth = newConfig_.recursionDepth;
            for (const auto &dpt : devicePathTracers_)
            {
                dpt->setRecursionDepth(config_.recursionDepth);
            }
        }

        if (config_.threadBlockSize.x != newConfig_.threadBlockSize.x || config_.threadBlockSize.y != newConfig_.threadBlockSize.y)
        {
            config_.threadBlockSize = newConfig_.threadBlockSize;
            for (const auto &dpt : devicePathTracers_)
            {
                dpt->setThreadBlockSize(config_.threadBlockSize);
            }
        }
    }

    void setGpuNumber(int gpuNumber)
    {
        // for DSDF only powers of 2 work
        if (config_.algorithmType == SchedulingAlgorithmType::DSDL) {
            int newGpuNumber = 1;
            while (newGpuNumber * 2 <= gpuNumber) {
                newGpuNumber *= 2;
            }
            gpuNumber = newGpuNumber;
        }

        newConfig_.gpuNumber = gpuNumber;
        shouldUpdatePathTracerParams = true;
    }

    void setStreamsPerGpu(int streamsPerGpu)
    {
        newConfig_.streamsPerGpu = streamsPerGpu;
        shouldUpdatePathTracerParams = true;
    }

    void setGpuAndStreamNumber(int gpuNumber, int streamsPerGpu)
    {
        setGpuNumber(gpuNumber);
        newConfig_.streamsPerGpu = streamsPerGpu;
        shouldUpdatePathTracerParams = true;
    }

    void setResolution(Resolution res)
    {
        newConfig_.resolution = res;
        shouldUpdatePathTracerParams = true;
    }

    void setSamplesPerPixel(unsigned int samples)
    {
        newConfig_.samplesPerPixel = samples;
        shouldUpdatePathTracerParams = true;
    }

    void setRecursionDepth(unsigned int depth)
    {
        newConfig_.recursionDepth = depth;
        shouldUpdatePathTracerParams = true;
    }

    void setThreadBlockSize(dim3 threadBlockSize)
    {
        newConfig_.threadBlockSize = threadBlockSize;
        shouldUpdatePathTracerParams = true;
    }

    void setSchedulingAlgorithm(SchedulingAlgorithmType alg)
    {
        newConfig_.algorithmType = alg;
        shouldUpdatePathTracerParams = true;
    }

    void setShowTasks(bool val)
    {
        newConfig_.showTasks = val;
        shouldUpdatePathTracerParams = true;
    }

    void reloadWorldIfNeeded()
    {
        if (!shouldReloadWorld)
        {
            return;
        }
        shouldReloadWorld = false;

        for (const auto &dpt : devicePathTracers_)
        {
            dpt->reloadWorld();
        }
    }

    void subdivide(int offset_x, int offset_y, int width, int height, int current, bool vert) {
        if (current == depth) {
            RenderTask rt = {
                config_.threadBlockSize.x * width,
                config_.threadBlockSize.y * height,
                config_.threadBlockSize.x * offset_x,
                config_.threadBlockSize.y * offset_y,
            };
            renderTasks_.push_back(rt);
            return;
        }

        float sum = 0;
        for (int xi = offset_x; xi < offset_x + width; xi++) {
            for (int yi = offset_y; yi < offset_y + height; yi++) {
                sum += blockTimes[yi][xi];
            }
        }

        float target = sum / 2;
        sum = 0;

        if (vert) {
            for (int yi = offset_y; yi < offset_y + height; yi++) {
                for (int xi = offset_x; xi < offset_x + width; xi++) {
                    sum += blockTimes[yi][xi];
                }
                if (target <= sum) {
                    subdivide(offset_x, offset_y, width, yi - offset_y, current + 1, false);
                    subdivide(offset_x, yi, width, height - yi, current + 1, false);
                    break;
                }
            }
        }
        else {
            for (int xi = offset_x; xi < offset_x + width; xi++) {
                for (int yi = offset_y; yi < offset_y + height; yi++) {
                    sum += blockTimes[yi][xi];
                }
                if (target <= sum) {
                    subdivide(offset_x, offset_y, xi - offset_x, height, current + 1, true);
                    subdivide(xi, offset_y, width - xi, height, current + 1, true);
                    break;
                }
            }
        }
    }

    void adjustTasksDSDL() {
        for (int i = 0; i < renderTasks_.size(); i++) {
            int blockOffsetHoriz = renderTasks_[i].offset_x / config_.threadBlockSize.x;
            int blockCountHoriz = renderTasks_[i].width / config_.threadBlockSize.x;
            int blockOffsetVert = renderTasks_[i].offset_y / config_.threadBlockSize.y;
            int blockCountVert = renderTasks_[i].height / config_.threadBlockSize.y;

            for (int xi = 0; xi < blockCountHoriz; xi++) {
                for (int yi = 0; yi < blockCountVert; yi++) {
                    blockTimes[blockOffsetVert + yi][blockOffsetVert + xi] = renderTasks_[i].time / (float)blockCountVert * blockCountHoriz;
                }
            }
        }
        renderTasks_ = {};

        int bv = framebuffer_->getResolution().width / config_.threadBlockSize.x;
        int bh = framebuffer_->getResolution().height / config_.threadBlockSize.y;

        subdivide(0, 0, bh, bv, 0, true);
    }

    // adjust tasks for dynamic task size and fixed layout variant
    void adjustTasksDSFL()
    {
        // calculate horizontal and vertical divison points
        auto horizDivPoints = getHorizDivPoints();
        auto vertDivPoints = getVertDivPoints();

        // adjust widths and offset_x for tasks
        for (int rowIdx = 0; rowIdx < taskLayout_.size(); rowIdx++)
        {
            renderTasks_[taskLayout_[rowIdx][0]].offset_x = 0;
            for (int i = 0; i < horizDivPoints[rowIdx].size(); i++)
            {
                auto &task = renderTasks_[taskLayout_[rowIdx][i]];
                auto &nextTask = renderTasks_[taskLayout_[rowIdx][i + 1]];
                int newWidth = std::max(horizDivPoints[rowIdx][i] - task.offset_x, 1);
                if (newWidth > task.width)
                {
                    task.width = std::min(newWidth, static_cast<int>(task.width + config_.threadBlockSize.x));
                    if (task.offset_x + task.width > framebuffer_->getResolution().width)
                    {
                        task.width = framebuffer_->getResolution().height - task.offset_x;
                    }
                    nextTask.offset_x = task.offset_x + task.width;
                }
                else
                {
                    task.width = std::max(newWidth, static_cast<int>(task.width - config_.threadBlockSize.x));
                    if (task.offset_x + task.width > framebuffer_->getResolution().width)
                    {
                        task.width = framebuffer_->getResolution().height - task.offset_x;
                    }
                    nextTask.offset_x = task.offset_x + task.width;
                }
            }
            auto &lastTask = renderTasks_[taskLayout_[rowIdx][taskLayout_[rowIdx].size() - 1]];
            lastTask.width = framebuffer_->getResolution().width - lastTask.offset_x;
        }

        // adjust heights and offset_y for tasks
        auto offsets = std::vector<int>(taskLayout_.size());
        auto heights = std::vector<int>(taskLayout_.size());
        offsets[0] = 0;
        for (int i = 0; i < vertDivPoints.size(); i++)
        {
            int taskHeight = renderTasks_[taskLayout_[i][0]].height;
            if (taskHeight > vertDivPoints[i] - offsets[i])
            {
                heights[i] = std::max(std::max(vertDivPoints[i] - offsets[i], 1), static_cast<int>(taskHeight - config_.threadBlockSize.x));
                if (offsets[i] + heights[i] > framebuffer_->getResolution().height)
                {
                    heights[i] = framebuffer_->getResolution().height - offsets[i];
                }
                offsets[i + 1] = offsets[i] + heights[i];
            }
            else
            {
                heights[i] = std::min(std::max(vertDivPoints[i] - offsets[i], 1), static_cast<int>(taskHeight + config_.threadBlockSize.x));
                if (offsets[i] + heights[i] > framebuffer_->getResolution().height)
                {
                    heights[i] = framebuffer_->getResolution().height - offsets[i];
                }
                offsets[i + 1] = offsets[i] + heights[i];
            }
        }
        heights[vertDivPoints.size()] = framebuffer_->getResolution().height - offsets[vertDivPoints.size()];

        for (int i = 0; i < taskLayout_.size(); i++)
        {
            for (int j = 0; j < taskLayout_[i].size(); j++)
            {
                renderTasks_[taskLayout_[i][j]].offset_y = offsets[i];
                renderTasks_[taskLayout_[i][j]].height = heights[i];
            }
        }
    }

    void renderFrame() {
        updatePathTracingParamsIfNeeded();
        reloadWorldIfNeeded();

        if (config_.algorithmType == SchedulingAlgorithmType::DSFL) {
            adjustTasksDSFL();
        }
        else if (config_.algorithmType == SchedulingAlgorithmType::DSDL) {
            adjustTasksDSDL();
        }

        barrier_.wait(); // wait for render threads

        // path tracer works here...

        barrier_.wait(); // wair for rendering end

        if (config_.showTasks)
        {
            markTasks();
        }
    }

    void updateMetrics(MonitorThread& monitorThreadObj) {
        for (int i = 0; i < renderTasks_.size(); i++) {
            monitorThreadObj.updateTimeOfRendering(i, renderTasks_[i].time);
        }

        float sum = 0;
        float max = 0;
        for (int i = 0; i < renderTasks_.size(); i++) {
            sum += renderTasks_[i].time;
            if (renderTasks_[i].time > max) {
                max = renderTasks_[i].time;
            }
        }
        monitorThreadObj.updateImbalance(max / (sum / renderTasks_.size()));
    }

    void markTasks() {
        int boldness = framebuffer_->getResolution().height / 300;
        for (int i = 0; i < renderTasks_.size(); i++) {
            if (renderTasks_[i].offset_y != 0) {
                markTaskBorder(
                    renderTasks_[i].offset_x,
                    renderTasks_[i].offset_x + renderTasks_[i].width,
                    renderTasks_[i].offset_y,
                    boldness,
                    true
                );
            }

            markTaskBorder(
                renderTasks_[i].offset_x,
                renderTasks_[i].offset_x + renderTasks_[i].width,
                renderTasks_[i].offset_y + renderTasks_[i].height,
                boldness,
                true
            );

            if (renderTasks_[i].offset_x != 0) {
                markTaskBorder(
                    renderTasks_[i].offset_y,
                    renderTasks_[i].offset_y + renderTasks_[i].height,
                    renderTasks_[i].offset_x,
                    boldness,
                    false
                );
            }

            markTaskBorder(
                renderTasks_[i].offset_y,
                renderTasks_[i].offset_y + renderTasks_[i].height,
                renderTasks_[i].offset_x + renderTasks_[i].width,
                boldness,
                false
            );
        }
    }

    void markTaskBorder(int begin, int end, int offset, int boldness, bool isVert) {
        if (isVert) {
            for (int x = begin; x < end; x++) {
                for (int yi = offset; yi <= offset + boldness && yi < framebuffer_->getResolution().height; yi++) {
                    int pixel_index = yi * framebuffer_->getResolution().width + x;
                    framebuffer_->updatePixel(pixel_index, 100, 100, 100);
                }
            }
        }
        else {
            for (int y = begin; y < end; y++) {
                for (int xi = offset; xi <= offset + boldness && xi < framebuffer_->getResolution().width; xi++) {
                    int pixel_index = y * framebuffer_->getResolution().width + xi;
                    framebuffer_->updatePixel(pixel_index, 100, 100, 100);
                }
            }
        }
    }

    std::shared_ptr<Framebuffer> &getFramebuffer()
    {
        return framebuffer_;
    }

    uint8_t *getCurrentFrame()
    {
        return framebuffer_->getRGBPtr();
    }

    uint8_t *getYUVFrame()
    {
        return framebuffer_->getYUVPtr();
    }

    unsigned int getCurrentFrameWidth()
    {
        return framebuffer_->getResolution().width;
    }

    unsigned int getCurrentFrameHeight()
    {
        return framebuffer_->getResolution().height;
    }

    void reloadScene()
    {
        std::string objPath = "../files/f" + config_.jobId + ".glb";
        hScene_ = sceneLoader_.load(objPath);
        shouldReloadWorld = true;
    }

    void updatePrimitives()
    {
        shouldReloadWorld = true;
    }

    std::vector<int> getRowDivPoints(std::vector<int> &taskTimes, std::vector<int> &taskLengths, int blockLength)
    {
        if (taskTimes.size() == 1)
        {
            return {};
        }

        std::vector<int> divPoints;
        float sum = 0;
        for (int i = 0; i < taskTimes.size(); i++)
        {
            sum += taskTimes[i];
        }
        float targetTime = sum / (float)taskTimes.size();
        int currentBlock = 0;
        float currentTime = 0;
        int divCount = taskTimes.size() - 1;

        // iterate over tasks in a row and block widths inside task,
        // if current time is exceedes target time, division point is found
        int blocks = framebuffer_->getResolution().width / blockLength;
        for (int i = 0; i < taskTimes.size(); i++)
        {
            int blockCount = taskLengths[i] / blockLength;
            float currentBlockTime = taskTimes[i] / (float)blockCount;
            for (int k = 0; k < blockCount; k++)
            {
                int remainingDivs = divCount - divPoints.size();
                // we add divion point if current time exceeds target time
                // or number of blocks to the end of row equals number of divion points we need to add
                if (currentTime + currentBlockTime > targetTime ||
                    blocks - currentBlock == remainingDivs)
                {
                    divPoints.push_back(currentBlock * blockLength);
                    currentTime = 0;
                }
                else
                {
                    currentTime += currentBlockTime;
                }
                currentBlock++;

                // if we found all division points (number of tasks in a row - 1), return
                if (divPoints.size() == divCount)
                {
                    return divPoints;
                }
            }
        }

        // we add divion points util we have the right number.
        while (divCount > divPoints.size())
        {
            divPoints.push_back(currentBlock * blockLength);
            currentBlock++;
        }
        return divPoints;
    }

    std::vector<std::vector<int>> getHorizDivPoints()
    {
        std::vector<std::vector<int>> divPoints{};
        for (int i = 0; i < taskLayout_.size(); i++)
        {
            std::vector<int> lengths;
            std::vector<int> times;
            for (int j = 0; j < taskLayout_[i].size(); j++)
            {
                lengths.push_back(renderTasks_[taskLayout_[i][j]].width);
                times.push_back(renderTasks_[taskLayout_[i][j]].time);
            }
            divPoints.push_back(getRowDivPoints(times, lengths, config_.threadBlockSize.x));
        }
        return divPoints;
    }

    std::vector<int> getVertDivPoints()
    {
        float sum = 0;
        std::vector<int> horizTimes(taskLayout_.size());
        std::vector<int> lengths;
        for (int i = 0; i < taskLayout_.size(); i++)
        {
            horizTimes[i] = 0;
            for (int j = 0; j < taskLayout_[i].size(); j++)
            {
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
    HostScene &hScene_;
    std::vector<RenderTask> renderTasks_{};
    std::vector<std::vector<std::shared_ptr<StreamThread>>> streamThreads_{};
    RendererConfig &config_;
    RendererConfig newConfig_{};
    bool shouldUpdatePathTracerParams = false;
    bool shouldReloadWorld = false;
    CameraConfig &cameraConfig_;
    Barrier barrier_;
    std::vector<std::vector<int>> taskLayout_;
    int threadCount_;
    SceneLoader &sceneLoader_;
    std::vector<std::vector<float>> blockTimes;
    int depth;
};
