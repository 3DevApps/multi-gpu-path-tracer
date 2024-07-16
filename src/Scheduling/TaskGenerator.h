#pragma once
#include <vector>
#include "../DevicePathTracer.h"


class TaskGenerator
{
public:
    TaskGenerator(int width, int height) : width_{width}, height_{height} {}

    //TODO: Maybe square tasks are faster or something cus of warp size

    void generateTasks(int num_tasks,std::vector<RenderTask> &tasks) {
        int task_width = width_ / num_tasks;
        std::cout << "Task width: " << task_width << std::endl;
        exit(0);
        int task_height = height_;
        for (int i = 0; i < num_tasks; i++) {
            tasks.push_back({task_width, task_height, task_width*i, 0});
        }
    }
    void generateTasks(int task_size_x, int task_size_y, std::vector<RenderTask> &tasks) {
        int task_width = task_size_x;
        int task_height = task_size_y;
        for (int i = 0; i < (width_ / task_size_x)+1; i++) {
            if(task_width*(i+1) >= width_) {
                    task_width = width_ - task_width*i;
            }
            else {
                task_width = task_size_x;
            }
            for (int j = 0; j < (height_ / task_size_y)+1; j++) {
                if(task_height*(j+1) >= height_) {
                    task_height = height_ - task_height*j;
                }
                else {
                    task_height = task_size_y;
                }
                tasks.push_back({task_width, task_height, task_width*i, task_height*j});
            }
        }
    }

private:
    int width_;
    int height_;
};