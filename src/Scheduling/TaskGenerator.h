#pragma once
#include <vector>
#include "../DevicePathTracer.h"


class TaskGenerator
{
public:
    TaskGenerator(int width, int height) : width_{width}, height_{height} {}
    TaskGenerator() {}

    //TODO: Maybe square tasks are faster or something cus of warp size

    void generateTasks(int num_tasks, std::vector<RenderTask> &tasks) {
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

    //generates tasks with equal size, positioned in a row
    std::vector<RenderTask> generateEqualTasks(int task_count, int width, int height) {
        int task_width = width / task_count;
        std::vector<RenderTask> tasks;

        for (int i = 0; i < task_count - 1; i++) {
            tasks.push_back({task_width, height, i * task_width, 0});
        }
        tasks.push_back({width - (task_count - 1) * task_width, height, (task_count - 1) * task_width, 0});
        return tasks;
    }

    //generates tasks with equal size, taskLayout specifies task positions on the screen
    std::vector<RenderTask> generateEqualTasks(int taskCount, std::vector<std::vector<int>> &taskLayout, int width, int height) {
	    std::vector<RenderTask> tasks(taskCount);
	    int taskHeight = height / taskLayout.size();
	    for (int i = 0; i < taskLayout.size(); i++) {
		    int taskWidth = width / taskLayout[i].size();
		    for (int j = 0; j < taskLayout[i].size(); j++) {
			    tasks[taskLayout[i][j]].width = taskWidth;
			    tasks[taskLayout[i][j]].offset_x = taskWidth * j;
			    tasks[taskLayout[i][j]].height = taskHeight;
			    tasks[taskLayout[i][j]].offset_y= taskHeight * i;
		    }
	    }

	    for (int i = 0; i < taskLayout.size(); i++) {
		    tasks[taskLayout[i][taskLayout[i].size() - 1]].width = width - tasks[taskLayout[i][taskLayout[i].size() - 1]].offset_x;
	    }

	    for (int i = 0; i < taskLayout[taskLayout.size() - 1].size(); i++) {
		    tasks[taskLayout[taskLayout.size() - 1][i]].height = height - tasks[taskLayout[taskLayout.size() - 1][i]].offset_y;
	    }

	    return tasks;
    }

    void setRes(int width, int height) {
        width_ = width;
        height_ = height;
    }

private:
    int width_;
    int height_;
};
