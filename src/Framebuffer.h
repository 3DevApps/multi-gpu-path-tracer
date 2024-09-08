#pragma once

#include "cuda_utils.h"

class Framebuffer {
public:
    Framebuffer(Resolution res) {
        resolution_ = res;
        size_t pixel_count = res.width * res.height * sizeof(uint8_t) * 3;
        checkCudaErrors(cudaMallocManaged((void **)&fb_ptr_, pixel_count));
    }

    void setResolution(Resolution res) {
        if (res.width == resolution_.width && res.height == resolution_.height) {
            return;
        }
        resolution_ = res;
        size_t pixel_count = resolution_.width * resolution_.height * sizeof(uint8_t) * 3;
        checkCudaErrors(cudaFree(fb_ptr_));
        checkCudaErrors(cudaMallocManaged((void **)&fb_ptr_, pixel_count));
    }

    Resolution getResolution() {
        return resolution_;
    }

    unsigned int getPixelCount() {
        return resolution_.width * resolution_.height;
    }

    uint8_t* getPtr() {
        return fb_ptr_;
    }

    ~Framebuffer() {
        checkCudaErrors(cudaFree(fb_ptr_));
    }

private:
    Resolution resolution_;
    uint8_t *fb_ptr_;
};