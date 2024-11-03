#pragma once

#include "cuda_utils.h"
#include "RendererConfig.h"

class Framebuffer
{
public:
    Framebuffer(Resolution res)
    {
        resolution_ = res;
        initializePointers();
    }

    void setResolution(Resolution res)
    {
        if (res.width == resolution_.width && res.height == resolution_.height)
        {
            return;
        }
        resolution_ = res;
        checkCudaErrors(cudaFree(fb_rgb_ptr_));
        checkCudaErrors(cudaFree(fb_yuv_ptr_));
        initializePointers();
    }

    void initializePointers()
    {
        // RGB format
        const int totalPixels = resolution_.width * resolution_.height;
        checkCudaErrors(cudaMallocManaged((void **)&fb_rgb_ptr_, totalPixels * 3 * sizeof(uint8_t)));
        // YUV 4:2:0 format
        const int uvSize = totalPixels / 4;
        checkCudaErrors(cudaMallocManaged((void **)&fb_yuv_ptr_, totalPixels + 2 * uvSize * sizeof(uint8_t)));
    }

    Resolution getResolution()
    {
        return resolution_;
    }

    unsigned int getPixelCount()
    {
        return resolution_.width * resolution_.height;
    }

    uint8_t *getRGBPtr()
    {
        return fb_rgb_ptr_;
    }

    uint8_t *getYUVPtr()
    {
        return fb_yuv_ptr_;
    }

    ~Framebuffer()
    {
        checkCudaErrors(cudaFree(fb_rgb_ptr_));
        checkCudaErrors(cudaFree(fb_yuv_ptr_));
    }

private:
    Resolution resolution_;
    uint8_t *fb_rgb_ptr_;
    uint8_t *fb_yuv_ptr_;
};