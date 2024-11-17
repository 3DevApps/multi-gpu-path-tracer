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

    void updatePixel(int pixel_index, uint8_t r, uint8_t g, uint8_t b)
    {
        // RGB format
        fb_rgb_ptr_[3 * pixel_index] = r;
        fb_rgb_ptr_[3 * pixel_index + 1] = g;
        fb_rgb_ptr_[3 * pixel_index + 2] = b;
        // YUV 4:2:0 format
        fb_yuv_ptr_[pixel_index] = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
        int blockRow = pixel_index / resolution_.width;
        int blockCol = pixel_index % resolution_.width;
        if (blockRow % 2 == 0 && blockCol % 2 == 0)
        {
            int totalPixels = resolution_.width * resolution_.height;
            int uvSize = totalPixels / 4;
            int uOffset = totalPixels;
            int vOffset = totalPixels + uvSize;
            int uvIndex = (blockRow / 2) * (resolution_.width / 2) + (blockCol / 2);
            fb_yuv_ptr_[uOffset + uvIndex] = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
            fb_yuv_ptr_[vOffset + uvIndex] = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
        }
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
