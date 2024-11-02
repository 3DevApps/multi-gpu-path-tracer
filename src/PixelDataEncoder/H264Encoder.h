#pragma once
#include "PixelDataEncoder.h"
#include <vector>
#include <cstdint>
#include <iostream>
#include <x264.h>

class H264Encoder : public PixelDataEncoder {
    public:
        H264Encoder();
        ~H264Encoder();
        bool encodePixelData(const std::vector<uint8_t>& pixelData, const int width, const int height, std::vector<uint8_t>& outputData) override;

    private:
        int width;
        int height;
        x264_t* encoder;
        x264_picture_t pic_in;
        x264_picture_t pic_out;
};