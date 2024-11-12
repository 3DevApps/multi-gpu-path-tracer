#pragma once
#include "PixelDataEncoder.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <x264.h>

class H264Encoder : public PixelDataEncoder
{
public:
    ~H264Encoder();
    bool encodePixelData(const uint8_t *frame, const int width, const int height, std::vector<uint8_t> &outputData) override;

private:
    x264_t *encoder = nullptr;
    x264_picture_t pic_in{};
    x264_picture_t pic_out{};
    int width = 0;
    int height = 0;
    bool first_frame = true;

    void initEncoder(const int width, const int height);
    void updateEncoderIfNeeded(const int width, const int height);
    void destroyEncoder();
};