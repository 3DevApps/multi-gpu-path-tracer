#pragma once
#include "PixelDataEncoder.h"
#include <vector>
#include <cstdint>
#include <iostream>
#include <x264.h>

class H264Encoder : public PixelDataEncoder
{
public:
    H264Encoder();
    ~H264Encoder();
    bool encodePixelData(const std::vector<uint8_t> &pixelData, const int width, const int height, std::vector<uint8_t> &outputData) override;

    void flush()
    {
        while (x264_encoder_delayed_frames(encoder))
        {
            x264_encoder_encode(encoder, nullptr, nullptr, nullptr, &pic_out);
        }
    }

private:
    bool first_frame = true;
    x264_t *encoder = nullptr;
    x264_picture_t pic_in{};
    x264_picture_t pic_out{};
    int width;
    int height;
    std::vector<uint8_t> encoded_data;
    std::vector<uint8_t> codec_config;
    bool config_retrieved = false;
    void convertRGBtoI420(const std::vector<uint8_t> &rgb_data)
    {
        uint8_t *y = pic_in.img.plane[0];
        uint8_t *u = pic_in.img.plane[1];
        uint8_t *v = pic_in.img.plane[2];

        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                int rgb_index = (i * width + j) * 3;
                int y_index = i * width + j;
                int uv_index = (i / 2) * (width / 2) + (j / 2);

                uint8_t r = rgb_data[rgb_index];
                uint8_t g = rgb_data[rgb_index + 1];
                uint8_t b = rgb_data[rgb_index + 2];

                // RGB to YUV conversion
                y[y_index] = ((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;

                if (i % 2 == 0 && j % 2 == 0)
                {
                    u[uv_index] = ((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                    v[uv_index] = ((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
                }
            }
        }
    }
};