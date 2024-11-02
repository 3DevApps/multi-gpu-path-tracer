#include "H264Encoder.h"
#include <chrono>

H264Encoder::H264Encoder()
{
    int fps = 30;
    int width = 400;
    int height = 400;

    this->width = width;
    this->height = height;

    x264_param_t param;
    x264_param_default_preset(&param, "medium", "zerolatency");

    param.i_width = width;
    param.i_height = height;
    param.i_fps_num = fps;
    param.i_fps_den = 1;
    param.i_csp = X264_CSP_I420;
    param.i_threads = 1;
    param.i_keyint_max = fps;
    param.b_repeat_headers = 1; // Force SPS/PPS on keyframes
    param.b_annexb = 1;         // Use Annex B format with start codes

    x264_param_apply_profile(&param, "baseline");

    encoder = x264_encoder_open(&param);
    if (!encoder)
    {
        throw std::runtime_error("Failed to open encoder");
    }

    if (x264_picture_alloc(&pic_in, X264_CSP_I420, width, height) < 0)
    {
        throw std::runtime_error("Failed to allocate picture");
    }
}

H264Encoder::~H264Encoder()
{
    if (encoder)
    {
        x264_encoder_close(encoder);
        x264_picture_clean(&pic_in);
    }
}

bool H264Encoder::encodePixelData(const std::vector<uint8_t> &pixelData, const int width, const int height, std::vector<uint8_t> &outputData)
{
    convertRGBtoI420(pixelData);

    x264_nal_t *nals;
    int num_nals;

    // Force keyframe for the first frame
    if (first_frame)
    {
        pic_in.i_type = X264_TYPE_IDR;
        first_frame = false;
    }

    if (x264_encoder_encode(encoder, &nals, &num_nals, &pic_in, &pic_out) < 0)
    {
        throw std::runtime_error("Failed to encode frame");
    }

    outputData.clear();
    for (int i = 0; i < num_nals; i++)
    {
        // Add start code (0x00000001)
        outputData.push_back(0);
        outputData.push_back(0);
        outputData.push_back(0);
        outputData.push_back(1);

        // Add NAL unit
        outputData.insert(outputData.end(),
                          nals[i].p_payload,
                          nals[i].p_payload + nals[i].i_payload);
    }

    return true;
}

void H264Encoder::convertRGBtoI420(const std::vector<uint8_t> &rgbData)
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

            uint8_t r = rgbData[rgb_index];
            uint8_t g = rgbData[rgb_index + 1];
            uint8_t b = rgbData[rgb_index + 2];

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