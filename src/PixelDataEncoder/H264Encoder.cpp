#include "H264Encoder.h"
#include <chrono>

H264Encoder::H264Encoder()
{
    x264_param_t param;

    // Initialize x264 parameters
    x264_param_default_preset(&param, "ultrafast", "zerolatency");

    param.i_csp = X264_CSP_I420;
    param.i_width = 400;
    param.i_height = 400;
    param.i_fps_num = 25;
    param.i_fps_den = 1;
    param.b_annexb = 1;     // Enable Annex B
    param.i_level_idc = 51; // Level 5.1

    // Apply profile restrictions
    x264_param_apply_profile(&param, "high");

    // Open the encoder
    encoder = x264_encoder_open(&param);
    if (!encoder)
    {
        throw std::runtime_error("Failed to open x264 encoder.");
    }

    // Allocate image and set up picture
    x264_picture_alloc(&pic_in, X264_CSP_I420, param.i_width, param.i_height);
}

H264Encoder::~H264Encoder()
{
    // Free resources
    x264_picture_clean(&pic_in);
    x264_encoder_close(encoder);
}

bool H264Encoder::encodePixelData(const std::vector<uint8_t> &pixelData, const int width, const int height, std::vector<uint8_t> &outputData)
{
    // Convert RGB to YUV420
    auto start = std::chrono::high_resolution_clock::now();
    int y_size = width * height;
    int u_size = y_size / 4;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            uint8_t r = pixelData[(y * width + x) * 3 + 0];
            uint8_t g = pixelData[(y * width + x) * 3 + 1];
            uint8_t b = pixelData[(y * width + x) * 3 + 2];

            int y_index = y * width + x;
            int u_index = (y / 2) * (width / 2) + (x / 2);
            int v_index = u_index;

            pic_in.img.plane[0][y_index] = (uint8_t)((66 * r + 129 * g + 25 * b + 128) >> 8) + 16;
            if (y % 2 == 0 && x % 2 == 0)
            {
                pic_in.img.plane[1][u_index] = (uint8_t)((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128;
                pic_in.img.plane[2][v_index] = (uint8_t)((112 * r - 94 * g - 18 * b + 128) >> 8) + 128;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "RGB to YUV420 conversion time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    // Note: pic_in.i_pts should be set appropriately in a real application
    pic_in.i_pts = 0;

    // Encode frame
    x264_nal_t *nal;
    int i_nal;
    int frame_size = x264_encoder_encode(encoder, &nal, &i_nal, &pic_in, &pic_out);
    if (frame_size < 0)
    {
        std::cerr << "Failed to encode frame." << std::endl;
        return false;
    }

    // Store encoded data in the output vector
    for (int i = 0; i < i_nal; ++i)
    {
        outputData.insert(outputData.end(), nal[i].p_payload, nal[i].p_payload + nal[i].i_payload);
    }

    return true;
}