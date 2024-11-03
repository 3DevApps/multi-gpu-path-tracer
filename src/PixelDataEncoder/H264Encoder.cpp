#include "H264Encoder.h"

H264Encoder::~H264Encoder()
{
    destroyEncoder();
}

void H264Encoder::initEncoder(const int width, const int height)
{
    int fps = 60;
    x264_param_t param;

    param.i_fps_num = fps;
    param.i_fps_den = 1;
    param.i_csp = X264_CSP_I420;
    param.i_keyint_max = fps;
    param.i_bframe = 0;
    param.b_intra_refresh = 1;
    param.b_repeat_headers = 1; // Force SPS/PPS on keyframes
    param.b_annexb = 1;         // Use Annex B format with start codes

    x264_param_default_preset(&param, "ultrafast", "zerolatency,ssim");
    x264_param_apply_profile(&param, "high");
    param.i_width = width;
    param.i_height = height;
    param.i_level_idc = 52;

    encoder = x264_encoder_open(&param);
    if (!encoder)
    {
        throw std::runtime_error("Failed to open encoder");
    }

    if (x264_picture_alloc(&pic_in, X264_CSP_I420, width, height) < 0)
    {
        throw std::runtime_error("Failed to allocate picture");
    }

    this->width = width;
    this->height = height;
}

void H264Encoder::updateEncoderIfNeeded(const int width, const int height)
{
    if (this->width != width || this->height != height)
    {
        destroyEncoder();
        initEncoder(width, height);
    }
}

void H264Encoder::destroyEncoder()
{
    if (encoder)
    {
        x264_encoder_close(encoder);
        x264_picture_clean(&pic_in);
    }
}

bool H264Encoder::encodePixelData(const uint8_t *frame, const int width, const int height, std::vector<uint8_t> &outputData)
{
    updateEncoderIfNeeded(width, height);

    const int ySize = width * height;
    const int uvSize = (width * height) / 4;

    uint8_t *y = pic_in.img.plane[0];
    uint8_t *u = pic_in.img.plane[1];
    uint8_t *v = pic_in.img.plane[2];

    std::memcpy(y, frame, ySize);
    std::memcpy(u, frame + ySize, uvSize);
    std::memcpy(v, frame + ySize + uvSize, uvSize);

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
        std::cout << "Failed to encode frame" << std::endl;
        return false;
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