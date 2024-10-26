#include "AVIFEncoder.h"

// heif_error writer_write(struct heif_context *ctx, const void *data, size_t size, void *userdata)
// {
//     AVIFEncoder::MemoryWriter *writer = static_cast<AVIFEncoder::MemoryWriter *>(userdata);
//     writer->write(data, size);
//     struct heif_error err
//     {
//         heif_error_Ok, heif_suberror_Unspecified, nullptr
//     };
//     return err;
// }

void AVIFEncoder::fill_new_plane_from_vector(heif_image *img, heif_channel channel, int w, int h, const std::vector<uint8_t> &data) const
{
    heif_error err;

    err = heif_image_add_plane(img, channel, w, h, 8);
    if (err.code != heif_error_Ok)
    {
        return;
    }

    int stride;
    uint8_t *p = heif_image_get_plane(img, channel, &stride);

    for (int y = 0; y < h; y++)
    {
        memcpy(p + y * stride, data.data() + y * w, w);
    }
}

AVIFEncoder::AVIFEncoder()
{
    ctx = heif_context_alloc();
    err = heif_context_get_encoder_for_format(ctx, heif_compression_AV1, &enc);

    if (err.code != heif_error_Ok)
    {
        std::cerr << "Failed to get encoder" << std::endl;
        heif_context_free(ctx);
        return;
    }

    options = heif_encoding_options_alloc();

    w.writer_api_version = 1;
    // w.write = writer_write;
}

AVIFEncoder::~AVIFEncoder()
{
    heif_encoder_release(enc);
    heif_encoding_options_free(options);
    heif_context_free(ctx);
}

bool AVIFEncoder::encodePixelData(const std::vector<uint8_t> &pixelData, const int input_width, const int input_height, std::vector<uint8_t> &outputData)
{
    heif_image *img;
    heif_image_handle *handle;

    err = heif_image_create(input_width, input_height, heif_colorspace_RGB, heif_chroma_interleaved_RGB, &img);
    if (err.code != heif_error_Ok)
    {
        std::cerr << "Failed to create image" << std::endl;
        return false;
    }

    fill_new_plane_from_vector(img, heif_channel_interleaved, input_width * 3, input_height, pixelData);

    err = heif_context_encode_image(ctx, img, enc, options, &handle);
    if (err.code != heif_error_Ok)
    {
        std::cerr << "Failed to encode image" << std::endl;
        heif_image_release(img);
        return false;
    }

    heif_context_write(ctx, &w, &writer);

    outputData.clear();
    outputData.reserve(writer.size_);
    outputData.insert(outputData.end(), writer.data_, writer.data_ + writer.size_);

    heif_image_handle_release(handle);
    heif_image_release(img);

    return true;
}