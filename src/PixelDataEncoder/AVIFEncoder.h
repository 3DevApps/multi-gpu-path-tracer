#pragma once
#include "PixelDataEncoder.h"
#include <vector>
#include <libheif/heif.h>
#include <cstdint>
#include <stdexcept>
#include <cstring>
#include <iostream>

class AVIFEncoder : public PixelDataEncoder
{
public:
    class MemoryWriter
    {
    public:
        MemoryWriter() : data_(nullptr), size_(0), capacity_(0)
        {
        }

        ~MemoryWriter()
        {
            free(data_);
        }

        const uint8_t *data() const
        {
            return data_;
        }

        size_t size() const
        {
            return size_;
        }

        void write(const void *data, size_t size)
        {
            if (capacity_ - size_ < size)
            {
                size_t new_capacity = capacity_ + size;
                uint8_t *new_data = static_cast<uint8_t *>(malloc(new_capacity));
                if (data_)
                {
                    memcpy(new_data, data_, size_);
                    free(data_);
                }
                data_ = new_data;
                capacity_ = new_capacity;
            }
            memcpy(&data_[size_], data, size);
            size_ += size;
        }

    public:
        uint8_t *data_;
        size_t size_;
        size_t capacity_;
    };

    AVIFEncoder();
    ~AVIFEncoder();

    heif_error err;
    heif_context *ctx;
    heif_encoder *enc;
    heif_encoding_options *options;
    heif_writer w;
    AVIFEncoder::MemoryWriter writer;

    void fill_new_plane_from_vector(heif_image *img, heif_channel channel, int w, int h, const std::vector<uint8_t> &data) const;
    bool encodePixelData(const std::vector<uint8_t> &pixelData, const int width, const int height, std::vector<uint8_t> &outputData) override;
};