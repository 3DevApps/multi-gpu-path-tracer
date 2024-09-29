#pragma once
#include "PixelDataEncoder.h"
#include <vector>
#include <libheif/heif.h>
#include <cstdint>

class AVIFEncoder : public PixelDataEncoder {
    public:
        bool encodePixelData(const std::vector<uint8_t>& pixelData, const int width, const int height, std::vector<uint8_t>& outputData) override;
};