#pragma once
#include "PixelDataEncoder.h"
#include <vector>
#include <turbojpeg.h> 
#include <cstdint>

class JPEGEncoder : public PixelDataEncoder {
    public:
        JPEGEncoder() : jpegQuality(100) {}
        JPEGEncoder(int jpegQuality) : jpegQuality(jpegQuality) {}
        bool encodePixelData(const std::vector<uint8_t>& pixelData, const int width, const int height, std::vector<uint8_t>& outputData) override;
    private:
        int jpegQuality;
};