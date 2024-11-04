#pragma once
#include "PixelDataEncoder.h"
#include <vector>
#include <png.h>

class PNGEncoder : public PixelDataEncoder
{
public:
    bool encodePixelData(const uint8_t *frame, const int width, const int height, std::vector<uint8_t> &outputData) override;

private:
    static void writePNGDataToVector(png_structp png_ptr, png_bytep data, png_size_t length);
};