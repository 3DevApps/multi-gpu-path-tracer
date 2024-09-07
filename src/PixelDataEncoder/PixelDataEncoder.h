#pragma once
#include <vector>
#include <iostream>

class PixelDataEncoder {
    public:
        virtual bool encodePixelData(const std::vector<uint8_t>& pixelData, const int width, const int height, std::vector<uint8_t>& outputData) = 0;
};