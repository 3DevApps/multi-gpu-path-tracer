#pragma once
#include <vector>
#include <iostream>
#include <cstdint>

class PixelDataEncoder
{
public:
    virtual bool encodePixelData(const uint8_t *frame, const int width, const int height, std::vector<uint8_t> &outputData) = 0;
};