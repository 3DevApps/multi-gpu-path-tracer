#pragma once

#include <cstdlib>
#include <iostream>

#include <curand_kernel.h>
#include "helper_math.h"

class HostTexture {
  public:
    ~HostTexture();
    bool load(const std::string& filename);

    int width() const;
    int height() const;

    static unsigned char float_to_byte(float value) {
        if (value <= 0.0)
            return 0;
        if (1.0 <= value)
            return 255;
        return static_cast<unsigned char>(256.0 * value);
    }

    void convert_to_bytes();

    float3 *data; // pixel data as RGB float3
    int index;
private:
    const int bytes_per_pixel = 3;
    float *fdata = nullptr;         // Linear floating point pixel data
    unsigned char *bdata = nullptr;         // Linear 8-bit pixel data
    int image_width = 0;         // Loaded image width
    int image_height = 0;        // Loaded image height
    int bytes_per_scanline = 0;
};