#define STB_IMAGE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include "../third-party/stb_image.h"

#include <cstdlib>
#include <iostream>
#include "HostTexture.h"

HostTexture::~HostTexture() {
    delete[] bdata;
    STBI_FREE(fdata);
}

bool HostTexture::load(const std::string& filename) {
    auto n = bytes_per_pixel; 
    fdata = stbi_loadf(filename.c_str(), &image_width, &image_height, &n, bytes_per_pixel);
    if (fdata == nullptr) return false;

    bytes_per_scanline = image_width * bytes_per_pixel;
    convert_to_bytes();
    return true;
}

int HostTexture::width() const { return (fdata == nullptr) ? 0 : image_width; }
int HostTexture::height() const { return (fdata == nullptr) ? 0 : image_height; }

void HostTexture::convert_to_bytes() {
    int total_bytes = image_width * image_height * bytes_per_pixel;
    bdata = new unsigned char[total_bytes];

    data = new float3[image_width * image_height];

    // Iterate through all pixel components, converting from [0.0, 1.0] float values to
    // unsigned [0, 255] byte values.

    auto *bptr = bdata;
    auto *fptr = fdata;
    for (auto i=0; i < total_bytes; i++, fptr++, bptr++) {
        *bptr = float_to_byte(*fptr);
    }

    for (int i = 0, j = 0; i < total_bytes; i += 3, j += 1) {
        data[j] = make_float3(bdata[i], bdata[i + 1], bdata[i + 2]);  
    }
}