#include "JPEGEncoder.h"

bool JPEGEncoder::encodePixelData(const std::vector<uint8_t>& pixelData, const int width, const int height, std::vector<uint8_t>& outputData) {
    tjhandle _jpegCompressor = tjInitCompress();
    if (_jpegCompressor == nullptr) {
        std::cerr << "Failed to initialize jpeg compressor" << std::endl;
        return false;
    }

    unsigned char* compressedImage = nullptr;
    unsigned long compressedSize = 0;

    if (tjCompress2(
            _jpegCompressor,
            (unsigned char*)pixelData.data(),
            width,
            0, // pitch (0 = width * bytes per pixel)
            height,
            TJPF_RGB, // Pixel format
            &compressedImage,
            &compressedSize,
            TJSAMP_444, // Subsampling
            jpegQuality, // JPEG quality
            TJFLAG_FASTDCT) != 0)
    {
        std::cerr << "Failed to compress image: " << tjGetErrorStr() << std::endl;
        tjDestroy(_jpegCompressor);
        return false;
    }

    outputData.assign(compressedImage, compressedImage + compressedSize);

    tjFree(compressedImage);
    tjDestroy(_jpegCompressor);

    return true;
}