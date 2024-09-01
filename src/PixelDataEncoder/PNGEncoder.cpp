#include "PNGEncoder.h"

void PNGEncoder::write_png_data_to_vector(png_structp png_ptr, png_bytep data, png_size_t length) {
    std::vector<uint8_t>* p = (std::vector<uint8_t>*)png_get_io_ptr(png_ptr);
    p->insert(p->end(), data, data + length);
}

bool PNGEncoder::encodePixelData(const std::vector<uint8_t>& pixelData, const int width, const int height, std::vector<uint8_t>& outputData) {
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) {
        std::cerr << "Failed to create png write struct" << std::endl;
        return false;
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, nullptr);
        std::cerr << "Failed to create png info struct" << std::endl;
        return false;
    }

    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        std::cerr << "Failed during png creation" << std::endl;
        return false;
    }

    // Set custom write function
    png_set_write_fn(png, &outputData, PNGEncoder::write_png_data_to_vector, nullptr);

    // Set the header
    png_set_IHDR(
        png,
        info,
        width, height,
        8,
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT
    );
    png_write_info(png, info);

    // Allocate memory for rows of pointers to each row's data
    std::vector<uint8_t*> row_pointers(height);
    for (int y = 0; y < height; ++y) {
        row_pointers[y] = (uint8_t*)&pixelData[y * width * 3];
    }

    png_write_image(png, row_pointers.data());
    png_write_end(png, nullptr);

    png_destroy_write_struct(&png, &info);

    return true;
}

