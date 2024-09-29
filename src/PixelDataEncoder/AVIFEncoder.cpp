#include "AVIFEncoder.h"

// Function to encode std::vector<uint8_t> to AVIF
// std::vector<uint8_t> encodeToAVIF(const std::vector<uint8_t>& image_data, int width, int height, int stride) {
//     // Initialize heif_context and encoder
//     heif_context* ctx = heif_context_alloc();
//     if (!ctx) {
//         throw std::runtime_error("Failed to create HEIF context");
//     }

//     heif_image* img = nullptr;
//     heif_encoder* encoder = nullptr;

//     // Create a new image in heif_image format (8-bit, RGB, or other format)
//     heif_image_create(width, height, heif_colorspace_RGB, heif_chroma_interleaved_RGB, &img);

//     // Get the data planes from the heif_image object
//     uint8_t* dst_data = heif_image_get_plane(img, heif_channel_interleaved, nullptr);
//     if (!dst_data) {
//         heif_image_release(img);
//         heif_context_free(ctx);
//         throw std::runtime_error("Failed to get image plane");
//     }

//     // Copy raw image data to the heif_image data buffer
//     for (int y = 0; y < height; ++y) {
//         const uint8_t* src_row = &image_data[y * stride];
//         uint8_t* dst_row = &dst_data[y * stride];
//         std::copy(src_row, src_row + stride, dst_row);
//     }

//     // Set image options (optional, e.g., set color profile)
//     heif_image_set_premultiplied_alpha(img, false);

//     // Create an AVIF encoder
//     heif_context_get_encoder_for_format(ctx, heif_compression_AV1, &encoder);

//     // Set encoding parameters (quality, etc.)
//     heif_encoder_set_lossy_quality(encoder, 50); // Set quality to 50%

//     // Encode the image to a HEIF (AVIF) image
//     heif_context_encode_image(ctx, img, encoder, nullptr, nullptr);

//     // Release the encoder and image resources
//     heif_encoder_release(encoder);
//     heif_image_release(img);

//     // Create a std::vector<uint8_t> to hold the encoded AVIF data
//     std::vector<uint8_t> avif_data;

//     // Get the size of the encoded data
//     heif_writer writer;
//     writer.write = [](heif_context* context, const void* data, size_t size, void* user_context) -> heif_error {
//         auto* output = static_cast<std::vector<uint8_t>*>(user_context);
//         output->insert(output->end(), (uint8_t*)data, (uint8_t*)data + size);

//         // Return heif_error with success status
//         heif_error err;
//         err.code = heif_error_Ok;
//         err.message = nullptr;
//         return err;
//     };

//     // Write the encoded data into the std::vector
//     heif_context_write(ctx, &writer, &avif_data);

//     // Free the HEIF context
//     heif_context_free(ctx);

//     return avif_data;
// }

bool AVIFEncoder::encodePixelData(const std::vector<uint8_t>& pixelData, const int width, const int height, std::vector<uint8_t>& outputData) {
    // Initialize heif_context and encoder
    heif_context* ctx = heif_context_alloc();
    if (!ctx) {
        std::cerr << "Failed to create HEIF context" << std::endl;
        return false;
    }

    heif_image* img = nullptr;
    heif_encoder* encoder = nullptr;

    // Create a new image in heif_image format (8-bit, RGB, or other format)
    heif_error err = heif_image_create(width, height, heif_colorspace_RGB, heif_chroma_interleaved_RGB, &img);
    if (err.code != heif_error_Ok) {
        std::cerr << "Failed to create HEIF image: " << err.message << std::endl;
        heif_context_free(ctx);
        return false;
    }

    // Get the data planes from the heif_image object
    uint8_t* dst_data = heif_image_get_plane(img, heif_channel_interleaved, nullptr);
    if (!dst_data) {
        std::cerr << "Failed to get image plane" << std::endl;
        heif_image_release(img);
        heif_context_free(ctx);
        return false;
    }

    // Copy raw pixel data to the heif_image data buffer (assuming stride is width * 3 for RGB)
    const int stride = width * 3; // 3 bytes per pixel for RGB
    for (int y = 0; y < height; ++y) {
        const uint8_t* src_row = &pixelData[y * stride];
        uint8_t* dst_row = &dst_data[y * stride];
        std::copy(src_row, src_row + stride, dst_row);
    }

    // Set image options (optional, e.g., set color profile)
    heif_image_set_premultiplied_alpha(img, false);

    // Create an AVIF encoder
    err = heif_context_get_encoder_for_format(ctx, heif_compression_AV1, &encoder);
    if (err.code != heif_error_Ok) {
        std::cerr << "Failed to get AVIF encoder: " << err.message << std::endl;
        heif_image_release(img);
        heif_context_free(ctx);
        return false;
    }

    // Set encoding parameters (e.g., quality)
    heif_encoder_set_lossy_quality(encoder, 50); // Set quality to 50%

    // Encode the image to a HEIF (AVIF) image
    err = heif_context_encode_image(ctx, img, encoder, nullptr, nullptr);
    if (err.code != heif_error_Ok) {
        std::cerr << "Failed to encode image: " << err.message << std::endl;
        heif_encoder_release(encoder);
        heif_image_release(img);
        heif_context_free(ctx);
        return false;
    }

    // Release the encoder and image resources
    heif_encoder_release(encoder);
    heif_image_release(img);

    // Define the writer structure to save encoded data into the output vector
    heif_writer writer;
    writer.write = [](heif_context* context, const void* data, size_t size, void* user_context) -> heif_error {
        auto* output = static_cast<std::vector<uint8_t>*>(user_context);
        output->insert(output->end(), (uint8_t*)data, (uint8_t*)data + size);

        // Return heif_error with success status
        heif_error err;
        err.code = heif_error_Ok;
        err.message = nullptr;
        return err;
    };

    // Write the encoded data into the outputData vector
    err = heif_context_write(ctx, &writer, &outputData);
    if (err.code != heif_error_Ok) {
        std::cerr << "Failed to write AVIF data: " << err.message << std::endl;
        heif_context_free(ctx);
        return false;
    }

    // Free the HEIF context
    heif_context_free(ctx);

    return true;
}