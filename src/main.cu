#include <iostream>
#include <vector>
#include <cassert>
#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <fstream>

int main() {

    nvjpegHandle_t nv_handle;
    nvjpegEncoderState_t nv_enc_state;
    nvjpegEncoderParams_t nv_enc_params;
    cudaStream_t stream;

    cudaStreamCreate(&stream);

    // initialize nvjpeg structures
    nvjpegCreateSimple(&nv_handle);
    nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream);
    nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream);

    nvjpegImage_t nv_image;
    // Fill nv_image with image data, let's say 640x480 image in RGB format
    auto imageBuffer = new unsigned char[640 * 480 * 3];
    // Set 255 to all pixels
    for (int i = 0; i < 640 * 480 * 3; i++) {
        imageBuffer[i] = 255;
    }
    
    nv_image.channel[0] = imageBuffer;
    nv_image.pitch[0] = 3 * 640;

    // Compress image
    nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params,
        &nv_image, NVJPEG_INPUT_RGB, 640, 480, stream);

    // get compressed stream size
    size_t length;
    nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream);
    // get stream itself
    cudaStreamSynchronize(stream);
    unsigned char *data = new unsigned char[length];
    nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, data, &length, stream);

    // Save compressed stream to file
    std::ofstream file("compressed.jpg", std::ios::binary);
    file.write(reinterpret_cast<char*>(data), length);
    file.close();



    return 0;
}
