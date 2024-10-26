#include <libheif/heif.h>
#include <vector>
#include <stdexcept>
#include <cstring> 
#include <iostream>
#include <chrono>

class MemoryWriter
{
public:
  MemoryWriter() : data_(nullptr), size_(0), capacity_(0)
  {}

  ~MemoryWriter()
  {
    free(data_);
  }

  const uint8_t* data() const
  { return data_; }

  size_t size() const
  { return size_; }

  void write(const void* data, size_t size)
  {
    if (capacity_ - size_ < size) {
      size_t new_capacity = capacity_ + size;
      uint8_t* new_data = static_cast<uint8_t*>(malloc(new_capacity));
      if (data_) {
        memcpy(new_data, data_, size_);
        free(data_);
      }
      data_ = new_data;
      capacity_ = new_capacity;
    }
    memcpy(&data_[size_], data, size);
    size_ += size;
  }

public:
  uint8_t* data_;
  size_t size_;
  size_t capacity_;
};

heif_error writer_write(struct heif_context* ctx, const void* data, size_t size, void* userdata)
{
  MemoryWriter* writer = static_cast<MemoryWriter*>(userdata);
  writer->write(data, size);
  struct heif_error err{heif_error_Ok, heif_suberror_Unspecified, nullptr};
  return err;
}

void fill_new_plane(heif_image* img, heif_channel channel, int w, int h)
{
  heif_error err;

  err = heif_image_add_plane(img, channel, w, h, 8);
    if (err.code != heif_error_Ok) {
        return;
    }

  int stride;
  uint8_t* p = heif_image_get_plane(img, channel, &stride);

  for (int y = 0; y < h; y++) {
    memset(p + y * stride, 255, w);
  }
}

int main() {

    auto start = std::chrono::high_resolution_clock::now();

    int input_width = 256, input_height = 256;

    heif_error err;
    heif_image* img;
    heif_image_create(input_width,input_height, heif_colorspace_YCbCr, heif_chroma_420, &img);
    fill_new_plane(img, heif_channel_Y, input_width, input_height);
    fill_new_plane(img, heif_channel_Cb, (input_width+1)/2, (input_height+1)/2);
    fill_new_plane(img, heif_channel_Cr, (input_width+1)/2, (input_height+1)/2);


    heif_context* ctx = heif_context_alloc();
    heif_encoder* enc;
    err = heif_context_get_encoder_for_format(ctx, heif_compression_AV1, &enc);

    if(err.code != heif_error_Ok) {
        std::cerr << "Failed to get encoder" << std::endl;
        heif_context_free(ctx);
        heif_image_release(img);
        return 1;
    }

    heif_encoding_options* options;
    options = heif_encoding_options_alloc();

    heif_image_handle* handle;
    err = heif_context_encode_image(ctx, img, enc, options, &handle);
    if (err.code != heif_error_Ok) {
        std::cerr << "Failed to encode image" << std::endl;
        heif_encoder_release(enc);
        heif_encoding_options_free(options);
        heif_context_free(ctx);
        heif_image_release(img);
        return 1;
    }

    err = heif_context_encode_image(ctx, img, enc, options, &handle);
    if (err.code != heif_error_Ok) {
        std::cerr << "Failed to encode image" << std::endl;
        heif_encoder_release(enc);
        heif_encoding_options_free(options);
        heif_context_free(ctx);
        heif_image_release(img);
        return 1;
    }


    // err = heif_context_write_to_file(ctx, "output.avif");
    // if (err.code != heif_error_Ok) {
    //     std::cerr << "Failed to write file" << std::endl;
    // }

    MemoryWriter writer;
    heif_writer w;
    w.writer_api_version = 1;
    w.write = writer_write;
    heif_context_write(ctx, &w, &writer);

    std::vector<uint8_t> output_data(writer.data_, writer.data_ + writer.size_);

    heif_image_handle_release(handle);
    heif_encoder_release(enc);
    heif_encoding_options_free(options);
    heif_context_free(ctx);
    heif_image_release(img);

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    return 0;
}
