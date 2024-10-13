
#pragma once 

class Texture {
  public:
    __device__ virtual float3 value(float2 texCoords, const float3& p) const = 0;
};

class SolidColorTexture : public Texture {
  public:
    __device__ SolidColorTexture(const float3& albedo) : albedo{albedo} {}

    __device__ SolidColorTexture(double red, double green, double blue) : SolidColorTexture(make_float3(red,green,blue)) {}

    __device__ float3 value(float2 texCoords, const float3& p) const override {
        return albedo;
    }

  private:
    float3 albedo;
};

class BaseColorTexture : public Texture {
  public:
    __device__ BaseColorTexture(int width, int height, float3* tex_data) {
      data_ = tex_data;
      width_ = width;
      height_ = height;
    }

    __device__ float3 value(float2 texCoords, const float3& p) const override {
        if (height_ <= 0) {
          return make_float3((float)242 / 255, (float)45 / 255, (float)27 / 255);
        } 

        float u  = texCoords.x;
        float v  = texCoords.y;

        u = fmod(u, 1.0f); 
        v = fmod(v, 1.0f);

        auto i = (int)(u * width_);
        auto j = (int)(v * height_);
        auto pixel = pixel_data(i, j);

        auto color_scale = 1.0 / 255.0;
        return make_float3(color_scale * pixel.x, color_scale * pixel.y, color_scale * pixel.z);
    }
  
    __device__ int clamp(int x, int low, int high) const {
        if (x < low) return low;
        if (x < high) return x;
        return high - 1;
    }

    __device__ float clamp(float x, float low, float high) const {
        if (x < low) return low;
        if (x < high) return x;
        return high;
    }

    __device__ float3 pixel_data(int x, int y) const {
        float3 magenta = make_float3(52, 27, 242);
        if (data_ == nullptr) return magenta;

        x = clamp(x, 0, width_);
        y = clamp(y, 0, height_);

        y = height_ - y;
        return data_[y * width_ + x]; 
    }

     float3 *data_ = nullptr;
     int width_ = 0;
     int height_ = 0;

  private:
};