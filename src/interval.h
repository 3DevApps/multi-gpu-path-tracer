#pragma once

class interval {
  public:
    float min, max;
    __device__ __host__ interval() : min(0), max(0) {}
    __device__ __host__ interval(float a, float b) : min(a), max(b) {}
    __device__ __host__ interval(interval a , interval b) : min(fmin(a.min, b.min)), max(fmax(a.max, b.max)) {}
    __device__ __host__ float size() const {
        return max - min;
    }

    //add padding to interval
    __device__ __host__ interval expand(float delta) const {
        auto padding = delta/2;
        return interval(min - padding, max + padding);
    }
};