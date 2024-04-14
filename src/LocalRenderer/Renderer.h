#pragma once

#include <memory>
#include <ctime>
#include <vector>
#include "Window.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "cuda_runtime.h"
#include <curand_kernel.h>

class Renderer
{
public:
    Renderer(Window& window);
    ~Renderer() = default;
    void renderFrame(const uint8_t *frame);
private:
    Window& window_;
    std::uint32_t width_;
    std::uint32_t height_;
    GLuint fboId_;
    GLuint texId_;
};