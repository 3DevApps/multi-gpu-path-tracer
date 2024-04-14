#pragma once

#include <memory>
#include <ctime>
#include <vector>
#include "Window.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>

class Renderer
{
public:
    Renderer(Window& window);
    ~Renderer() = default;
    void renderFrame(const std::vector<uint8_t> &frame);
private:
    Window& window_;
    std::uint32_t width_;
    std::uint32_t height_;
    GLuint fboId_;
    GLuint texId_;
};