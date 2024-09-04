#pragma once

#include <memory>
#include <ctime>
#include <vector>
#include "Window.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "../Renderer.h"

class LocalRenderer : public Renderer {
public:
    LocalRenderer(Window& window);
    ~LocalRenderer() = default;
    void renderFrame(const uint8_t *frame) override;
    bool shouldStopRendering() override;
    void send(const std::string& data) override {};
private:
    Window& window_;
    std::uint32_t width_;
    std::uint32_t height_;
    GLuint fboId_;
    GLuint texId_;
};