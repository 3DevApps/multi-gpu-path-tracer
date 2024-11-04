#pragma once

#ifdef USE_LOCAL_RENDERER

#include <memory>
#include <ctime>
#include <vector>
#include "Window.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "../Renderer.h"
#include "../../Framebuffer.h"

class LocalRenderer : public Renderer
{
public:
    LocalRenderer(Window &window, std::shared_ptr<Framebuffer> framebuffer);
    ~LocalRenderer() = default;
    void renderFrame() override;
    bool shouldStopRendering() override;
    void send(const std::string &data) override {};

private:
    Window &window_;
    std::uint32_t width_;
    std::uint32_t height_;
    GLuint fboId_;
    GLuint texId_;
    std::shared_ptr<Framebuffer> framebuffer_;
};

#endif