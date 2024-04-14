
#include "Renderer.h"
#include "Utils.h"

Renderer::Renderer(Window& window)
    : window_(window)
    , width_(window.getWidth())
    , height_(window.getHeight()) {

    CheckedGLCall(glGenTextures(1, &texId_));
    CheckedGLCall(glGenFramebuffers(1, &fboId_));
}

void Renderer::renderFrame(const std::vector<uint8_t> &frame) {
    CheckedGLCall(glBindTexture(GL_TEXTURE_2D, texId_));
    CheckedGLCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width_, height_, 0, GL_RGB, GL_UNSIGNED_BYTE, frame.data()));

    CheckedGLCall(glBindFramebuffer(GL_READ_FRAMEBUFFER, fboId_));
    CheckedGLCall(glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texId_, 0));

    CheckedGLCall(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)); 
    CheckedGLCall(glBlitFramebuffer(0, 0, width_, height_, 0, 0, width_, height_, GL_COLOR_BUFFER_BIT, GL_NEAREST));
}
