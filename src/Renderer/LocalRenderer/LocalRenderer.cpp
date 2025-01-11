
#include "LocalRenderer.h"
#include "Utils.h"

#ifdef USE_LOCAL_RENDERER

LocalRenderer::LocalRenderer(Window &window, std::shared_ptr<Framebuffer> framebuffer)
    : window_(window), width_(window.getWidth()), height_(window.getHeight()),
      framebuffer_(framebuffer)
{

    CheckedGLCall(glGenTextures(1, &texId_));
    CheckedGLCall(glGenFramebuffers(1, &fboId_));
}

void LocalRenderer::renderFrame()
{
    uint8_t *frame = framebuffer_->getRGBPtr();
    CheckedGLCall(glBindTexture(GL_TEXTURE_2D, texId_));
    CheckedGLCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width_, height_, 0, GL_RGB, GL_UNSIGNED_BYTE, frame));

    CheckedGLCall(glBindFramebuffer(GL_READ_FRAMEBUFFER, fboId_));
    CheckedGLCall(glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texId_, 0));

    CheckedGLCall(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
    CheckedGLCall(glBlitFramebuffer(0, 0, width_, height_, 0, 0, width_, height_, GL_COLOR_BUFFER_BIT, GL_NEAREST));
}

void LocalRenderer::renderFrame(long long duration)
{
    uint8_t *frame = framebuffer_->getRGBPtr();
    CheckedGLCall(glBindTexture(GL_TEXTURE_2D, texId_));
    CheckedGLCall(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width_, height_, 0, GL_RGB, GL_UNSIGNED_BYTE, frame));

    CheckedGLCall(glBindFramebuffer(GL_READ_FRAMEBUFFER, fboId_));
    CheckedGLCall(glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texId_, 0));

    CheckedGLCall(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0));
    CheckedGLCall(glBlitFramebuffer(0, 0, width_, height_, 0, 0, width_, height_, GL_COLOR_BUFFER_BIT, GL_NEAREST));
}


bool LocalRenderer::shouldStopRendering()
{
    window_.swapBuffers();
    window_.pollEvents();
    return window_.shouldClose();
}

#endif
