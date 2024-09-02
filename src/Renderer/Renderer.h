#pragma once

#include <cstdint>

class Renderer {
public:
    virtual void renderFrame(const uint8_t *frame) = 0;
};