#pragma once

#include <cstdint>

class Renderer
{
public:
    virtual void renderFrame() = 0;
    virtual bool shouldStopRendering() = 0;
    virtual void send(const std::string &data) = 0;
};