#pragma once

#ifdef USE_LOCAL_RENDERER

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <stdexcept>
#include <memory>
#include <vector>
#include <functional>
#include <string>
#include <iostream>
#include "../../HostScene.h"
#include "../../CameraParams.h"

enum class MouseButton {
    Left,
    Right,
    Middle,
};

class Window {
public:
    Window(Window const&) = delete;
    Window& operator=(Window const&) = delete;

    Window(std::uint32_t width, std::uint32_t height, const std::string& title, CameraParams& camParams);
    ~Window();

    void getMousePos(int& x, int& y) const;
    void setMousePos(int x, int y) const;
    void pollEvents();
    bool shouldClose() const;
    void swapBuffers();
    void addScrollCallback(std::function<void(float)> callback) { scroll_callbacks_.push_back(callback); }
    std::uint32_t getWidth() const;
    std::uint32_t getHeight() const;
    bool getMouseButton(MouseButton button) const;
    bool newEvent();

    //public for static callbacks
    CameraParams& camParams;
    bool firstMouse = 0;
    float lastX = 0;
    float lastY = 0;
    float pitch = 0;
    float yaw = 0;    
    int newEvent_ = 0;

private:
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos);

    std::unique_ptr<GLFWwindow, void(*)(GLFWwindow*)> window_;
    std::vector<std::function<void(float)>> scroll_callbacks_;
    std::uint32_t width_ = 0u;
    std::uint32_t height_ = 0u;

};

#endif