#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdint>
#include <memory>
#include <vector>
#include <functional>
#include <string>
#include <iostream>
// #include "../obj_loader.h"
#include "../CameraParams.h"

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

    //public for static callbacks
    CameraParams& camParams;
    bool firstMouse;
    float lastX;
    float lastY;
    float pitch;
    float yaw;

private:
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos);

    std::unique_ptr<GLFWwindow, void(*)(GLFWwindow*)> window_;
    std::vector<std::function<void(float)>> scroll_callbacks_;
    std::uint32_t width_ = 0u;
    std::uint32_t height_ = 0u;
};