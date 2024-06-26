#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdint>
#include <memory>
#include <vector>
#include <functional>
#include <string>

enum class MouseButton {
    Left,
    Right,
    Middle,
};


class Window {
public:
    Window(Window const&) = delete;
    Window& operator=(Window const&) = delete;

    Window(std::uint32_t width, std::uint32_t height, const std::string& title);
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

private:
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
    std::unique_ptr<GLFWwindow, void(*)(GLFWwindow*)> window_;
    std::vector<std::function<void(float)>> scroll_callbacks_;
    std::uint32_t width_ = 0u;
    std::uint32_t height_ = 0u;
};