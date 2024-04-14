#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "Window.h"
#include <stdexcept>
#include <cstdio>

const std::unordered_map<MouseButton, int> mouseButtonToGlfwButton = {
    { MouseButton::Left,   GLFW_MOUSE_BUTTON_LEFT   },
    { MouseButton::Right,  GLFW_MOUSE_BUTTON_RIGHT  },
    { MouseButton::Middle, GLFW_MOUSE_BUTTON_MIDDLE },
};

Window::Window(std::uint32_t width, std::uint32_t height, const std::string& title)
    : window_(nullptr, glfwDestroyWindow)
    , width_(width)
    , height_(height) {

    if (!glfwInit()) {
        throw std::runtime_error("glfwInit() failed");
    }
    // glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window_.reset(glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr));
    if (!window_.get()) {
        throw std::runtime_error("Failed to create GLFW window!");
    }

    glfwMakeContextCurrent(window_.get());
    glfwSetWindowUserPointer(window_.get(), this);
    glfwSetScrollCallback(window_.get(), scrollCallback);

    GLenum glew_status = glewInit();

    if (glew_status != GLEW_OK) {
        fprintf(stderr, "Error: %s\n", glewGetErrorString(glew_status));
        throw std::runtime_error("Failed to init GLEW");
    }
}

Window::~Window() {
    glfwTerminate();
}

std::uint32_t Window::getWidth() const {
    return width_;
}

std::uint32_t Window::getHeight() const {
    return height_;
}

void Window::pollEvents() {
    glfwPollEvents();
}

bool Window::shouldClose() const {
    return glfwWindowShouldClose(window_.get());
}

bool Window::getMouseButton(MouseButton button) const {
    auto b = mouseButtonToGlfwButton.find(button);
    return glfwGetMouseButton(window_.get(), b->second) == GLFW_PRESS;
}

void Window::getMousePos(int& x, int& y) const {
    double xpos, ypos;
    glfwGetCursorPos(window_.get(), &xpos, &ypos);
    x = static_cast<int>(xpos);
    y = static_cast<int>(ypos);
}

void Window::setMousePos(int x, int y) const {
    glfwSetCursorPos(window_.get(), (double)x, (double)y);
}

void Window::swapBuffers() {
    glfwSwapBuffers(window_.get());
}

void Window::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    Window* this_window = (Window*)glfwGetWindowUserPointer(window);
    for (auto callback : this_window->scroll_callbacks_)
    {
        callback(yoffset);
    }
}