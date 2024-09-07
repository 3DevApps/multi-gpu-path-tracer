#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "Window.h"
#include <stdexcept>
#include <cstdio>
#include <cmath>

const std::unordered_map<MouseButton, int> mouseButtonToGlfwButton = {
    { MouseButton::Left,   GLFW_MOUSE_BUTTON_LEFT   },
    { MouseButton::Right,  GLFW_MOUSE_BUTTON_RIGHT  },
    { MouseButton::Middle, GLFW_MOUSE_BUTTON_MIDDLE },
};

Window::Window(std::uint32_t width, std::uint32_t height, const std::string& title, CameraParams& camParams)
    : window_(nullptr, glfwDestroyWindow)
    , width_(width)
    , height_(height)
    , camParams(camParams) {

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
    glfwSetKeyCallback(window_.get(), keyCallback);
    glfwSetCursorPosCallback(window_.get(), cursorPositionCallback);
    glfwSetKeyCallback(window_.get(), keyCallback);

    GLenum glew_status = glewInit();

    // GLEW_ERROR_NO_GLX_DISPLAY occurs when running the program on a WSL system.
    // This error is not critical and can be ignored.
    if (glew_status != GLEW_OK && glew_status != GLEW_ERROR_NO_GLX_DISPLAY) {
        fprintf(stderr, "Error: %s\n", glewGetErrorString(glew_status));
        throw std::runtime_error("Failed to init GLEW");
    }

    // glfwSetInputMode(window_.get(), GLFW_CURSOR, GLFW_CURSOR_DISABLED);
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

void Window::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    Window* self = (Window*)glfwGetWindowUserPointer(window);
    const float cameraSpeed = 0.5f;

    // std::cout << self->camParams.lookFrom.x << " " << self->camParams.lookFrom.y << " " << self->camParams.lookFrom.z << std::endl;
    
    if (key == GLFW_KEY_W && (action == GLFW_PRESS || action == GLFW_REPEAT))
        self->camParams.lookFrom += cameraSpeed * self->camParams.front;
    if (key == GLFW_KEY_S && (action == GLFW_PRESS || action == GLFW_REPEAT))
        self->camParams.lookFrom -= cameraSpeed * self->camParams.front;
    if (key == GLFW_KEY_A && (action == GLFW_PRESS || action == GLFW_REPEAT))
        self->camParams.lookFrom -= normalize(cross(self->camParams.front, make_float3(0, 1, 0))) * cameraSpeed;
    if (key == GLFW_KEY_D && (action == GLFW_PRESS || action == GLFW_REPEAT))
        self->camParams.lookFrom += normalize(cross(self->camParams.front, make_float3(0, 1, 0))) * cameraSpeed;
}

float getRadians(float rad) {
    return rad * M_PI / 180;
}

void Window::cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
    Window* self = (Window*)glfwGetWindowUserPointer(window);
    
    if (self->firstMouse) {
        self->lastX = (double)xpos;
        self->lastY = (double)ypos;
        self->firstMouse = false;
    }

    double xoffset = (double)xpos - self->lastX;
    double yoffset = self->lastY - (double)ypos; 
    self->lastX = xpos;
    self->lastY = ypos;

    double sensitivity = 0.5f;
    xoffset *= sensitivity;
    yoffset *= sensitivity;

    self->yaw += xoffset;
    self->pitch += yoffset;

    if (self->pitch > 89.0f)
        self->pitch = 89.0f;
    if (self->pitch < -89.0f)
        self->pitch = -89.0f;

    self->camParams.front = normalize(make_float3(
        cos(getRadians(self->yaw)) * cos(getRadians(self->pitch)),
        sin(getRadians(self->pitch)), 
        sin(getRadians(self->yaw)) * cos(getRadians(self->pitch))));
}