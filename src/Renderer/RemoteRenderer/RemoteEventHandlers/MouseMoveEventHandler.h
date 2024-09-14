#pragma once

#include <cmath>
#include "EventHandler.h"

class MouseMoveEventHandler: EventHandler {
    public:
        MouseMoveEventHandler(CameraParams& camParams) : camParams(camParams) {};

        void handleMouseMove(double xpos, double ypos) {
            if (firstMouse) {
                lastX = (double)xpos;
                lastY = (double)ypos;
                firstMouse = false;
            }

            double xoffset = (double)xpos - lastX;
            double yoffset = lastY - (double)ypos; 
            lastX = xpos;
            lastY = ypos;

            xoffset *= sensitivity;
            yoffset *= sensitivity;

            yaw += xoffset;
            pitch += yoffset;

            if (pitch > 89.0f)
                pitch = 89.0f;
            if (pitch < -89.0f)
                pitch = -89.0f;

            camParams.front = normalize(make_float3(
                cos(getRadians(yaw)) * cos(getRadians(pitch)),
                sin(getRadians(pitch)), 
                sin(getRadians(yaw)) * cos(getRadians(pitch))));
        };

        void handleEvent(const std::string& message) override {
            std::cout << "Handling mouse move event" << std::endl;
            auto sepPos = message.find("#");
            auto xpos = std::stod(message.substr(0, sepPos));
            auto ypos = std::stod(message.substr(sepPos+1));
            handleMouseMove(xpos, ypos);
        };
        
        std::string getEventName() override {
            return "MOUSE_MOVE";
        };

    private:
        CameraParams& camParams;
        double sensitivity = 0.75f;
        bool firstMouse;
        float lastX;
        float lastY;
        float pitch;
        float yaw;

        double getRadians(double value) {
            return M_PI * value / 180.0;
        }
};