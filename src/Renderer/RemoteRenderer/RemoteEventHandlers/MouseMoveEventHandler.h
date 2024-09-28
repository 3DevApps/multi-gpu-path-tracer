#pragma once

#include <cmath>
#include "EventHandler.h"
#include "../../../HostScene.h"

class MouseMoveEventHandler: EventHandler {
    public:
        MouseMoveEventHandler(CameraParams& camParams) : camParams(camParams) {
            yaw = getDegrees(atan2(camParams.front.z, camParams.front.x));
            pitch = getDegrees(asin(camParams.front.y));
        };

        void handleMouseMove(double xoffset, double yoffset) {
            xoffset *= sensitivity;
            yoffset *= -sensitivity;

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
            auto sepPos = message.find("#");
            auto xoffset = std::stod(message.substr(0, sepPos));
            auto yoffset = std::stod(message.substr(sepPos+1));
            handleMouseMove(xoffset, yoffset);
        };
        
        std::string getEventName() override {
            return "MOUSE_MOVE";
        };

    private:
        CameraParams& camParams;
        double sensitivity = 0.75f;
        float pitch;
        float yaw;

        double getRadians(double value) {
            return M_PI * value / 180.0;
        }

        float getDegrees(float radians) {
            return radians * 180.0 / M_PI;
        }
};