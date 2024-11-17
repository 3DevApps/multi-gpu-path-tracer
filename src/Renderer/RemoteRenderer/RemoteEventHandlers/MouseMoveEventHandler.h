#pragma once

#include <cmath>
#include "EventHandler.h"
#include "../../../HostScene.h"

class MouseMoveEventHandler: EventHandler {
    public:
        MouseMoveEventHandler(CameraConfig& cameraConfig) : cameraConfig(cameraConfig) {
            cameraConfig.yaw = getDegrees(atan2(cameraConfig.front.z, cameraConfig.front.x));
            cameraConfig.pitch = getDegrees(asin(cameraConfig.front.y));
        };

        void handleMouseMove(double xoffset, double yoffset) {
            xoffset *= sensitivity;
            yoffset *= -sensitivity;

            cameraConfig.yaw += xoffset;
            cameraConfig.pitch += yoffset;

            if (cameraConfig.pitch > 89.0f)
                cameraConfig.pitch = 89.0f;
            if (cameraConfig.pitch < -89.0f)
                cameraConfig.pitch = -89.0f;

            cameraConfig.front = normalize(make_float3(
                cos(getRadians(cameraConfig.yaw)) * cos(getRadians(cameraConfig.pitch)),
                sin(getRadians(cameraConfig.pitch)),
                sin(getRadians(cameraConfig.yaw)) * cos(getRadians(cameraConfig.pitch))));
        };

        void handleEvent(const Event& event) override {
            auto mouseMoveEvent = event.mousemove();
            handleMouseMove(mouseMoveEvent.xoffset(), mouseMoveEvent.yoffset());
        };

        Event::EventType getEventType() override {
            return Event::MOUSE_MOVE;
        };

    private:
        CameraConfig& cameraConfig;
        double sensitivity = 0.75f;

        double getRadians(double value) {
            return M_PI * value / 180.0;
        }

        float getDegrees(float radians) {
            return radians * 180.0 / M_PI;
        }
};
