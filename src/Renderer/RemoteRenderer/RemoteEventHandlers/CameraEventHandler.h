#pragma once

#include <cmath>
#include <iostream>
#include "EventHandler.h"

class CameraEventHandler : EventHandler
{
public:
    CameraEventHandler(CameraConfig &cameraConfig) : cameraConfig(cameraConfig) {};

    void handleEvent(const Event &event) override
    {
        auto cameraEvent = event.camera();
        switch (cameraEvent.type()) {
            case CameraEvent::FORWARD:
                cameraConfig.lookFrom += cameraConfig.front * cameraEvent.movespeed();
                break;
            case CameraEvent::BACKWARD:
                cameraConfig.lookFrom -= cameraConfig.front * cameraEvent.movespeed();
                break;
            case CameraEvent::LEFT:
                cameraConfig.lookFrom += cross(cameraConfig.front, make_float3(0.0f, -1.0f, 0.0f)) * cameraEvent.movespeed();
                break;
            case CameraEvent::RIGHT:
                cameraConfig.lookFrom += cross(cameraConfig.front, make_float3(0.0f, 1.0f, 0.0f)) * cameraEvent.movespeed();
                break;
            case CameraEvent::UP:
                cameraConfig.lookFrom += cross(cameraConfig.front, make_float3(-1.0f, 0.0f, 0.0f)) * cameraEvent.movespeed();
                break;
            case CameraEvent::DOWN:
                cameraConfig.lookFrom += cross(cameraConfig.front, make_float3(1.0f, 0.0f, 0.0f)) * cameraEvent.movespeed();
                break;
            case CameraEvent::FOV_DECREASE:
                cameraConfig.vfov += 1.0f;
                cameraConfig.hfov += 1.0f;
                break;
            case CameraEvent::FOV_INCREASE:
                cameraConfig.vfov -= 1.0f;
                cameraConfig.hfov -= 1.0f;
                break;
            case CameraEvent::PITCH_YAW:
                cameraConfig.yaw = cameraEvent.rotation().yaw();
                cameraConfig.pitch = cameraEvent.rotation().pitch();
                cameraConfig.front = normalize(make_float3(
                    cos(getRadians(cameraConfig.yaw)) * cos(getRadians(cameraConfig.pitch)),
                    sin(getRadians(cameraConfig.pitch)),
                    sin(getRadians(cameraConfig.yaw)) * cos(getRadians(cameraConfig.pitch))));
                break;
            case CameraEvent::SCENE_POSITION:
                cameraConfig.lookFrom = make_float3(cameraEvent.position().x(), cameraEvent.position().y(), cameraEvent.position().z());
                break;
            default:
                break;
        }
    };

    Event::EventType getEventType() override
    {
        return Event::CAMERA_EVENT;
    };

private:
    CameraConfig &cameraConfig;

    double getRadians(double value) {
        return M_PI * value / 180.0;
    }
};
