#pragma once

#include <cmath>
#include "EventHandler.h"

class CameraEventHandler : EventHandler
{
public:
    CameraEventHandler(CameraConfig &cameraConfig) : cameraConfig(cameraConfig) {};

    void handleEvent(const std::string &message) override
    {
        auto sepPos = message.find("#");
        auto command = message.substr(0, sepPos);
        auto parsedRawData = message.substr(sepPos + 1);

        if (command == "FORWARD")
        {
            auto speed = std::stof(parsedRawData);
            cameraConfig.lookFrom += speed * cameraConfig.front;
        }
        else if (command == "BACKWARD")
        {
            auto speed = std::stof(parsedRawData);
            cameraConfig.lookFrom -= speed * cameraConfig.front;
        }
        else if (command == "LEFT")
        {
            auto speed = std::stof(parsedRawData);
            cameraConfig.lookFrom += speed * cross(cameraConfig.front, make_float3(0.0f, -1.0f, 0.0f));
        }
        else if (command == "RIGHT")
        {
            auto speed = std::stof(parsedRawData);
            cameraConfig.lookFrom += speed * cross(cameraConfig.front, make_float3(0.0f, 1.0f, 0.0f));
        }
        else if (command == "UP")
        {
            auto speed = std::stof(parsedRawData);
            cameraConfig.lookFrom += speed * cross(cameraConfig.front, make_float3(-1.0f, 0.0f, 0.0f));
        }
        else if (command == "DOWN")
        {
            auto speed = std::stof(parsedRawData);
            cameraConfig.lookFrom += speed * cross(cameraConfig.front, make_float3(1.0f, 0.0f, 0.0f));
        }
        else if (command == "FOV-")
        {
            cameraConfig.vfov += 1.0f;
            cameraConfig.hfov += 1.0f;
        }
        else if (command == "FOV+")
        {
            cameraConfig.vfov -= 1.0f;
            cameraConfig.hfov -= 1.0f;
        }
        else if (command == "PITCH_YAW")
        {
            auto sepPos = parsedRawData.find("#");
            auto pitch = std::stof(parsedRawData.substr(0, sepPos));
            auto yaw = std::stof(parsedRawData.substr(sepPos + 1));
            cameraConfig.front = normalize(make_float3(
                cos(getRadians(yaw)) * cos(getRadians(pitch)),
                sin(getRadians(pitch)),
                sin(getRadians(yaw)) * cos(getRadians(pitch))));
        }
        else if (command == "SCENE_POSITION")
        {
            auto sepPos1 = parsedRawData.find("#");
            auto sepPos2 = parsedRawData.find("#", sepPos1 + 1);
            auto x = std::stof(parsedRawData.substr(0, sepPos1));
            auto y = std::stof(parsedRawData.substr(sepPos1 + 1, sepPos2 - sepPos1 - 1));
            auto z = std::stof(parsedRawData.substr(sepPos2 + 1));
            cameraConfig.lookFrom = make_float3(x, y, z);
        }
    };

    std::string getEventName() override
    {
        return "CAMERA_EVENT";
    };

private:
    CameraConfig &cameraConfig;
};