#pragma once

#include <cmath>
#include "EventHandler.h"

class KeyboardEventHandler: EventHandler {
    public:
        KeyboardEventHandler(HostScene &hScene) : hScene(hScene) {};

        void handleEvent(const std::string& message) override {
            auto sepPos = message.find("#");
            auto command = message.substr(0, sepPos);
            auto parsedRawData = message.substr(sepPos+1);

            if (command == "FORWARD") {
                auto speed = std::stof(parsedRawData);
                hScene.cameraParams.lookFrom += speed * hScene.cameraParams.front;
            } else if (command == "BACKWARD") {
                auto speed = std::stof(parsedRawData);
                hScene.cameraParams.lookFrom -= speed * hScene.cameraParams.front;
            } else if (command == "LEFT") {
                auto speed = std::stof(parsedRawData);
                hScene.cameraParams.lookFrom += speed * cross(hScene.cameraParams.front, make_float3(0.0f, -1.0f, 0.0f));
            } else if (command == "RIGHT") {
                auto speed = std::stof(parsedRawData);
                hScene.cameraParams.lookFrom += speed * cross(hScene.cameraParams.front, make_float3(0.0f, 1.0f, 0.0f));
            } else if (command == "UP") {
                auto speed = std::stof(parsedRawData);
                hScene.cameraParams.lookFrom += speed * cross(hScene.cameraParams.front, make_float3(-1.0f, 0.0f, 0.0f));
            } else if (command == "DOWN") {
                auto speed = std::stof(parsedRawData);
                hScene.cameraParams.lookFrom += speed * cross(hScene.cameraParams.front, make_float3(1.0f, 0.0f, 0.0f));
            } else if (command == "FOV-") {
                hScene.cameraParams.vfov += 1.0f;
                hScene.cameraParams.hfov += 1.0f;
            } else if (command == "FOV+") {
                hScene.cameraParams.vfov -= 1.0f;
                hScene.cameraParams.hfov -= 1.0f;
            } 
        };
        
        std::string getEventName() override {
            return "KEYBOARD_EVENT";
        };

    private:
        HostScene &hScene;
};