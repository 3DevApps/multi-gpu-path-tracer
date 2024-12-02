#pragma once

#include <cmath>
#include <iostream>
#include "EventHandler.h"

class SynchronizeEventHandler : EventHandler
{
public:
    SynchronizeEventHandler(RemoteRenderer &remoteRenderer, RenderManager &manager, HostScene &hScene, CameraConfig &cameraConfig) : remoteRenderer(remoteRenderer), manager(manager), hScene(hScene), cameraConfig(cameraConfig) {};

    void handleEvent(const Event &event) override {
        // // Create the event
        // Event syncEvent;
        // syncEvent.set_type(Event::CAMERA_EVENT);
        // CameraEvent cameraEvent;
        // cameraEvent.set_type(CameraEvent::PITCH_YAW);
        // CameraEvent_Rotation *rotation = new CameraEvent_Rotation();
        // cameraEvent.set_allocated_rotation(::CameraEvent_Rotation *value)

        // std::string serializedEvent;
        // syncEvent.SerializeToString(serializedEvent);

        // // Send the serialized event using websocket (assuming remoteRenderer has a send method)
        // remoteRenderer.send(serializedEvent);
    };

    Event::EventType getEventType() override
    {
        return Event::SYNCHRONIZE_EVENT;
    };

private:
    RemoteRenderer &remoteRenderer;
    RenderManager &manager;
    HostScene &hScene;
    CameraConfig &cameraConfig;
};
