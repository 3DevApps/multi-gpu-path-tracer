#pragma once 

#include <vector>
#include "RemoteRenderer.h"
#include "EventHandler.h"
#include "EventHandlers/MouseMoveEventHandler.h"
#include "../../CameraParams.h"

class RemoteEventHandlers {
    public:
        RemoteEventHandlers(RemoteRenderer &remoteRenderer, CameraParams& camParams) {
            addEventHandler<MouseMoveEventHandler>(remoteRenderer, camParams);
        };
        
        template <typename T, typename... Args>
        void addEventHandler(RemoteRenderer &remoteRenderer, Args&&... args) {
            static_assert(std::is_base_of<EventHandler, T>::value, "T must inherit from EventHandler");
            auto handler = std::make_shared<T>(std::forward<Args>(args)...);
            remoteRenderer.addMessageListener(handler->getEventName(), [handler](const std::string& message) {
                handler->handleEvent(message);
            });
        }
};