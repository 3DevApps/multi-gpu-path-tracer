#pragma once 

#include <vector>
#include "../RemoteRenderer.h"
#include "EventHandler.h"
#include "MouseMoveEventHandler.h"
#include "RenderManagerEventHander.h"
#include "KeyboardEventHandler.h"
#include "../../../HostScene.h"
#include "../../../RenderManager.h"

class RemoteEventHandlers {
    public:
        RemoteEventHandlers(RemoteRenderer &remoteRenderer, RenderManager& manager, HostScene& hScene) {
            addEventHandler<MouseMoveEventHandler>(remoteRenderer, hScene.cameraParams);
            addEventHandler<RenderManagerEventHander>(remoteRenderer, manager, hScene);
            addEventHandler<KeyboardEventHandler>(remoteRenderer, hScene);
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