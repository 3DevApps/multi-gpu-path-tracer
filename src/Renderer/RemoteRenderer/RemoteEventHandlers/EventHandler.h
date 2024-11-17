#pragma once
#include "../../../genproto/main.pb.h"

class EventHandler {
    public:
        virtual Event::EventType getEventType() = 0;
        virtual void handleEvent(const Event& event) = 0;
};
