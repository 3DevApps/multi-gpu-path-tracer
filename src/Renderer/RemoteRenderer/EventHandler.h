#pragma once

class EventHandler {
    public:
        virtual std::string getEventName() = 0;
        virtual void handleEvent(const std::string& message) = 0;
};