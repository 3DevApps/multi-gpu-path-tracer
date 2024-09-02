#pragma once

#include <string>
#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXUserAgent.h>
#include <ixwebsocket/IXWebSocketSendData.h>
#include "../Renderer.h"
#include "../../PixelDataEncoder/PixelDataEncoder.h"

class RemoteRenderer : public Renderer {
    public:
        using LambdaFunction = std::function<void()>;

        RemoteRenderer(std::string& jobId, std::uint32_t view_width, std::uint32_t view_height); 
        void renderFrame(const uint8_t *frame) override;
        void addMessageListener(std::string eventName, LambdaFunction listener);
        void removeMessageListener(std::string eventName);
    private:
        const std::string SERVER_URL = "wss://pathtracing-relay-server.klatka.it/?path-tracing-job=true&jobId=";
        std::string& jobId;
        ix::WebSocket webSocket;
        std::unordered_map<std::string, LambdaFunction> eventListeners;
        std::uint32_t view_width;
        std::uint32_t view_height;
        std::vector<uint8_t> pixelData
        PixelDataEncoder pixelDataEncoder;

        void onMessage(const ix::WebSocketMessagePtr& msg);
};