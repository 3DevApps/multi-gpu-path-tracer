#pragma once

#include <string>
#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXUserAgent.h>
#include <ixwebsocket/IXWebSocketSendData.h>
#include "../Renderer.h"
#include "../../PixelDataEncoder/PixelDataEncoder.h"
#include "../../PixelDataEncoder/JPEGEncoder.h"

class RemoteRenderer : public Renderer {
    public:
        using LambdaFunction = std::function<void()>;

        RemoteRenderer(std::string& jobId, std::uint32_t view_width, std::uint32_t view_height);
        ~RemoteRenderer();
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
        std::vector<uint8_t> pixelData;
        std::shared_ptr<PixelDataEncoder> pixelDataEncoder;

        void onMessage(const ix::WebSocketMessagePtr& msg);
};