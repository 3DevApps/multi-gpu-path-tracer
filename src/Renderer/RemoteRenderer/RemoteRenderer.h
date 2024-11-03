#pragma once

#include <string>
#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <ixwebsocket/IXUserAgent.h>
#include <ixwebsocket/IXWebSocketSendData.h>
#include "../Renderer.h"
#include "../../PixelDataEncoder/PixelDataEncoder.h"
#include "../../PixelDataEncoder/JPEGEncoder.h"
#include "../../PixelDataEncoder/PNGEncoder.h"
#include "../../RendererConfig.h"
#include "../../CameraConfig.h"

class RemoteRenderer : public Renderer {
    public:
        using LambdaFunction = std::function<void(std::string)>;

        RemoteRenderer(std::string& jobId, RendererConfig& config);
        ~RemoteRenderer();
        std::vector<uint8_t> processFrame(const uint8_t *frame, bool useJpegEncoder = true);
        void renderFrame(const uint8_t *frame) override;
        void send(const std::string& data) override;
        bool shouldStopRendering() override;
        void addMessageListener(std::string eventName, LambdaFunction listener);
        void removeMessageListener(std::string eventName);
        void generateAndSendSnapshot();
    private:
        const std::string SERVER_URL = "wss://pathtracing-relay-server.klatka.it/?path-tracing-job=true&jobId=";
        std::string& jobId;
        ix::WebSocket webSocket;
        std::unordered_map<std::string, LambdaFunction> eventListeners;
        std::uint32_t view_width;
        std::uint32_t view_height;
        std::vector<uint8_t> pixelData;
        std::shared_ptr<PixelDataEncoder> jpegPixelDataEncoder;
        std::shared_ptr<PixelDataEncoder> pngPixelDataEncoder;
        bool stopRenderer = false;
        RendererConfig& config;
        bool generateAndSendSnapshotFlag = false;

        void onMessage(const ix::WebSocketMessagePtr& msg);
        void generateAndSendSnapshotIfNeeded(const uint8_t *frame);
};