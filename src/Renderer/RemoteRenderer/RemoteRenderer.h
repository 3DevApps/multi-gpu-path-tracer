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
#include "../../PixelDataEncoder/H264Encoder.h"
#include "../../RendererConfig.h"
#include "../../CameraConfig.h"
#include "../../Framebuffer.h"
#include "../../genproto/main.pb.h"

class RemoteRenderer : public Renderer
{
public:
    using LambdaFunction = std::function<void(Event&)>;

    RemoteRenderer(std::string &jobId, RendererConfig &config, std::shared_ptr<Framebuffer> &framebuffer);
    ~RemoteRenderer();
    std::vector<uint8_t> processFrameForStreaming(const uint8_t *frame);
    std::vector<uint8_t> processFrameForSnapshot(const uint8_t *frame);
    void renderFrame() override;
    void send(const std::string &data) override;
    bool shouldStopRendering() override;
    void addMessageListener(Event::EventType eventType, LambdaFunction listener);
    void removeMessageListener(Event::EventType eventType);
    void generateAndSendSnapshot();

private:
    const std::string SERVER_URL = "wss://pathtracing-relay-server.klatka.it/?path-tracing-job=true&jobId=";
    const std::string STREAMING_SERVER_URL = "wss://pathtracing-streaming-server.klatka.it/?path-tracing-job=true&jobId=";
    std::string &jobId;
    ix::WebSocket webSocket;
    ix::WebSocket streamingWebSocket;
    std::unordered_map<Event::EventType, LambdaFunction> eventListeners;
    std::uint32_t view_width;
    std::uint32_t view_height;
    std::shared_ptr<PixelDataEncoder> snapshotDataEncoder;
    std::shared_ptr<PixelDataEncoder> frameDataEncoder;
    bool stopRenderer = false;
    RendererConfig &config;
    bool generateAndSendSnapshotFlag = false;
    std::shared_ptr<Framebuffer> &framebuffer;

    void onMessage(const ix::WebSocketMessagePtr &msg);
    void generateAndSendSnapshotIfNeeded();
    std::vector<uint8_t> prepareFrame(const uint8_t *frame, const std::shared_ptr<PixelDataEncoder> &pixelDataEncoder);
};
