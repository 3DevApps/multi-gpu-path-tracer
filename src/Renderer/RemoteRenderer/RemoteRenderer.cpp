#include "RemoteRenderer.h"

RemoteRenderer::RemoteRenderer(std::string &jobId, RendererConfig &config, std::shared_ptr<Framebuffer> &framebuffer)
    : jobId(jobId), config(config), framebuffer(framebuffer)
{
    snapshotDataEncoder = std::make_shared<PNGEncoder>();
    frameDataEncoder = std::make_shared<H264Encoder>();

    webSocket.setUrl(SERVER_URL + jobId);
    webSocket.setOnMessageCallback(std::bind(&RemoteRenderer::onMessage, this, std::placeholders::_1));
    webSocket.start();

    streamingWebSocket.setUrl(STREAMING_SERVER_URL + jobId);
    streamingWebSocket.setOnMessageCallback([](const ix::WebSocketMessagePtr &msg) {
        std::cout << "Streaming WebSocket message: " << msg->str << std::endl;
    });
    streamingWebSocket.start();
}

RemoteRenderer::~RemoteRenderer()
{
    webSocket.stop();
}

void RemoteRenderer::addMessageListener(Event::EventType eventType, LambdaFunction listener)
{
    eventListeners.try_emplace(eventType, listener);
}

void RemoteRenderer::removeMessageListener(Event::EventType eventType)
{
    eventListeners.erase(eventType);
}

void RemoteRenderer::onMessage(const ix::WebSocketMessagePtr &msg)
{
    if (msg->type == ix::WebSocketMessageType::Message)
    {
        Event event;
        event.ParseFromString(msg->str);
        auto listener = eventListeners.find(event.type());
        if (listener != eventListeners.end())
        {
            listener->second(event);
        }
    }
    else if (msg->type == ix::WebSocketMessageType::Open)
    {
        webSocket.send("NOTIFICATION#SUCCESS#JOB_INIT#Job has started!");
        std::cout << "Connection established" << std::endl;
    }
    else if (msg->type == ix::WebSocketMessageType::Error)
    {
        // Maybe SSL is not configured properly
        std::cout << "Connection error: " << msg->errorInfo.reason << std::endl;
        stopRenderer = true;
    }
}

std::vector<uint8_t> RemoteRenderer::prepareFrame(const uint8_t *frame, const std::shared_ptr<PixelDataEncoder> &pixelDataEncoder)
{
    int view_width = config.resolution.width;
    int view_height = config.resolution.height;
    std::vector<uint8_t> outputData;
    auto start = std::chrono::high_resolution_clock::now();
    if (!pixelDataEncoder->encodePixelData(frame, view_width, view_height, outputData))
    {
        outputData.clear();
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Encoding time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    return outputData;
}

std::vector<uint8_t> RemoteRenderer::processFrameForStreaming(const uint8_t *frame)
{
    return prepareFrame(frame, frameDataEncoder);
}

std::vector<uint8_t> RemoteRenderer::processFrameForSnapshot(const uint8_t *frame)
{
    return prepareFrame(frame, snapshotDataEncoder);
}

void RemoteRenderer::renderFrame()
{
    uint8_t *frame = framebuffer->getYUVPtr();
    std::vector<uint8_t> outputData = processFrameForStreaming(frame);
    if (!outputData.empty())
    {
        ix::IXWebSocketSendData IXPixelData(outputData);
        streamingWebSocket.sendBinary(IXPixelData);
    }
    generateAndSendSnapshotIfNeeded();
}

void RemoteRenderer::generateAndSendSnapshotIfNeeded()
{
    if (!generateAndSendSnapshotFlag)
    {
        return;
    }
    uint8_t *frame = framebuffer->getRGBPtr();
    std::vector<uint8_t> outputData = processFrameForSnapshot(frame);
    if (!outputData.empty())
    {
        std::string messagePrefix = "SNAPSHOT#";
        std::vector<uint8_t> messagePrefixVec(messagePrefix.begin(), messagePrefix.end());
        outputData.insert(outputData.begin(), messagePrefixVec.begin(), messagePrefixVec.end());
        ix::IXWebSocketSendData IXPixelData(outputData);
        webSocket.sendBinary(IXPixelData);
    }
    generateAndSendSnapshotFlag = false;
}

void RemoteRenderer::generateAndSendSnapshot()
{
    // When next frame is rendered, snapshot will be sent
    generateAndSendSnapshotFlag = true;
}

bool RemoteRenderer::shouldStopRendering()
{
    return stopRenderer;
}

void RemoteRenderer::send(const std::string &data)
{
    webSocket.send(data);
}
