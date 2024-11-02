#include "RemoteRenderer.h"

RemoteRenderer::RemoteRenderer(std::string &jobId, RendererConfig &config)
    : jobId(jobId), config(config)
{
    pixelData.resize(config.resolution.width * config.resolution.height * 3);

    snapshotDataEncoder = std::make_shared<PNGEncoder>();
    frameDataEncoder = std::make_shared<H264Encoder>();

    webSocket.setUrl(SERVER_URL + jobId);
    webSocket.setOnMessageCallback(std::bind(&RemoteRenderer::onMessage, this, std::placeholders::_1));
    webSocket.start();
}

RemoteRenderer::~RemoteRenderer()
{
    webSocket.stop();
}

void RemoteRenderer::addMessageListener(std::string eventName, LambdaFunction listener)
{
    eventListeners.try_emplace(eventName, listener);
}

void RemoteRenderer::removeMessageListener(std::string eventName)
{
    eventListeners.erase(eventName);
}

void RemoteRenderer::onMessage(const ix::WebSocketMessagePtr &msg)
{
    if (msg->type == ix::WebSocketMessageType::Message)
    {
        auto sepPos = msg->str.find("#");
        std::string messageKey = msg->str.substr(0, sepPos);
        auto listener = eventListeners.find(messageKey);
        if (listener != eventListeners.end())
        {
            listener->second(msg->str.substr(sepPos + 1));
        }
    }
    else if (msg->type == ix::WebSocketMessageType::Open)
    {
        webSocket.send("JOB_MESSAGE#NOTIFICATION#SUCCESS#JOB_INIT#Job has started!");
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

    if (pixelData.size() != view_width * view_height * 3)
    {
        pixelData.resize(view_width * view_height * 3);
    }

    for (int y = view_height - 1; y >= 0; --y)
    {
        for (int x = 0; x < view_width; ++x)
        {
            int fbi = (y * view_width + x) * 3;
            int pdi = ((view_height - y - 1) * view_width + x) * 3;
            pixelData[pdi] = frame[fbi];
            pixelData[pdi + 1] = frame[fbi + 1];
            pixelData[pdi + 2] = frame[fbi + 2];
        }
    }

    std::vector<uint8_t> outputData;
    auto start = std::chrono::high_resolution_clock::now();
    if (!pixelDataEncoder->encodePixelData(pixelData, view_width, view_height, outputData))
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

void RemoteRenderer::renderFrame(const uint8_t *frame)
{
    std::vector<uint8_t> outputData = processFrameForStreaming(frame);
    if (!outputData.empty())
    {
        std::string messagePrefix = "JOB_MESSAGE#RENDER#";
        std::vector<uint8_t> messagePrefixVec(messagePrefix.begin(), messagePrefix.end());
        outputData.insert(outputData.begin(), messagePrefixVec.begin(), messagePrefixVec.end());
        ix::IXWebSocketSendData IXPixelData(outputData);
        webSocket.sendBinary(IXPixelData);
    }
    generateAndSendSnapshotIfNeeded(frame);
}

void RemoteRenderer::generateAndSendSnapshotIfNeeded(const uint8_t *frame)
{
    if (!generateAndSendSnapshotFlag)
    {
        return;
    }
    std::vector<uint8_t> outputData = processFrameForSnapshot(frame);
    if (!outputData.empty())
    {
        std::string messagePrefix = "JOB_MESSAGE#SNAPSHOT#";
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