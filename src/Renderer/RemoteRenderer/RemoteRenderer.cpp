#include "RemoteRenderer.h"

RemoteRenderer::RemoteRenderer(std::string& jobId, std::uint32_t view_width, std::uint32_t view_height)
    : jobId(jobId), view_width(view_width), view_height(view_height) {
    pixelData.resize(view_width * view_height * 3);
    
    // pixelDataEncoder = std::make_shared<PNGEncoder>();
    pixelDataEncoder = std::make_shared<JPEGEncoder>(75);

    webSocket.setUrl(SERVER_URL + jobId);
    webSocket.setOnMessageCallback(std::bind(&RemoteRenderer::onMessage, this, std::placeholders::_1));
    webSocket.start();
}

RemoteRenderer::~RemoteRenderer() {
    webSocket.stop();
}

void RemoteRenderer::addMessageListener(std::string eventName, LambdaFunction listener) {
    eventListeners.try_emplace(eventName, listener);
}

void RemoteRenderer::removeMessageListener(std::string eventName) {
    eventListeners.erase(eventName);
}

void RemoteRenderer::onMessage(const ix::WebSocketMessagePtr& msg) {
    if (msg->type == ix::WebSocketMessageType::Message) {
        auto sepPos = msg->str.find("#");
        std::string messageKey = msg->str.substr(0, sepPos);
        auto listener = eventListeners.find(messageKey);
        if (listener != eventListeners.end()) {
            listener->second(msg->str.substr(sepPos + 1));
        }
    } 
    else if (msg->type == ix::WebSocketMessageType::Open)
    {
        std::cout << "Connection established" << std::endl;
    }
    else if (msg->type == ix::WebSocketMessageType::Error)
    {
        // Maybe SSL is not configured properly
        std::cout << "Connection error: " << msg->errorInfo.reason << std::endl;
    }
}

void RemoteRenderer::renderFrame(const uint8_t *frame) {
    for (int y = view_height - 1; y >= 0; --y) {
        for (int x = 0; x < view_width; ++x) {
            int fbi = (y * view_width + x) * 3;
            int pdi = ((view_height - y - 1) * view_width + x) * 3;
            pixelData[pdi] = frame[fbi];
            pixelData[pdi + 1] = frame[fbi + 1];
            pixelData[pdi + 2] = frame[fbi + 2];
        }
    }

    std::vector<uint8_t> outputData;
    if (pixelDataEncoder->encodePixelData(pixelData, view_width, view_height, outputData)) {
        std::string messagePrefix = "JOB_MESSAGE#RENDER#";
        std::vector<uint8_t> messagePrefixVec(messagePrefix.begin(), messagePrefix.end());
        outputData.insert(outputData.begin(), messagePrefixVec.begin(), messagePrefixVec.end());
        ix::IXWebSocketSendData IXPixelData(outputData);
        webSocket.sendBinary(IXPixelData);
    } 
}