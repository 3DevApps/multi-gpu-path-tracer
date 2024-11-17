#pragma once

#include <cmath>
#include "EventHandler.h"
#include "../../../RenderManager.h"
#include "../../../PixelDataEncoder/PNGEncoder.h"

class RenderManagerEventHander: EventHandler {
    public:
        RenderManagerEventHander(RemoteRenderer& remoteRenderer, RenderManager& manager, HostScene &hScene) : remoteRenderer(remoteRenderer), manager(manager), hScene(hScene) {};

        void handleEvent(const Event& event) override {
            auto rendererEvent = event.renderer();
            switch (rendererEvent.type()) {
                case RendererEvent::GPU_NUMBER:
                    manager.setGpuNumber(rendererEvent.numbervalue());
                    break;
                case RendererEvent::STREAMS_PER_GPU:
                    manager.setStreamsPerGpu(rendererEvent.numbervalue());
                    break;
                case RendererEvent::SAMPLES_PER_PIXEL:
                    manager.setSamplesPerPixel(rendererEvent.numbervalue());
                    break;
                case RendererEvent::RECURSION_DEPTH:
                    manager.setRecursionDepth(rendererEvent.numbervalue());
                    break;
                case RendererEvent::LOAD_UPLOADED_SCENE:
                    manager.reloadScene();
                    break;
                case RendererEvent::DOWNLOAD_SCENE_SNAPSHOT:
                    remoteRenderer.generateAndSendSnapshot();
                    break;
                case RendererEvent::THREAD_BLOCK_SIZE: {
                    auto threadBlockSizeX = rendererEvent.blockvalue().x();
                    auto threadBlockSizeY = rendererEvent.blockvalue().y();
                    dim3 threadBlockSize(threadBlockSizeX, threadBlockSizeY);
                    manager.setThreadBlockSize(threadBlockSize);
                    break;
                }
                case RendererEvent::IMAGE_RESOLUTION: {
                    unsigned int width = rendererEvent.blockvalue().x();
                    unsigned int height = rendererEvent.blockvalue().y();
                    manager.setResolution({width, height});
                    break;
                }
                case RendererEvent::SHOW_TASK_GRID:
                    manager.setShowTasks(rendererEvent.booleanvalue());
                    break;
                case RendererEvent::LOAD_BALANCING_ALGORITHM: {
                    auto loadBalancingAlgorithm = parseSchedulingAlgorithmType(rendererEvent.loadbalancingalgorithm());
                    manager.setSchedulingAlgorithm(loadBalancingAlgorithm);
                    break;
                }
                default:
                    break;
            }
        };

        SchedulingAlgorithmType parseSchedulingAlgorithmType(const RendererEvent::LoadBalancingAlgorithm& algorithm) {
            if (algorithm == RendererEvent::FST) {
                 return SchedulingAlgorithmType::FST;
             } else if (algorithm == RendererEvent::DTFL) {
                 return SchedulingAlgorithmType::DTFL;
             } else if (algorithm == RendererEvent::DT) {
                 return SchedulingAlgorithmType::DT;
             } else {
                 return SchedulingAlgorithmType::FST;
             }
        };

        Event::EventType getEventType() override {
            return Event::RENDERER_EVENT;
        };

    private:
        RemoteRenderer& remoteRenderer;
        RenderManager& manager;
        HostScene& hScene;
        PNGEncoder pngEncoder;
};
