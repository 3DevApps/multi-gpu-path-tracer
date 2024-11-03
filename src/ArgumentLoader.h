#pragma once

#include <string>
#include "RendererConfig.h"

class ArgumentLoader {
    public:
        ArgumentLoader(int argc, char** argv): argc(argc), argv(argv) {};

        void loadArguments(RendererConfig& config) {
            config.jobId = (argc > 1 && argv[1][0] != '\0') ? argv[1] : "0";
            config.modelPath = (argc > 2 && argv[2][0] != '\0') ? argv[2] : "models/cornel/cornell_box.gltf";
        };  
    private:
        int argc;
        char** argv;
};