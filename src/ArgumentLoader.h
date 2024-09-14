#pragma once

#include <string>

class ArgumentLoader {
    public:
        struct Arguments {
            std::string filePath;
            std::string jobId;
        };
        ArgumentLoader(int argc, char** argv): argc(argc), argv(argv) {};
        void loadArguments() {
            arguments.filePath = argc > 2 ? argv[2] : "models/cornell-box.obj";
            arguments.jobId = argc > 1 ? argv[1] : "0";
        };  
        Arguments loadAndGetArguments() {
            loadArguments();
            return arguments;
        };          
    private:
        int argc;
        char** argv;
        Arguments arguments;
};