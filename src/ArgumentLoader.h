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
            arguments.jobId = (argc > 1 && argv[1][0] != '\0') ? argv[1] : "0";
            arguments.filePath = (argc > 2 && argv[2][0] != '\0') ? argv[2] : "models/f" + arguments.jobId + ".obj";
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