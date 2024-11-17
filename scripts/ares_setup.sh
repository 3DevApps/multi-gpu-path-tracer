module load libpng/1.6.39 # TODO: move to conan
module load openssl/1.1 # TODO: move to conan (with IXWebSocket)
module load python/3.11.3-gcccore-12.3.0
module load cuda/12.2.0
module load cmake/3.26.3

pip install conan
conan profile detect --force
conan install . --build=missing --output-folder=build
