wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6 git cmake nvidia-cuda-toolkit libxkbcommon-dev libwayland-dev xorg-dev libglu1-mesa-dev pkg-config libx11-dev libxrandr-dev libssl-dev libpng-dev

pip install conan
conan profile detect --force
conan install . --build=missing --output-folder=build
