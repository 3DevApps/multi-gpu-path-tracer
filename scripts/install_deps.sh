#!/bin/bash

GLFW_URL="https://github.com/glfw/glfw/archive/refs/tags/3.4.tar.gz"
GLEW_URL="https://github.com/nigels-com/glew/releases/download/glew-2.2.0/glew-2.2.0.tgz"
ASSIMP_URL="https://github.com/assimp/assimp/archive/refs/tags/v5.4.0.tar.gz"


LIB_PREFIX=~/libs
ARCHIVE="sources"

clear () {
	rm -rf ~/.tmp_deps
	mkdir ~/.tmp_deps
	cd ~/.tmp_deps
}

cmake_install () {
	cmake .
	cmake --build .
	cmake --install . --prefix "$LIB_PREFIX"
}

download_lib () {
	wget $1 -O "$ARCHIVE"
	tar xf "$ARCHIVE"
	rm -rf "$ARCHIVE"	
}

clear
download_lib "$GLFW_URL"
cd *
cmake_install

clear
download_lib "$GLEW_URL"
cd */build/cmake
cmake_install

clear
download_lib "$ASSIMP_URL"
cd *
cmake_install

clear





















