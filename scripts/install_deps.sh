#!/bin/bash

GLFW_URL="https://github.com/glfw/glfw/archive/refs/tags/3.4.tar.gz"
GLEW_URL="https://github.com/nigels-com/glew/releases/download/glew-2.2.0/glew-2.2.0.tgz"
ASSIMP_URL="https://github.com/assimp/assimp/archive/refs/tags/v5.4.0.tar.gz"
IXWEBSOCKET_URL="https://github.com/machinezone/IXWebSocket.git"
LIBHEIF_URL="https://github.com/strukturag/libheif.git"


LIB_PREFIX=~/libs
ARCHIVE="sources"
TMP_DIR=~/.tmp_deps

clear () {
	rm -rf $TMP_DIR
	mkdir $TMP_DIR
	cd $TMP_DIR
}

cmake_install () {
	cmake .
	cmake --build .
	cmake --install . --prefix "$LIB_PREFIX"
}

cmake_install_ixwebsocket () {
	clear
	git clone --branch v11.4.5 --depth 1 $IXWEBSOCKET_URL
	cd *
	cmake -DUSE_TLS=1 .
	cmake --build .
	cmake --install . --prefix "$LIB_PREFIX"
}

cmake_install_libheif () {
	clear
	git clone --branch v1.18.2 --depth 1 $LIBHEIF_URL
	cd *
	cmake --preset=release
	cmake -DCMAKE_INSTALL_PREFIX:PATH=~/libs .
	cmake -DCMAKE_INSTALL_FULL_LIBDIR:PATH=~/libs .
	cmake --build .
	cmake --install .
}

download_lib () {
	wget $1 -O "$ARCHIVE"
	tar xf "$ARCHIVE"
	rm -rf "$ARCHIVE"	
}

urls=("$GLFW_URL" "$GLEW_URL" "$ASSIMP_URL")

for url in "${urls[@]}"; do
	clear
	download_lib "$url"

	# For GLEW we need to call cmake_install from a different directory
	if [[ $url == "$GLEW_URL" ]]; then
		cd */build/cmake
	else
		cd *
	fi
	
	cmake_install

	cd -	
done

cmake_install_ixwebsocket
clear

cmake_install_libheif
clear