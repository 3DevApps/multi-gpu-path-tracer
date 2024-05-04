# Multi-GPU Path Tracer

## Setup

Run before building dependencies from source:

### Ares

```bash
source scripts/ares_setup.sh
```

### Ubuntu

```bash
source scripts/ubuntu_setup.sh
```

## Building dependencies

scripts/install_deps.sh downloads specified releases, builds them and installs them in ~/libs.

```bash
./scripts/install_deps.sh
```

## Compiling and Running

To compile and run the project, follow the steps below:

1. Create a build directory

```bash
mkdir build
```

2. Build the project

```bash
cmake --build build
```

> [!NOTE]
> If you get `Cmake Error: could not load cache` error, run `cmake -S . -B build` and re-run the build command.

3. Run the project

```bash
build/cuda_project
```

## Visualizing the output

The output file is saved in the PPM format. To visualize it, you can utilize the GNOME Image Viewer.

```bash
eog build/out.ppm
```

## Using custom .obj and .mtl files

To use custom .obj and .mtl files, specify the paths to these files in the main.cpp file.

Currently, the program only supports the following materials with the corresponding properties:

- lambertian
  - Ka
- metal
  - Ka
  - Ns
- dielectric
  - Ni
- diffuse_light
  - Kd

To use these materials, ensure that the material's name in the .mtl file matches with one from the list above.
