# Multi-GPU Path Tracer

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
