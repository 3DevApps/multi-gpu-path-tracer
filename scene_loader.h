#pragma once
#include "scene3d.h"

// Class which loads a scene from a obj file
class scene_loader {
public:
    // Function to load a scene from a obj file
    static scene3d *load_scene(const char *filename);
};

// Function to load a scene from a obj file
// TODO: not working yet, just some boiler code from Copilot
scene3d *scene_loader::load_scene(const char *filename) {
    // Open the file
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Could not open file %s\n", filename);
        return NULL;
    }

    // Count the number of triangles in the file
    int num_triangles = 0;
    char line[1024];
    while (fgets(line, 1024, file)) {
        if (line[0] == 'f' && line[1] == ' ') {
            num_triangles++;
        }
    }

    // Allocate memory for the triangles
    hitable **triangles = (hitable **)malloc(num_triangles * sizeof(hitable *));

    // Go back to the beginning of the file
    fseek(file, 0, SEEK_SET);

    // Read the triangles from the file
    int triangle_index = 0;
    while (fgets(line, 1024, file)) {
        if (line[0] == 'f' && line[1] == ' ') {
            int v0, v1, v2;
            sscanf(line, "f %d %d %d", &v0, &v1, &v2);
            triangles[triangle_index] = new triangle(vertices[v0 - 1], vertices[v1 - 1], vertices[v2 - 1], material());
            triangle_index++;
        }
    }

    // Close the file
    fclose(file);

    // Create the scene
    scene3d *scene = new scene3d(triangles, num_triangles);

    // Return the scene
    return scene;
}
