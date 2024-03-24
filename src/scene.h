#pragma once

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"

// #include "hitable.h"



// class scene: public hitable
class scene
{
public:
    scene(const char* path);
};

scene::scene(const char* path)
{
    Assimp::Importer importer;

    const aiScene *scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenNormals);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        printf("ERROR::ASSIMP::%s\n", importer.GetErrorString());
        return;
    }


    // Print meshes
    for (unsigned int i = 0; i < scene->mNumMeshes; i++)
    {
        aiMesh *mesh = scene->mMeshes[i];
        printf("Mesh %d\n", i);
        printf("  Vertices: %d\n", mesh->mNumVertices);
        printf("  Faces: %d\n", mesh->mNumFaces);
        printf("  Material: %d\n", mesh->mMaterialIndex);
    }

    return;
}
