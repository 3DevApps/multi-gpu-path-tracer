#pragma once

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "helper_math.h"
#include <unordered_map>
#include <vector>
#include "triangle.h"
#include "object3d.h"
#include "material.h"

class obj_loader
{
public:
    obj_loader() {};
    int get_number_of_meshes(const char* path);
    void get_number_of_faces(const char* path, int *num_faces);
    void load(const char* path, object3d **objects);
};

int obj_loader::get_number_of_meshes(const char* path)
{
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FindDegenerates);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        throw std::runtime_error(importer.GetErrorString());
    }

    return scene->mNumMeshes;
}

// Get number of meshes
void obj_loader::get_number_of_faces(const char* path, int *num_faces){
    Assimp::Importer importer;

    const aiScene *scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FindDegenerates);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        throw std::runtime_error(importer.GetErrorString());
    }

    if (scene->HasMeshes()) {
        for (unsigned int i = 0; i < scene->mNumMeshes; i++)
        {
            const aiMesh* mesh = scene->mMeshes[i];
            num_faces[i] = mesh->mNumFaces;
        }
    }
}

void obj_loader::load(const char* path, object3d **objects)
{
    Assimp::Importer importer;

    const aiScene *scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FindDegenerates);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        throw std::runtime_error(importer.GetErrorString());
    }

    if (scene->HasMeshes()) {
        for (unsigned int i = 0; i < scene->mNumMeshes; i++)
        {
            const aiMesh* mesh = scene->mMeshes[i];
            for (unsigned int j = 0; j < mesh->mNumFaces; j++)
            {
                const aiFace& face = mesh->mFaces[j];

                if (face.mNumIndices != 3)
                {
                    throw std::runtime_error("Face is not a triangle");
                }

                aiVector3D v1 = mesh->mVertices[face.mIndices[0]];
                aiVector3D v2 = mesh->mVertices[face.mIndices[1]];
                aiVector3D v3 = mesh->mVertices[face.mIndices[2]];

                objects[i]->triangles[j] = triangle(make_float3(v1.x, v1.y, v1.z), make_float3(v2.x, v2.y, v2.z), make_float3(v3.x, v3.y, v3.z));
            }

            objects[i]->num_triangles = mesh->mNumFaces;            
        }
    }
}
