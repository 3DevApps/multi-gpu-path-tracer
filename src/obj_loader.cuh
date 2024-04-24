#pragma once

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "helper_math.h"
#include <unordered_map>
#include <vector>
#include "triangle.h"
#include "material.h"
#include "cuda_utils.cuh"

class obj_loader
{
public:
    obj_loader(const char* path) : file_path(path) {};
    int get_total_number_of_faces();
    void get_faces(triangle *triangles);

    const char* file_path;
};

int obj_loader::get_total_number_of_faces()
{
    Assimp::Importer importer;
    const aiScene *scene = importer.ReadFile(file_path, aiProcess_Triangulate | aiProcess_FindDegenerates);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        throw std::runtime_error(importer.GetErrorString());
    }

    int total_faces = 0;

    if (scene->HasMeshes()) {
        for (unsigned int i = 0; i < scene->mNumMeshes; i++)
        {
            const aiMesh* mesh = scene->mMeshes[i];
            total_faces += mesh->mNumFaces;
        }
    }

    return total_faces;
}

void obj_loader::get_faces(triangle *triangles){
    Assimp::Importer importer;

    const aiScene *scene = importer.ReadFile(file_path, aiProcess_Triangulate | aiProcess_FindDegenerates);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        throw std::runtime_error(importer.GetErrorString());
    }

    int index = 0;

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

               checkCudaErrors(cudaMallocManaged((void **)&triangles[index], sizeof(material)));

                aiVector3D v1 = mesh->mVertices[face.mIndices[0]];
                aiVector3D v2 = mesh->mVertices[face.mIndices[1]];
                aiVector3D v3 = mesh->mVertices[face.mIndices[2]];

                triangles[index].v0 = make_float3(v1.x, v1.y, v1.z);
                triangles[index].v1 = make_float3(v2.x, v2.y, v2.z);
                triangles[index].v2 = make_float3(v3.x, v3.y, v3.z);
                index++;
            }
        }
    }

}
