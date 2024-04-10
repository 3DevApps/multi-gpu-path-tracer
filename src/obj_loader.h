#pragma once

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "helper_math.h"
#include <unordered_map>
#include <vector>
#include "triangle.h"
#include "object3d.h"

class obj_loader
{
public:
    obj_loader() {};
    void load(object3d *obj, const char* path);
};

void obj_loader::load(object3d *obj, const char* path)
{
    printf("Loading obj file: %s\n", path);
    Assimp::Importer importer;
    std::vector<triangle> triangles;

    const aiScene *scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FindDegenerates);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        printf("Error: %s\n", importer.GetErrorString());
        std::runtime_error(importer.GetErrorString());
        return;
    }


    printf("Number of meshes: %d\n", scene->mNumMeshes);

    if (scene->HasMeshes()) {
        for (unsigned int i = 0; i < scene->mNumMeshes; i++)
        {
            // Create array of triangles for each mesh
            const aiMesh* mesh = scene->mMeshes[i];
            for (unsigned int j = 0; j < mesh->mNumFaces; j++)
            {
                const aiFace& face = mesh->mFaces[j];

                if (face.mNumIndices != 3)
                {
                    std::runtime_error("Face is not a triangle");
                    return;
                }

                aiVector3D v1 = mesh->mVertices[face.mIndices[0]];
                aiVector3D v2 = mesh->mVertices[face.mIndices[1]];
                aiVector3D v3 = mesh->mVertices[face.mIndices[2]];

                triangles.push_back(triangle(make_float3(v1.x, v1.y, v1.z), make_float3(v2.x, v2.y, v2.z), make_float3(v3.x, v3.y, v3.z)));
            }
        }
    }

    obj->set_triangles(triangles.data(), triangles.size());
}
