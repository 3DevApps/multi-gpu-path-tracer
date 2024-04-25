#pragma once

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "helper_math.h"
#include <unordered_map>
#include <vector>
#include "triangle.h"
#include "material.h"
#include "cuda_utils.h"
#include "hitable.h"

class obj_loader
{
public:
    obj_loader(const char* path) : file_path(path) {};
    int get_total_number_of_faces();
    void load_faces(hitable **d_list);

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

__global__ void assign_triangle(hitable **d_list, material** materials, aiMaterial *mesh_material, int material_index, int index, aiVector3D v0, aiVector3D v1, aiVector3D v2) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (materials[0] == nullptr) {
            // Get mesh_material name
            

            // Get material name
            aiString name;
            mesh_material->Get(AI_MATKEY_NAME, name);

            printf("Material name: %s\n", name.C_Str());


            // // get COLOR_AMBIENT
            // aiColor3D color;
            // mesh_material->Get(AI_MATKEY_COLOR_AMBIENT, color);
            // printf("Material ambient color: %f %f %f\n", color.r, color.g, color.b);


            // printf("Material name: %s\n", name.C_Str());




            materials[0] = new lambertian(make_float3(0.8, 0.3, 0.3));
        }


        material *mat = materials[0];

        // material *mat = new metal(make_float3(0.8, 0.6, 0.2), 0.0);
        // material *mat = new dielectric(1.5);                       
        d_list[index] = new triangle(make_float3(v0.x, v0.y, v0.z), make_float3(v1.x, v1.y, v1.z), make_float3(v2.x, v2.y, v2.z), mat);
    }
}

void obj_loader::load_faces(hitable **d_list) {
    Assimp::Importer importer;

    const aiScene *scene = importer.ReadFile(file_path, aiProcess_Triangulate | aiProcess_FindDegenerates);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        throw std::runtime_error(importer.GetErrorString());
    }

    int index = 0;

    // Array for storing materials
    int num_materials = scene->mNumMaterials;

    material **materials;
    checkCudaErrors(cudaMalloc((void **)&materials, num_materials * sizeof(material*)));

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

                aiVector3D v0 = mesh->mVertices[face.mIndices[0]];
                aiVector3D v1 = mesh->mVertices[face.mIndices[1]];
                aiVector3D v2 = mesh->mVertices[face.mIndices[2]];

                aiMaterial *material = scene->mMaterials[mesh->mMaterialIndex];       


                // return;         

                assign_triangle<<<1, 1>>>(d_list, materials, material, mesh->mMaterialIndex, index, v0, v1, v2);
                index++;
            }
        }
    }

    checkCudaErrors(cudaFree(materials));

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
