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

enum material_type {
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    UNKNOWN
};

struct mAiMaterial {
    material_type type = UNKNOWN;
    float3 color_ambient;
    float index_of_refraction;
};

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

__global__ void assign_triangle(hitable **d_list,mAiMaterial *d_mat, material** materials, int material_index, int index, aiVector3D v0, aiVector3D v1, aiVector3D v2) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (materials[material_index] == nullptr) {
            if (d_mat->type == LAMBERTIAN) {
                materials[material_index] = new lambertian(d_mat->color_ambient);
            }
            else if (d_mat->type == METAL) {
                materials[material_index] = new metal(d_mat->color_ambient, d_mat->index_of_refraction);
            }
            else if (d_mat->type == DIELECTRIC) {
                materials[material_index] = new dielectric(d_mat->index_of_refraction);
            }
        }

        material *mat = materials[material_index];
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

    mAiMaterial d_materials[num_materials];

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
                mAiMaterial *d_mat;

                if (d_materials[mesh->mMaterialIndex].type == UNKNOWN) {
                    mAiMaterial m;
                    aiString name;
                    material->Get(AI_MATKEY_NAME, name);

                    if (name == aiString("lambertian")) {
                        aiColor3D color;
                        material->Get(AI_MATKEY_COLOR_AMBIENT, color);

                        m.type = LAMBERTIAN;
                        m.color_ambient = make_float3(color.r, color.g, color.b);
                    } 
                    else if (name == aiString("metal")) {
                        aiColor3D color;
                        material->Get(AI_MATKEY_COLOR_AMBIENT, color);

                        float fuzz_value;
                        material->Get(AI_MATKEY_REFRACTI, fuzz_value);

                        m.type = METAL;
                        m.color_ambient = make_float3(color.r, color.g, color.b); 
                        m.index_of_refraction = fuzz_value; // Fuzz value not supported by Assimp
                    } 
                    else if (name == aiString("dielectric")) {
                        float index_of_refraction;
                        material->Get(AI_MATKEY_REFRACTI, index_of_refraction);
                    
                        m.type = DIELECTRIC;
                        m.index_of_refraction = index_of_refraction;
                    }
                    
                    checkCudaErrors(cudaMalloc((void **)&d_mat, sizeof(mAiMaterial)));
                    checkCudaErrors(cudaMemcpy(d_mat, &m, sizeof(mAiMaterial), cudaMemcpyHostToDevice));
                }


                assign_triangle<<<1, 1>>>(d_list, d_mat, materials, mesh->mMaterialIndex, index, v0, v1, v2);
                index++;
            }
        }
    }

    checkCudaErrors(cudaFree(materials));

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
