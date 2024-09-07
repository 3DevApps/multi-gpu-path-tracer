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
#include <memory.h>
#include "HostScene.h"

class obj_loader
{
public:
    obj_loader(const char* path) : file_path(path) {};
    std::vector<Triangle> load_triangles();
    const char* file_path;
};

float3 convertToFloat3(aiVector3D &v) {
    return make_float3(v.x, v.y, v.z);
}


std::vector<Triangle> obj_loader::load_triangles() {
    Assimp::Importer importer;

    const aiScene *scene = importer.ReadFile(file_path, aiProcess_Triangulate | aiProcess_FindDegenerates);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        throw std::runtime_error(importer.GetErrorString());
    }

    int num_materials = scene->mNumMaterials;
    std::vector<Triangle> triangles; 
    
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

                auto v0 = convertToFloat3(mesh->mVertices[face.mIndices[0]]);
                auto v1 = convertToFloat3(mesh->mVertices[face.mIndices[1]]);
                auto v2 = convertToFloat3(mesh->mVertices[face.mIndices[2]]);

                aiMaterial *material = scene->mMaterials[mesh->mMaterialIndex];      

                m_ai_material m;
                aiString ai_string_name;
                material->Get(AI_MATKEY_NAME, ai_string_name);

                std::string name = ai_string_name.C_Str();
                
                if (name.rfind("lambertian", 0) != std::string::npos) {
                    aiColor3D color;
                    material->Get(AI_MATKEY_COLOR_AMBIENT, color);

                    m.type = LAMBERTIAN;
                    m.color_ambient = make_float3(color.r, color.g, color.b);
                } 
                else if (name.rfind("metal", 0) != std::string::npos) {
                    aiColor3D color;
                    material->Get(AI_MATKEY_COLOR_AMBIENT, color);

                    float shininess;
                    material->Get(AI_MATKEY_SHININESS, shininess); // Represented as Ns in the mtl file

                    m.type = METAL;
                    m.color_ambient = make_float3(color.r, color.g, color.b); 
                    m.shininess = shininess; 
                } 
                else if (name.rfind("dielectric", 0) != std::string::npos) {
                    float index_of_refraction;
                    material->Get(AI_MATKEY_REFRACTI, index_of_refraction); // Represented as Ni in the mtl file
                
                    m.type = DIELECTRIC;
                    m.index_of_refraction = index_of_refraction;
                }
                else if (name.rfind("diffuse_light", 0) != std::string::npos) {
                    aiColor3D color;
                    material->Get(AI_MATKEY_COLOR_DIFFUSE, color); // Represented as Kd in the mtl file

                    m.type = DIFFUSE_LIGHT;
                    m.color_diffuse = make_float3(color.r, color.g, color.b);
                }
                triangles.push_back(Triangle{v0, v1, v2, std::move(m)});
            }
        }
    }
    return triangles;
}