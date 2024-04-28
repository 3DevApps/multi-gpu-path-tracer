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

// Materials supported by the obj loader
enum material_type {
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    DIFFUSE_LIGHT
};

struct m_ai_material {
    material_type type ;
    float3 color_ambient;
    float3 color_diffuse;
    float index_of_refraction;
    float shininess;
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

__global__ void assign_triangle(hitable **d_list, int index, material** materials, int material_index, m_ai_material *d_mat, aiVector3D v0, aiVector3D v1, aiVector3D v2) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (materials[material_index] == nullptr) {
            if (d_mat->type == LAMBERTIAN) {
                materials[material_index] = new lambertian(d_mat->color_ambient);
            }
            else if (d_mat->type == METAL) {
                materials[material_index] = new metal(d_mat->color_ambient, d_mat->shininess);
            }
            else if (d_mat->type == DIELECTRIC) {
                materials[material_index] = new dielectric(d_mat->index_of_refraction);
            }
            else if (d_mat->type == DIFFUSE_LIGHT) {
                materials[material_index] = new diffuse_light(d_mat->color_diffuse);
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
    int num_materials = scene->mNumMaterials;

    material **materials;
    checkCudaErrors(cudaMalloc((void **)&materials, num_materials * sizeof(material*)));
    int is_material_allocated[num_materials] = {0};
    
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

                // This struct is needed to pass material properties to the kernel
                m_ai_material *d_mat; 

                if (is_material_allocated[mesh->mMaterialIndex] == 0) {
                    aiMaterial *material = scene->mMaterials[mesh->mMaterialIndex];      

                    m_ai_material m;
                    aiString ai_string_name;
                    material->Get(AI_MATKEY_NAME, ai_string_name);

                    std::string name = ai_string_name.C_Str();
                    
                    if (name.find("lambertian") != std::string::npos) {
                        aiColor3D color;
                        material->Get(AI_MATKEY_COLOR_AMBIENT, color);

                        m.type = LAMBERTIAN;
                        m.color_ambient = make_float3(color.r, color.g, color.b);
                    } 
                    else if (name.find("metal") != std::string::npos) {
                        aiColor3D color;
                        material->Get(AI_MATKEY_COLOR_AMBIENT, color);

                        float shininess;
                        material->Get(AI_MATKEY_SHININESS, shininess); // Represented as Ns in the mtl file

                        m.type = METAL;
                        m.color_ambient = make_float3(color.r, color.g, color.b); 
                        m.shininess = shininess; 
                    } 
                    else if (name.find("dielectric") != std::string::npos) {
                        float index_of_refraction;
                        material->Get(AI_MATKEY_REFRACTI, index_of_refraction); // Represented as Ni in the mtl file
                    
                        m.type = DIELECTRIC;
                        m.index_of_refraction = index_of_refraction;
                    }
                    else if (name.find("diffuse_light") != std::string::npos) {
                        aiColor3D color;
                        material->Get(AI_MATKEY_COLOR_DIFFUSE, color); // Represented as Kd in the mtl file

                        m.type = DIFFUSE_LIGHT;
                        m.color_diffuse = make_float3(color.r, color.g, color.b);
                    }
                    
                    checkCudaErrors(cudaMalloc((void **)&d_mat, sizeof(m_ai_material)));
                    checkCudaErrors(cudaMemcpy(d_mat, &m, sizeof(m_ai_material), cudaMemcpyHostToDevice));
                }

                assign_triangle<<<1, 1>>>(d_list, index, materials, mesh->mMaterialIndex, d_mat, v0, v1, v2);

                if (is_material_allocated[mesh->mMaterialIndex] == 0) {
                    is_material_allocated[mesh->mMaterialIndex] = 1;
                    checkCudaErrors(cudaFree(d_mat));
                }

                index++;
            }
        }
    }

    checkCudaErrors(cudaFree(materials));

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}
