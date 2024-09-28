
#pragma once

#include <curand_kernel.h>
#include "helper_math.h"
#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include "RendererConfig.h"

struct CameraParams {
    float3 front;
    float3 lookFrom;
    float vfov = 45.0f;
    float hfov = 45.0f;
};

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

struct Triangle {
    float3 v0;
    float3 v1;
    float3 v2;
    m_ai_material material_params;
};

class CameraObserver {
public:
    virtual void updateCamera() = 0;
};

class PrimitivesObserver {
public:
    virtual void updatePrimitives() = 0;
};

class HostScene {
public:
    HostScene(RendererConfig &config, float3 lookFrom, float3 front, float vfov = 45.0f, float hfov = 45.0f) 
    : cameraParams{front, lookFrom}, config(config) {
        triangles = loadTriangles(config.objPath.c_str());
    }

    void loadUploadedScene() {
        std::string objPath = "../files/f" + config.jobId + ".obj";
        triangles = loadTriangles(objPath.c_str());
        notifyPrimitivesObservers();
    }

    void registerPrimitivesObserver(PrimitivesObserver* observer) {
        primitivesObservers_.push_back(observer);
    }

    void removePrimitivesObserver(PrimitivesObserver* observer) {
        auto it = std::find(
            primitivesObservers_.begin(), 
            primitivesObservers_.end(), 
            observer); 
    
        if (it != primitivesObservers_.end()) { 
            primitivesObservers_.erase(it); 
        } 
    }

    std::vector<Triangle> triangles{};
    CameraParams cameraParams;

private:
    float3 convertToFloat3(aiVector3D &v) {
        return make_float3(v.x, v.y, v.z);
    }

    std::vector<Triangle> loadTriangles(const char* file_path) {
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

    void notifyPrimitivesObservers() {
        for (PrimitivesObserver* observer : primitivesObservers_) {
            observer->updatePrimitives();
        }
    }

    std::vector<CameraObserver*> cameraObservers_;
    std::vector<PrimitivesObserver*> primitivesObservers_;
    RendererConfig &config;
};