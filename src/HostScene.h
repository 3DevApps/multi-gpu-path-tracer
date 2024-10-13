
#pragma once

#include <curand_kernel.h>
#include "helper_math.h"
#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include <iostream>
#include <vector>
#include "HostTexture.h"
#include <memory>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <glm/glm.hpp>
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
    DIFFUSE_LIGHT,
    UNIVERSAL
};

struct m_ai_material {
    material_type type ;
    float3 color_ambient;
    float3 color_diffuse;
    float index_of_refraction;
    float shininess;
};

struct Vertex {
    float3 position;
    float2 texCoords;
};

struct Triangle {
    Vertex v0;
    Vertex v1;
    Vertex v2;
    m_ai_material material_params;
    std::shared_ptr<HostTexture> texture;
    int textureIdx;
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
    // HostScene(std::string &objPath, float3 lookFrom, float3 front, float vfov = 45.0f, float hfov = 45.0f);

    void setObjPath(std::string &objPath) {
        // triangles = loadTriangles(objPath.c_str());
        notifyPrimitivesObservers();
    }

    HostScene(RendererConfig &config, float3 lookFrom, float3 front, float vfov = 45.0f, float hfov = 45.0f);
    // : cameraParams{front, lookFrom}, config(config);
    // {
    //     triangles = loadTriangles(config.objPath.c_str());
    // }

    void loadUploadedScene() {
        // std::string objPath = "../files/f" + config.jobId + ".obj";

        // const aiScene *scene = importer.ReadFile(objPath, aiProcess_Triangulate | aiProcess_FindDegenerates);
        // if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        //     throw std::runtime_error(importer.GetErrorString());
        // }

        // triangles = loadTrianglesOBJ(scene);
        // notifyPrimitivesObservers();
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
    std::vector<std::shared_ptr<HostTexture>> textures{};
    std::vector<std::vector<std::shared_ptr<HostTexture>>> textures_{AI_TEXTURE_TYPE_MAX};
    // std::vector<Texture> textures{};

private:
    float3 convertToFloat3(aiVector3D &v) {
        return make_float3(v.x, v.y, v.z);
    }

    std::vector<Triangle> loadTrianglesGLTF(const aiScene *scene);
    std::vector<Triangle> loadTrianglesOBJ(const aiScene *scene);

    void notifyPrimitivesObservers() {
        for (PrimitivesObserver* observer : primitivesObservers_) {
            observer->updatePrimitives();
        }
    }

    bool processMaterial(const aiMaterial *ai_material, aiTextureType textureType, Triangle &triangle);
    bool processMesh(const aiMesh *ai_mesh, const aiScene *ai_scene, std::vector<Triangle> &triangles, glm::mat4 &transform);
    bool processNode(const aiNode *ai_node, const aiScene *ai_scene, std::vector<Triangle> &triangles, glm::mat4 &transform);


    std::shared_ptr<HostTexture> loadTextureFile(const std::string &path);
    void preloadTextureFiles(const aiScene *scene, const std::string &resDir);

    std::unordered_map<std::string, std::shared_ptr<HostTexture>> textureDataCache_;
    std::vector<CameraObserver*> cameraObservers_;
    std::vector<PrimitivesObserver*> primitivesObservers_;
    std::mutex texCacheMutex_;
    std::string resourcePath_;
    RendererConfig &config;
    Assimp::Importer importer;
};