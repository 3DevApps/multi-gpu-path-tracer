
#pragma once

#include <curand_kernel.h>
#include "helper_math.h"
#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
#include "assimp/scene.h"
#include <iostream>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <glm/glm.hpp>
#include "RendererConfig.h"
#include <optional>

// Materials supported by the obj loader
enum material_type {
    LAMBERTIAN,
    METAL,
    DIELECTRIC,
    DIFFUSE_LIGHT,
    UNIVERSAL
};

struct Vertex {
    float3 position;
    float2 texCoords;
};

struct HostMaterial {
    //TODO Other material params
    float3 baseColorFactor;
    int baseColorTextureIdx;
};

struct Triangle {
    Vertex v0;
    Vertex v1;
    Vertex v2;
    int textureIdx;
    int materialIdx;
};

struct HostTexture {
    int width;
    int height;
    std::vector<float3> data;
};

struct HostScene {
    std::vector<Triangle> triangles{};
    std::vector<HostTexture> textures{};
    std::vector<HostMaterial> materials{};
};

class SceneLoader {
public:
    HostScene load(std::string& path);

private:
    float3 convertToFloat3(aiVector3D &v) {
        return make_float3(v.x, v.y, v.z);
    }
    std::vector<Triangle> loadTrianglesGLTF(const aiScene *scene);

    HostMaterial processMaterial(const aiMaterial *ai_material);
    bool processMesh(const aiMesh *ai_mesh, const aiScene *ai_scene, std::vector<Triangle> &triangles, glm::mat4 &transform);
    bool processNode(const aiNode *ai_node, const aiScene *ai_scene, std::vector<Triangle> &triangles, glm::mat4 &transform);
    HostTexture loadTextureFromFile(const std::string& filename);
    std::vector<HostTexture> loadTextures(const aiScene *scene, const std::string &resDir);
    std::vector<HostMaterial> loadMaterials(const aiScene *ai_scene);

    std::unordered_map<std::string, int> textureDataCache_;
    std::mutex texCacheMutex_;
    std::string resourcePath_;
    Assimp::Importer importer;
};