#include "HostScene.h"
#include <glm/glm.hpp>
#include <thread>
#include <set>

#define STB_IMAGE_IMPLEMENTATION
#include "../third-party/stb_image.h"


unsigned char floatToByte(float value) {
    if (value <= 0.0)
        return 0;
    if (1.0 <= value)
        return 255;
    return static_cast<unsigned char>(256.0 * value);
}

HostTexture SceneLoader::loadTextureFromFile(const std::string& filename) {
    float *fdata = nullptr; 
    int bytes_per_pixel = 3;
    auto n = bytes_per_pixel; 
    std::vector<float3> data;
    int width, height;

    fdata = stbi_loadf(filename.c_str(), &width, &height, &n, bytes_per_pixel);
    if (fdata == nullptr) {
        throw std::runtime_error("Cannot load texture data, path: " + filename);
    }

    data.resize(width * height);
    int total_bytes = width * height * bytes_per_pixel;

    for(int i = 0, j = 0; i < total_bytes; i += 3, j++) {
        data[j] = make_float3(
            floatToByte(fdata[i]),
            floatToByte(fdata[i + 1]),
            floatToByte(fdata[i + 2])
        );
    }

    STBI_FREE(fdata);
    HostTexture tex{width, height, std::move(data)};
    return tex;
}

std::vector<HostTexture> SceneLoader::loadTextures(const aiScene *scene, const std::string &resDir) {
    std::vector<HostTexture> textures;
    for (int materialIdx = 0; materialIdx < scene->mNumMaterials; materialIdx++) {
        aiMaterial *material = scene->mMaterials[materialIdx];
        for (int texType = aiTextureType_NONE; texType <= AI_TEXTURE_TYPE_MAX; texType++) {
            auto textureType = static_cast<aiTextureType>(texType);
            size_t texCnt = material->GetTextureCount(textureType);
            for (size_t i = 0; i < texCnt; i++) {
                aiTextureMapMode texMapMode[2]; 
                aiString textPath;
                aiReturn retStatus = material->GetTexture(textureType, i, &textPath, nullptr, nullptr, nullptr, nullptr, &texMapMode[0]);
                if (retStatus != aiReturn_SUCCESS || textPath.length == 0) {
                    continue;
                }

                std::string path = resDir + "/" + textPath.C_Str();
                textureDataCache_[path] = textures.size();
                textures.push_back(loadTextureFromFile(path));
            }
        }
    }
    return textures;
}

glm::mat4 convertMatrix(const aiMatrix4x4 &m) {
    glm::mat4 ret;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            ret[j][i] = m[i][j];
        }
    }
    return ret;
}

std::vector<HostMaterial> SceneLoader::loadMaterials(const aiScene *scene) {
    std::vector<HostMaterial> materials;
    if (scene->HasMaterials()) {
        for (unsigned int i = 0; i < scene->mNumMaterials; ++i) {
            aiMaterial* material = scene->mMaterials[i];
            aiString name;
            material->Get(AI_MATKEY_NAME, name);
            materials.push_back(processMaterial(material));
        }
    }
    return materials;
}

HostScene SceneLoader::load(std::string& path) {
    const aiScene *scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FindDegenerates);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        throw std::runtime_error(importer.GetErrorString());
    }

    size_t typePos = path.find_last_of('.');
    if (typePos == std::string::npos) {
        std::cout << "Wrong file type" << std::endl;
        throw std::runtime_error("Wrong file name privided: " + path);
    }


    HostScene hostScene;
    auto type = path.substr(typePos + 1);
    if (type == "gltf") {
        std::cout << "loading GLTF ..." << std::endl;
        resourcePath_ = path.substr(0, path.find_last_of('/'));
        hostScene.textures = loadTextures(scene, resourcePath_);

        std::cout << "Loading materials" << std::endl; 
        hostScene.materials = loadMaterials(scene);

        std::cout << "loafing triangles gltf" << std::endl;
        hostScene.triangles = loadTrianglesGLTF(scene);
    }
    else {
        //TODO support for old objs
        // throw std::runtime_error("Unsupported file type" + path); 
    }

    std::cout << "Number of triangles: " << hostScene.triangles.size() << std::endl;
    return hostScene;
}

HostMaterial SceneLoader::processMaterial(const aiMaterial *ai_material) {
    HostMaterial material;
    for (int k = 0; k <= AI_TEXTURE_TYPE_MAX; k++) {
        auto textureType = static_cast<aiTextureType>(k);
        for (size_t i = 0; i < ai_material->GetTextureCount(textureType); i++) {
            aiTextureMapMode texMapMode[2];  // [u, v] //only clamp
            aiString texPath;
            aiReturn retStatus = ai_material->GetTexture(textureType, i, &texPath, nullptr, nullptr, nullptr, nullptr, &texMapMode[0]);

            if (retStatus != aiReturn_SUCCESS || texPath.length == 0) {
                std::cout << "load texture type=" << textureType << "failed with return value=" << retStatus  << "texPath: " << texPath.C_Str() << std::endl;
                continue;
            }

            std::string absolutePath = resourcePath_ + "/" + texPath.C_Str();
            if (textureDataCache_.find(absolutePath) == textureDataCache_.end()) {
                std::cout << "Texture not loaded, path: " << absolutePath;
                continue;
            }

            auto tex = textureDataCache_[absolutePath];
            switch (textureType) {
                case aiTextureType_BASE_COLOR:
                    material.baseColorTextureIdx = tex;
                    break;
                default:
                    continue;  
            }
        }
    }
    return material;
}

bool SceneLoader::processMesh(const aiMesh *ai_mesh, const aiScene *ai_scene, std::vector<Triangle> &triangles, glm::mat4 &transform) {
    std::vector<Vertex> vertices;
    Triangle triangle;

    for (size_t i = 0; i < ai_mesh->mNumVertices; i++) {
        vertices.push_back({make_float3(0, 0, 0), make_float2(0, 0)});


        if (ai_mesh->HasPositions()) {
            glm::vec4 vec(
                ai_mesh->mVertices[i].x,
                ai_mesh->mVertices[i].y,
                ai_mesh->mVertices[i].z,
                1.0f
            );

            vec = vec * transform;
            vertices[i].position.x = vec.x;
            vertices[i].position.y = vec.y;
            vertices[i].position.z = vec.z;

            if (ai_mesh->HasTextureCoords(0)) {
                vertices[i].texCoords = make_float2(
                  ai_mesh->mTextureCoords[0][i].x,
                  ai_mesh->mTextureCoords[0][i].y
                );
            } 
            else {
                vertices[i].texCoords = make_float2(0.0f, 0.0f);
            }
        }

        triangle.materialIdx = ai_mesh->mMaterialIndex;
    }

    for (size_t i = 0; i < ai_mesh->mNumFaces; i++) {
        aiFace face = ai_mesh->mFaces[i];
        if (face.mNumIndices != 3) {
            std::cout << "ModelLoader::processMesh, mesh not transformed to triangle mesh." << std::endl;
            return false;
        }

        triangle.v0 = vertices[(int)(face.mIndices[0])];
        triangle.v1 = vertices[(int)(face.mIndices[1])];
        triangle.v2 = vertices[(int)(face.mIndices[2])];
        triangles.push_back(triangle);
    }
    return true;
}

bool SceneLoader::processNode(const aiNode *ai_node, const aiScene *ai_scene, std::vector<Triangle> &triangles, glm::mat4 &transform) {
    if (!ai_node) {
        std::cout << "node empty" << std::endl;
        return false;
    }

    glm::mat4 t = convertMatrix(ai_node->mTransformation);
    auto currTransform = transform * t;
    for (size_t i = 0; i < ai_node->mNumMeshes; i++) {
        const aiMesh *meshPtr = ai_scene->mMeshes[ai_node->mMeshes[i]];
        if (meshPtr) {
            if (!processMesh(meshPtr, ai_scene, triangles, currTransform)) {
                std::cout << "mesh processing failed" << std::endl;
            }
        }
        else {
            std::cout << "processing mesh, but pointer empty" << std::endl;
        }
    }

    for (size_t i = 0; i < ai_node->mNumChildren; i++) {
        if (processNode(ai_node->mChildren[i], ai_scene, triangles, currTransform)) {}
    }
    return true;
}

std::vector<Triangle> SceneLoader::loadTrianglesGLTF(const aiScene *scene) {
    int num_materials = scene->mNumMaterials;
    std::vector<Triangle> triangles; 

    auto currTransform = glm::mat4(1.f);

    std::cout << "preocess node here" << std::endl;
    processNode(scene->mRootNode, scene, triangles, currTransform);
    return triangles;
}