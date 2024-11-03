#include "HostScene.h"
#include <glm/glm.hpp>
#include <thread>
#include <set>

#define STB_IMAGE_IMPLEMENTATION
#include "../third-party/stb_image.h"


HostTexture SceneLoader::loadTextureFromFile(const aiScene *scene, const std::string& resDir, const std::string& filename) {
    float *fdata = nullptr; 
    int bytes_per_pixel = 3;
    auto n = bytes_per_pixel; 
    std::vector<float3> data;
    int width, height, nrComponents;
    unsigned char* raw_data;

    if (auto* embeddedTex = scene->GetEmbeddedTexture(filename.c_str())) {
	    raw_data = stbi_load_from_memory(
            reinterpret_cast<unsigned char*>(embeddedTex->pcData), 
            embeddedTex->mWidth, 
            &width, 
            &height, 
            &nrComponents, 
            0
        );
    }
    else {
        std::string path = resDir + "/" + filename;
        raw_data = stbi_load(path.c_str(), &width, &height, &n, bytes_per_pixel);
    }

    if (raw_data == nullptr) {
        throw std::runtime_error("Cannot load texture data, path: " + filename);
    }

    data.resize(width * height);
    int total_bytes = width * height * bytes_per_pixel;

    for(int i = 0, j = 0; i < total_bytes; i += 3, j++) {
        data[j] = make_float3(
            raw_data[i],
            raw_data[i + 1],
            raw_data[i + 2]
        );
    }

    STBI_FREE(raw_data);
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
                textureDataCache_[textPath.C_Str()] = textures.size();
                textures.push_back(loadTextureFromFile(scene, resDir, textPath.C_Str()));
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
    importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_POINT | aiPrimitiveType_LINE);

    const aiScene *scene = importer.ReadFile(
        path, 
        aiProcess_Triangulate | 
        aiProcess_FindDegenerates | 
        aiProcess_PreTransformVertices | 
        aiProcess_FindDegenerates |
        aiProcess_SortByPType
    );
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
    if (type == "gltf" || type == "glb") {
        std::cout << "loading GLTF ..." << std::endl;
        resourcePath_ = path.substr(0, path.find_last_of('/'));
        hostScene.textures = loadTextures(scene, resourcePath_);    

        std::cout << "Loading materials" << std::endl; 
        hostScene.materials = loadMaterials(scene);

        std::cout << "loafing triangles gltf" << std::endl;
        hostScene.triangles = loadTrianglesGLTF(scene);
    }
    else {
        throw std::runtime_error("Unsupported file type" + path); 
    }

    std::cout << "Number of triangles: " << hostScene.triangles.size() << std::endl;
    return hostScene;
}

float3 convertColorToFloat3(aiColor3D color) {
    return make_float3(color.r, color.g, color.b);
}

HostMaterial SceneLoader::processMaterial(const aiMaterial *ai_material) {
    HostMaterial material;

    //base color
    aiColor3D baseColor(1.f, 1.f, 1.f);
    ai_material->Get(AI_MATKEY_BASE_COLOR, baseColor);
    material.baseColor = convertColorToFloat3(baseColor);

    //emissiveFactor
    aiColor3D emissiveFactor(1.f, 1.f, 1.f);
    ai_material->Get(AI_MATKEY_COLOR_EMISSIVE, emissiveFactor);
    material.emissiveFactor = convertColorToFloat3(emissiveFactor);

    for (int k = 0; k <= AI_TEXTURE_TYPE_MAX; k++) {
        auto textureType = static_cast<aiTextureType>(k);
        if (ai_material->GetTextureCount(textureType) == 0) {
            continue;
        }
            aiTextureMapMode texMapMode[2];  
            aiString texPath;
            aiReturn retStatus = ai_material->GetTexture(textureType, 0, &texPath, nullptr, nullptr, nullptr, nullptr, &texMapMode[0]);

            if (retStatus != aiReturn_SUCCESS || texPath.length == 0) {
                std::cout << "load texture type=" << textureType << "failed with return value=" << retStatus  << "texPath: " << texPath.C_Str() << std::endl;
                continue;
            }

            if (textureDataCache_.find(texPath.C_Str()) == textureDataCache_.end()) {
                std::cout << "Texture not loaded, path: " << texPath.C_Str();
                continue;
            }

            auto tex = textureDataCache_[texPath.C_Str()];
            switch (textureType) {
                case aiTextureType_BASE_COLOR:
                    material.baseColorTextureIdx = tex;
                    break;
                case aiTextureType_EMISSIVE:
                    std::cout << "Loading emissive" << std::endl;
                    material.emissiveTextureIdx = tex;
                default:
                    continue;  
            }
    }
    return material;
}

bool SceneLoader::processMesh(const aiMesh *ai_mesh, const aiScene *ai_scene, std::vector<Triangle> &triangles) {
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
        if (face.mNumIndices < 3) {
            std::cout << "Igoring line" << std::endl;
            return false;
        }

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

bool SceneLoader::processNode(const aiNode *ai_node, const aiScene *ai_scene, std::vector<Triangle> &triangles) {
    if (!ai_node) {
        std::cout << "node empty" << std::endl;
        return false;
    }

    std::cout << "Processing node" << std::endl;

    for (size_t i = 0; i < ai_node->mNumMeshes; i++) {
        const aiMesh *meshPtr = ai_scene->mMeshes[ai_node->mMeshes[i]];
        if (meshPtr) {
            if (!processMesh(meshPtr, ai_scene, triangles)) {
                std::cout << "mesh processing failed" << std::endl;
            }
        }
        else {
            std::cout << "processing mesh, but pointer empty" << std::endl;
        }
    }

    for (size_t i = 0; i < ai_node->mNumChildren; i++) {
        if (processNode(ai_node->mChildren[i], ai_scene, triangles)) {}
    }
    return true;
}

std::vector<Triangle> SceneLoader::loadTrianglesGLTF(const aiScene *scene) {
    int num_materials = scene->mNumMaterials;
    std::vector<Triangle> triangles; 

    std::cout << "preocess node here" << std::endl;
    processNode(scene->mRootNode, scene, triangles);
    return triangles;
}