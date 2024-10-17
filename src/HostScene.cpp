#include "HostScene.h"
#include <glm/glm.hpp>
#include <thread>
#include <set>
#include "ThreadPool.h"

std::shared_ptr<HostTexture> HostScene::loadTextureFile(const std::string &path) {
    texCacheMutex_.lock();
    if (textureDataCache_.find(path) != textureDataCache_.end()) {
        auto &tex = textureDataCache_[path];
        texCacheMutex_.unlock();
        return tex;
    }
    texCacheMutex_.unlock();

    auto tex = std::make_shared<HostTexture>();
    if (!tex->load(path)) {
        std::cout << "load texture failed, path" << std::endl;
        return nullptr;
    }

    texCacheMutex_.lock();
    textureDataCache_[path] = tex;
    tex->index = textures.size();
    textures.push_back(tex);
    texCacheMutex_.unlock();
    return tex;
}

void HostScene::preloadTextureFiles(const aiScene *scene, const std::string &resDir) {
    std::set<std::string> texPaths;
    for (int materialIdx = 0; materialIdx < scene->mNumMaterials; materialIdx++) {
        aiMaterial *material = scene->mMaterials[materialIdx];
        for (int texType = aiTextureType_NONE; texType <= AI_TEXTURE_TYPE_MAX; texType++) {
            auto textureType = static_cast<aiTextureType>(texType);
            size_t texCnt = material->GetTextureCount(textureType);
            for (size_t i = 0; i < texCnt; i++) {
                aiTextureMapMode texMapMode[2]; 
                aiString textPath;
                aiReturn retStatus = material->GetTexture(textureType, i, &textPath, nullptr, nullptr, nullptr, nullptr, &texMapMode[0]);
                // std::cout << "Texture mode: " << texMapMode[0] << " " << texMapMode[1] << std::endl; 
                if (retStatus != aiReturn_SUCCESS || textPath.length == 0) {
                    continue;
                }
                texPaths.insert(resDir + "/" + textPath.C_Str());
            }
        }
    }
    if (texPaths.empty()) {
        return;
    }

    ThreadPool pool(std::min(texPaths.size(), (size_t) std::thread::hardware_concurrency()));
    for (auto &path : texPaths) {
        pool.pushTask([&](int thread_id) {
            loadTextureFile(path);
        });
    }
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

void HostScene::loadMaterials(const aiScene *scene) {
    if (scene->HasMaterials()) {
        for (unsigned int i = 0; i < scene->mNumMaterials; ++i) {
            aiMaterial* material = scene->mMaterials[i];
            aiString name;
            material->Get(AI_MATKEY_NAME, name);

            auto mat = std::make_shared<HostMaterial>(material, textureDataCache_, resourcePath_);
            materials.push_back(mat);
            // Only base color for now
            // auto texFound = scene->mMaterials[i]->GetTexture(aiTextureType_BASE_COLOR, i, &name);

            // if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
            //     aiString path;
            //     if (material->GetTexture(aiTextureType_DIFFUSE, 0, &path, NULL, NULL, NULL, NULL, NULL) == AI_SUCCESS) {
            //         std::string fullPath = path.data;
            //     }
            // }
        }
    }
}



HostScene::HostScene(RendererConfig &config, float3 lookFrom, float3 front, float vfov, float hfov)
    : cameraParams{front, lookFrom}, config(config) {

    auto& objPath = config.objPath;
    const aiScene *scene = importer.ReadFile(objPath, aiProcess_Triangulate | aiProcess_FindDegenerates);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        throw std::runtime_error(importer.GetErrorString());
    }

    size_t typePos = objPath.find_last_of('.');
    if (typePos == std::string::npos) {
        std::cout << "Wrong file type" << std::endl;
        return;
    }

    auto type = objPath.substr(typePos + 1);
    if (type == "obj") {
        std::cout << "loading OBJ ..." << std::endl;
        triangles = loadTrianglesOBJ(scene);
    } 
    else if (type == "gltf") {
        std::cout << "loading GLTF ..." << std::endl;
        resourcePath_ = objPath.substr(0, objPath.find_last_of('/'));
        preloadTextureFiles(scene, resourcePath_);

        std::cout << "Loading materials" << std::endl; 
        loadMaterials(scene);

        std::cout << "loafing triangles gltf" << std::endl;
        triangles = loadTrianglesGLTF(scene);
    }
    else {
        std::cout << "Unsupported file type" << std::endl;
        return;   
    }

    std::cout << "Number of triangles: " << triangles.size() << std::endl;
    // triangles = std::vector<Triangle>(triangles.begin(), triangles.begin() + 100000);
    // std::cout << "Number of triangles: " << triangles.size() << std::endl;
}

// bool HostScene::processMaterialNew(const aiMaterial *ai_material) {
//     if (ai_material->GetTextureCount(textureType) <= 0) {
//         return true;
//     }

//     for (aiTextureType textureType = 0; textureType <= AI_TEXTURE_TYPE_MAX; textureType++) {
//         for (size_t i = 0; i < ai_material->GetTextureCount(textureType); i++) {
//             aiTextureMapMode texMapMode[2];  // [u, v] //only clamp
//             aiString texPath;
//             aiReturn retStatus = ai_material->GetTexture(textureType, i, &texPath, nullptr, nullptr, nullptr, nullptr, &texMapMode[0]);

//             if (retStatus != aiReturn_SUCCESS || texPath.length == 0) {
//                 std::cout << "load texture type=" << textureType << "failed with return value=" << retStatus  << "texPath: " << texPath.C_Str() << std::endl;
//                 continue;
//             }

//             std::string absolutePath = resourcePath_ + "/" + texPath.C_Str();
//             if (textureDataCache_.find(absolutePath) == textureDataCache_.end()) {
//             std::cout << "Texture not loaded, path: " << absolutePath;
//             return false;
//             }

//             auto tex = textureDataCache_[absolutePath];
//             switch (textureType) {
//                 case aiTextureType_BASE_COLOR:
//                 case aiTextureType_DIFFUSE:
//                     triangle.textureIdx = tex->index;
//                     break;
//                 default:
//                     // std::cout << "Texture type unsupported" << std::endl;
//                     return false;  
//             }
//         }
//     }
// }


bool HostScene::processMaterial(const aiMaterial *ai_material, aiTextureType textureType, Triangle &triangle) {
    if (ai_material->GetTextureCount(textureType) <= 0) {
        return true;
    }

    for (size_t i = 0; i < ai_material->GetTextureCount(textureType); i++) {
        aiTextureMapMode texMapMode[2];  // [u, v] //only clamp
        aiString texPath;
        aiReturn retStatus = ai_material->GetTexture(textureType, i, &texPath, nullptr, nullptr, nullptr, nullptr, &texMapMode[0]);
        if (retStatus != aiReturn_SUCCESS || texPath.length == 0) {
            std::cout << "load texture type=" << textureType << "texPath: " << texPath.C_Str() << "idk material" << std::endl;
            continue;
        }

        std::string absolutePath = resourcePath_ + "/" + texPath.C_Str();
        if (textureDataCache_.find(absolutePath) == textureDataCache_.end()) {
          std::cout << "Texture not loaded, path: " << absolutePath;
          return false;
        }

        auto tex = textureDataCache_[absolutePath];
        switch (textureType) {
            case aiTextureType_BASE_COLOR:
            case aiTextureType_DIFFUSE:
                triangle.textureIdx = tex->index;
                break;
            default:
                // std::cout << "Texture type unsupported" << std::endl;
                return false;  
        }
    }
}

bool HostScene::processMesh(const aiMesh *ai_mesh, const aiScene *ai_scene, std::vector<Triangle> &triangles, glm::mat4 &transform) {
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
        if (ai_mesh->mMaterialIndex >= 0) {
            const aiMaterial *material = ai_scene->mMaterials[ai_mesh->mMaterialIndex];

            for (int i = 0; i <= AI_TEXTURE_TYPE_MAX; i++) {
                processMaterial(material, static_cast<aiTextureType>(i), triangle);
            }
        }

    }
    for (size_t i = 0; i < ai_mesh->mNumFaces; i++) {

        // std::cout << "ModelLoader:: number of facesss" << std::endl;
        aiFace face = ai_mesh->mFaces[i];
        if (face.mNumIndices != 3) {
            std::cout << "ModelLoader::processMesh, mesh not transformed to triangle mesh." << std::endl;
            return false;
        }

        triangle.v0 = vertices[(int)(face.mIndices[0])];
        triangle.v1 = vertices[(int)(face.mIndices[1])];
        triangle.v2 = vertices[(int)(face.mIndices[2])];

        triangle.material_params.type = UNIVERSAL;
        triangle.material_params.color_ambient = make_float3(0.5, 0.5, 0.5);
        triangles.push_back(triangle);
    }
    return true;
}

bool HostScene::processNode(const aiNode *ai_node, const aiScene *ai_scene, std::vector<Triangle> &triangles, glm::mat4 &transform) {
    if (!ai_node) {
        std::cout << "node empty" << std::endl;
        return false;
    }




    // std::cout << "processing node" << std::endl;

    glm::mat4 t = convertMatrix(ai_node->mTransformation);
    auto currTransform = transform * t;

    for (size_t i = 0; i < ai_node->mNumMeshes; i++) {
        // std::cout << "processing mesh, are there any meshes??" << std::endl;
        const aiMesh *meshPtr = ai_scene->mMeshes[ai_node->mMeshes[i]];
        if (meshPtr) {

            // std::cout << "processing mesh" << std::endl;

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

std::vector<Triangle> HostScene::loadTrianglesGLTF(const aiScene *scene) {
    int num_materials = scene->mNumMaterials;
    std::vector<Triangle> triangles; 

    auto currTransform = glm::mat4(1.f);

    std::cout << "preocess node here" << std::endl;
    processNode(scene->mRootNode, scene, triangles, currTransform);
    return triangles;
}

std::vector<Triangle> HostScene::loadTrianglesOBJ(const aiScene *scene) {
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
                Triangle triangle; 

                triangle.v0.position = v0;
                triangle.v1.position = v1;
                triangle.v2.position = v2;

                triangle.material_params = m;
                triangles.push_back(triangle);
            }
        }
    }
    return triangles;
}


