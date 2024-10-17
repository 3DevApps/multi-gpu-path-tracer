#pragma once

#include <cstdlib>
#include <iostream>

#include <curand_kernel.h>
#include "helper_math.h"

class HostMaterial {
public:
    HostMaterial(const aiMaterial *ai_material, std::unordered_map<std::string, std::shared_ptr<HostTexture>>& textureCache, std::string &resourcePath) {
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

                std::string absolutePath = resourcePath + "/" + texPath.C_Str();
                if (textureCache.find(absolutePath) == textureCache.end()) {
                    std::cout << "Texture not loaded, path: " << absolutePath;
                    return;
                }

                auto tex = textureCache[absolutePath];
                switch (textureType) {
                    case aiTextureType_BASE_COLOR:
                        baseColorTextureIdx = tex->index;
                        break;
                    default:
                        // std::cout << "Texture type unsupported" << std::endl;
                        return;  
                }
            }
        }
    }
    int baseColorTextureIdx = 0;
private:
    // material_type type;
    float3 color_ambient;
    float3 color_diffuse;
    float index_of_refraction;
    float shininess;
    
};