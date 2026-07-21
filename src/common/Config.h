#pragma once
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include "cereal/archives/json.hpp"
#include "cereal/cereal.hpp"
#include "cereal/types/vector.hpp"
#include "cereal/types/map.hpp"

struct HikConfig {
    std::string ip;
    int port;
    std::string user;
    std::string pass;

    template <class Archive>
    void serialize(Archive & archive) {
        archive(CEREAL_NVP(ip), 
                CEREAL_NVP(port), 
                CEREAL_NVP(user), 
                CEREAL_NVP(pass));
    }
};

struct CameraConfig {
    std::string name;
    std::string url;
    int width = 1920;
    int height = 1080;
    int queue_max_length = 25;
    int capture_interval_ms = 0;
    bool is_full_drop = false;
    std::string decode_mode = "HARDWARE";

    template <class Archive>
    void serialize(Archive & archive) {
        archive(CEREAL_NVP(name),
                CEREAL_NVP(url),
                CEREAL_NVP(width),
                CEREAL_NVP(height),
                CEREAL_NVP(queue_max_length),
                CEREAL_NVP(capture_interval_ms),
                CEREAL_NVP(is_full_drop),
                CEREAL_NVP(decode_mode));
    }
};

struct ModelEntry {
    std::string key;
    std::string type;
    std::string path;
    std::vector<std::string> labels;
    std::map<std::string, std::string> params;

    template <class Archive>
    void serialize(Archive & archive) {
        archive(CEREAL_NVP(key),
                CEREAL_NVP(type),
                CEREAL_NVP(path),
                CEREAL_NVP(labels),
                CEREAL_NVP(params));
    }
};

struct DisplayConfig {
    int resize_width = 640;
    int resize_height = 360;

    template <class Archive>
    void serialize(Archive & archive) {
        archive(CEREAL_NVP(resize_width),
                CEREAL_NVP(resize_height));
    }
};

struct SavePathsConfig {
    std::string snapshot_dir = "./snapshots";
    std::string record_dir = "./records";

    template <class Archive>
    void serialize(Archive & archive) {
        archive(CEREAL_NVP(snapshot_dir),
                CEREAL_NVP(record_dir));
    }
};

struct RecordingConfig {
    double fps = 25.0;
    std::string codec = "MJPG";
    int queue_size = 25;

    template <class Archive>
    void serialize(Archive & archive) {
        archive(CEREAL_NVP(fps),
                CEREAL_NVP(codec),
                CEREAL_NVP(queue_size));
    }
};

struct AppConfig {
    std::vector<CameraConfig> cameras;
    std::vector<ModelEntry> models;
    DisplayConfig display;
    SavePathsConfig save_paths;
    RecordingConfig recording;
    int inference_queue_max_size = 2;

    template <class Archive>
    void serialize(Archive & archive) {
        archive(CEREAL_NVP(cameras),
                CEREAL_NVP(models),
                CEREAL_NVP(display),
                CEREAL_NVP(save_paths),
                CEREAL_NVP(recording),
                CEREAL_NVP(inference_queue_max_size));
    }

    static AppConfig loadFromFile(const std::string& path) {
        AppConfig config;
        std::ifstream file(path);
        if (!file.is_open()) {
            return config;
        }
        try {
            cereal::JSONInputArchive archive(file);
            archive(cereal::make_nvp("config", config));
        } catch (const std::exception& e) {
            fprintf(stderr, "[Config] Failed to parse %s: %s\n", path.c_str(), e.what());
        }
        fprintf(stderr, "[Config] Loaded: %zu cameras, %zu models, display %dx%d\n",
                config.cameras.size(), config.models.size(),
                config.display.resize_width, config.display.resize_height);
        return config;
    }
};
