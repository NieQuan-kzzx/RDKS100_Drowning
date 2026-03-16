#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <map>
#include <memory>

/**
 * @brief 自定义比较器，解决 cv::Size 无法直接作为 std::map 的 Key 的问题
 */
struct MatPoolKeyCompare {
    bool operator()(const std::pair<cv::Size, int>& a, const std::pair<cv::Size, int>& b) const {
        // 先比较宽度
        if (a.first.width != b.first.width)
            return a.first.width < b.first.width;
        // 宽度相同时比较高度
        if (a.first.height != b.first.height)
            return a.first.height < b.first.height;
        // 宽高相同时比较类型 (type)
        return a.second < b.second;
    }
};

/**
 * @brief 内存池类，用于重用cv::Mat对象，减少内存分配开销
 */
class MatPool {
public:
    MatPool(size_t initial_size = 10,
            cv::Size default_size = cv::Size(1920, 1080),
            int default_type = CV_8UC3);
    ~MatPool();

    cv::Mat getMat(cv::Size size = cv::Size(), int type = -1);
    void returnMat(cv::Mat mat);
    void preallocate(size_t count);
    void clear();
    size_t availableCount() const;
    size_t inUseCount() const;

    // Performance monitoring
    struct PoolStats {
        size_t total_requests = 0;
        size_t cache_hits = 0;
        size_t cache_misses = 0;
        size_t memory_usage = 0;

        double hit_rate() const {
            return total_requests > 0 ?
                static_cast<double>(cache_hits) / total_requests : 0.0;
        }
    };

    PoolStats getStats() const;
    void printStats() const;

private:
    struct MatInfo {
        cv::Mat mat;
        bool in_use;
        MatInfo(cv::Size size, int type) : mat(cv::Mat::zeros(size, type)), in_use(false) {}
    };

    std::vector<std::unique_ptr<MatInfo>> pool_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;

    cv::Size default_size_;
    int default_type_;
    std::atomic<size_t> in_use_count_;

    // Performance statistics
    std::atomic<size_t> total_requests_;
    std::atomic<size_t> cache_hits_;
    std::atomic<size_t> cache_misses_;

    std::unique_ptr<MatInfo> createMat(cv::Size size, int type);
    MatInfo* findAvailableMat(cv::Size size, int type);
};

/**
 * @brief MatPool的单例管理器
 */
class MatPoolManager {
public:
    static MatPool& getInstance() {
        static MatPool instance;
        return instance;
    }

    static MatPool& getPool(cv::Size size, int type = CV_8UC3) {
        // 修改点：显式指定自定义比较器 MatPoolKeyCompare
        static std::map<std::pair<cv::Size, int>, std::unique_ptr<MatPool>, MatPoolKeyCompare> pools;
        static std::mutex pool_mutex;

        std::lock_guard<std::mutex> lock(pool_mutex);
        auto key = std::make_pair(size, type);

        if (pools.find(key) == pools.end()) {
            pools[key] = std::make_unique<MatPool>(10, size, type);
        }

        return *pools[key];
    }

private:
    MatPoolManager() = delete;
    MatPoolManager(const MatPoolManager&) = delete;
    MatPoolManager& operator=(const MatPoolManager&) = delete;
};