#include "MatPool.h"
#include <plog/Log.h>
#include <algorithm>

MatPool::MatPool(size_t initial_size, cv::Size default_size, int default_type)
    : default_size_(default_size)
    , default_type_(default_type)
    , in_use_count_(0)
    , total_requests_(0)
    , cache_hits_(0)
    , cache_misses_(0) {

    PLOGI << "Initializing MatPool with " << initial_size << " matrices ("
          << default_size.width << "x" << default_size.height << ", type: " << default_type << ")";

    preallocate(initial_size);
}

MatPool::~MatPool() {
    clear();
    PLOGI << "MatPool destroyed. Total in-use matrices at destruction: " << in_use_count_.load();
}

cv::Mat MatPool::getMat(cv::Size size, int type) {
    if (size == cv::Size()) {
        size = default_size_;
    }
    if (type == -1) {
        type = default_type_;
    }

    std::unique_lock<std::mutex> lock(mutex_);

    // 更新请求统计
    total_requests_++;

    // 查找可用的矩阵
    MatInfo* available_mat = findAvailableMat(size, type);

    if (available_mat != nullptr) {
        available_mat->in_use = true;
        in_use_count_++;
        cache_hits_++;
        PLOGV << "Reused matrix from pool. In-use count: " << in_use_count_.load();
        return available_mat->mat;
    }

    // 如果没有找到可用的矩阵，创建新的
    auto new_mat_info = createMat(size, type);
    cv::Mat new_mat = new_mat_info->mat;
    new_mat_info->in_use = true;

    pool_.push_back(std::move(new_mat_info));
    in_use_count_++;
    cache_misses_++;

    PLOGD << "Created new matrix (size: " << size.width << "x" << size.height
          << ", type: " << type << "). Pool size: " << pool_.size()
          << ", In-use count: " << in_use_count_.load();

    return new_mat;
}

void MatPool::returnMat(cv::Mat mat) {
    if (mat.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // 查找对应的MatInfo
    for (auto& mat_info : pool_) {
        // 通过 mat.data 指针判断是否为同一个显存块
        if (mat_info->mat.data == mat.data) {
            if (mat_info->in_use) {
                mat_info->in_use = false;
                if (in_use_count_ > 0) in_use_count_--;
                
                // 注意：地平线 RDK 上如果涉及硬件加速，通常不建议频繁 setTo(0)，会消耗 CPU
                // mat.setTo(cv::Scalar::all(0)); 

                PLOGV << "Returned matrix to pool. In-use count: " << in_use_count_.load();
            }
            return;
        }
    }

    PLOGW << "Attempted to return matrix that was not from this pool.";
}

void MatPool::preallocate(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t current_size = pool_.size();
    for (size_t i = 0; i < count; ++i) {
        pool_.push_back(createMat(default_size_, default_type_));
    }

    PLOGI << "Preallocated " << count << " matrices. Total pool size: " << pool_.size();
}

void MatPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (in_use_count_ > 0) {
        PLOGW << "Clearing pool with " << in_use_count_ << " matrices still in use!";
    }

    pool_.clear();
    in_use_count_ = 0;

    PLOGI << "MatPool cleared.";
}

size_t MatPool::availableCount() const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t count = 0;
    for (const auto& mat_info : pool_) {
        if (!mat_info->in_use) {
            count++;
        }
    }

    return count;
}

size_t MatPool::inUseCount() const {
    return in_use_count_.load();
}

std::unique_ptr<MatPool::MatInfo> MatPool::createMat(cv::Size size, int type) {
    try {
        auto mat_info = std::make_unique<MatInfo>(size, type);
        return mat_info;
    } catch (const std::exception& e) {
        PLOGE << "Failed to create matrix (size: " << size.width << "x" << size.height
              << ", type: " << type << "): " << e.what();
        throw;
    }
}

MatPool::MatInfo* MatPool::findAvailableMat(cv::Size size, int type) {
    for (auto& mat_info : pool_) {
        if (!mat_info->in_use &&
            mat_info->mat.size() == size &&
            mat_info->mat.type() == type) {
            return mat_info.get();
        }
    }
    return nullptr;
}

MatPool::PoolStats MatPool::getStats() const {
    std::lock_guard<std::mutex> lock(mutex_);

    PoolStats stats;
    stats.total_requests = total_requests_.load();
    stats.cache_hits = cache_hits_.load();
    stats.cache_misses = cache_misses_.load();

    // Calculate memory usage
    size_t total_memory = 0;
    for (const auto& mat_info : pool_) {
        cv::Mat mat = mat_info->mat;
        total_memory += mat.total() * mat.elemSize();
    }
    stats.memory_usage = total_memory;

    return stats;
}

void MatPool::printStats() const {
    auto stats = getStats();
    PLOGI << "=== MatPool Performance Statistics ===";
    PLOGI << "Total Requests: " << stats.total_requests;
    PLOGI << "Cache Hits: " << stats.cache_hits;
    PLOGI << "Cache Misses: " << stats.cache_misses;
    PLOGI << "Hit Rate: " << (stats.hit_rate() * 100) << "%";
    PLOGI << "Memory Usage: " << stats.memory_usage / 1024 / 1024 << " MB";
    PLOGI << "Pool Size: " << pool_.size();
    PLOGI << "In Use Count: " << in_use_count_.load();
    PLOGI << "Available Count: " << availableCount();
    PLOGI << "=====================================";
}