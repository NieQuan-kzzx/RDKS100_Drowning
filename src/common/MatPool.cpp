#include "MatPool.h"
#include <plog/Log.h>
#include <algorithm>
#include <chrono>

MatPool::MatPool(size_t initial_size, cv::Size default_size, int default_type, size_t max_pool_size)
    : default_size_(default_size)
    , default_type_(default_type)
    , max_pool_size_(max_pool_size)
    , in_use_count_(0)
    , total_requests_(0)
    , cache_hits_(0)
    , cache_misses_(0) {

    PLOGI << "Initializing MatPool with " << initial_size << " matrices ("
          << default_size.width << "x" << default_size.height << ", type: " << default_type
          << "), max pool size: " << max_pool_size << "";

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
        available_mat->last_used = std::chrono::steady_clock::now();
        in_use_count_++;
        cache_hits_++;
        PLOGV << "Reused matrix from pool. In-use count: " << in_use_count_.load();
        return available_mat->mat;
    }

    // 如果没有找到可用的矩阵，创建新的
    // 检查是否需要清理空间
    if (pool_.size() >= max_pool_size_) {
        cleanupUnused(max_pool_size_ / 2); // 清理到一半大小
    }

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
                
                mat_info->last_used = std::chrono::steady_clock::now();

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

void MatPool::cleanupUnused(size_t target_free_count) {
    std::lock_guard<std::mutex> lock(mutex_);

    // 计算可用的（未使用）矩阵数量
    size_t available_count = 0;
    for (const auto& mat_info : pool_) {
        if (!mat_info->in_use) {
            available_count++;
        }
    }

    if (available_count <= target_free_count) {
        PLOGV << "No cleanup needed. Available count: " << available_count
              << ", target: " << target_free_count;
        return;
    }

    // 按最后使用时间排序未使用的矩阵（LRU策略）
    std::vector<std::pair<std::chrono::steady_clock::time_point, size_t>> unused_indices;
    for (size_t i = 0; i < pool_.size(); ++i) {
        if (!pool_[i]->in_use) {
            unused_indices.emplace_back(pool_[i]->last_used, i);
        }
    }

    // 按时间排序（最久未使用的在前）
    std::sort(unused_indices.begin(), unused_indices.end());

    // 计算要删除的数量
    size_t to_remove = available_count - target_free_count;
    PLOGD << "Cleaning up " << to_remove << " unused matrices from pool. Pool size: " << pool_.size();

    // 标记要删除的矩阵（从后往前删除以避免索引问题）
    for (size_t i = 0; i < to_remove && i < unused_indices.size(); ++i) {
        size_t index = unused_indices[i].second;

        // 计算内存节省
        cv::Mat mat = pool_[index]->mat;
        size_t memory_freed = mat.total() * mat.elemSize();

        PLOGD << "Removing unused matrix (size: " << mat.size() << ", type: " << mat.type()
              << ", memory freed: " << memory_freed / 1024 << " KB)";

        // 使用swap-and-pop技术删除元素
        pool_[index] = std::move(pool_.back());
        pool_.pop_back();
    }

    PLOGI << "Cleanup completed. New pool size: " << pool_.size()
          << ", target free count: " << target_free_count;
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