#pragma once

#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

template <typename T>
class ThreadSafeQueue
{
public:
    explicit ThreadSafeQueue(size_t max_size = SIZE_MAX)
        : max_size(max_size) {}

    // 入队操作（队列满时丢弃最旧元素）
    void enqueue(const T &item)
    {
        std::lock_guard<std::mutex> lock(mtx);

        // 如果满，则丢弃队头（最老的数据）
        if (q.size() >= max_size)
        {
            q.pop();
        }

        q.push(item);
        cv.notify_one();
    }

    // 清空队列
    void clear()
    {
        std::lock_guard<std::mutex> lock(mtx);
        std::queue<T> empty;
        std::swap(q, empty);
    }

    // 唤醒所有等待的线程（防止退出时 dequeue 卡死）
    void abort()
    {
        cv.notify_all();
    }

    // 阻塞式出队操作
    T dequeue()
    {
        std::unique_lock<std::mutex> lock(mtx);

        cv.wait(lock, [this]()
                { return !q.empty(); });

        T item = q.front();
        q.pop();
        return item;
    }

    T dequeue_nonblocking()
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (q.empty())
        {
            throw std::runtime_error("Queue is empty");
        }
        T item = q.front();
        q.pop();
        return item;
    }

    // 非阻塞的检查队列是否为空
    bool empty() const
    {
        std::lock_guard<std::mutex> lock(mtx);
        return q.empty();
    }

    // 获取队列的大小
    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mtx);
        return q.size();
    }

private:
    std::queue<T> q;
    size_t max_size;

    mutable std::mutex mtx;
    std::condition_variable cv;
};
