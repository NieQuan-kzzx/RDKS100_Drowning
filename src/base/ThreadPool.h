#pragma once

#include <future>
#include <queue>

class ThreadPool
{
public:
    ThreadPool(size_t);
    ~ThreadPool();

    template <class F, class... Args>
    auto enqueue(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>;

private:
    void work();                             // 线程池中各线程执行的内容
    bool isStop;                             // 是否停止线程池
    std::vector<std::thread> workers;        // 线程集合(线程池)
    std::queue<std::function<void()>> tasks; // 任务队列
    std::mutex queue_mutex;
    std::condition_variable cv;
};

inline void ThreadPool::work()
{
    while (1)
    {
        std::function<void()> task;

        {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            cv.wait(lock, [this]()
                    { return this->isStop || !this->tasks.empty(); });
            if (this->isStop && tasks.empty())
                return;

            // 从任务队列取出任务执行
            task = std::move(this->tasks.front());
            this->tasks.pop();
        }

        task();
    }
}

inline ThreadPool::ThreadPool(size_t threads)
    : isStop(false)
{
    for (size_t i = 0; i < threads; ++i)
        workers.emplace_back([this]()
                             { this->work(); });
}

inline ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        isStop = true;
    }
    // 确保线程池里的所有任务都退出
    cv.notify_all();
    for (std::thread &worker : workers)
        worker.join();
}

// 向任务队列添加个任务,并通知一个线程来执行该任务,结果通过std::future返回
template <class F, class... Args>
auto ThreadPool::enqueue(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>
{
    // 获得f的返回值类型
    using return_type = typename std::result_of<F(Args...)>::type;

    //将f(args...)包装成一个异步任务,并用一个智能指针指向它
    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();

    //将f(args...)包装成一个std::function<void()>类型的可调用对象
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        if (isStop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task]()
                      { (*task)(); });
    }

    // 通知一个线程来执行该任务
    cv.notify_one();
    return res;
}
