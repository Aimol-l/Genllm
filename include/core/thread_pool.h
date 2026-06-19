#pragma once
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <vector>

#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>



// 没啥收益，暂时弃用

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency())
        : stop_(false), outstanding_(0)
    {
        workers_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock lock(queue_mutex_);
                        cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                        if (stop_ && tasks_.empty())
                            return;
                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }
                    // 执行任务，并保证异常安全
                    try {
                        task();
                    } catch (...) {
                        // 异常情况下仍需递减计数器，否则 wait() 会永久阻塞
                    }
                    // 任务完成，递减待处理计数器
                    {
                        std::lock_guard lock(wait_mutex_);
                        --outstanding_;
                        if (outstanding_ == 0)
                            wait_cv_.notify_all();
                    }
                }
            });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard lock(queue_mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable())
                t.join();
        }
    }

    // 禁止拷贝/移动
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // 提交任务
    void submit(std::function<void()> task) {
        {
            std::lock_guard lock(wait_mutex_);
            ++outstanding_;
        }
        {
            std::lock_guard lock(queue_mutex_);
            tasks_.push(std::move(task));
        }
        cv_.notify_one();
    }

    // 等待所有已提交任务完成
    void wait() {
        std::unique_lock lock(wait_mutex_);
        wait_cv_.wait(lock, [this] { return outstanding_ == 0; });
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;

    std::mutex queue_mutex_;
    std::condition_variable cv_;
    bool stop_;

    std::mutex wait_mutex_;
    std::condition_variable wait_cv_;
    int outstanding_;            // 由 wait_mutex_ 保护
};