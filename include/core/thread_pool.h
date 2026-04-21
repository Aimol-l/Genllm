#pragma once
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>
#include <vector>

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads) : stop_(false) {
        workers_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this] { this->run(); });
        }
    }

    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& w : workers_) w.join();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // 提交一个任务
    void submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push(std::move(task));
            ++pending_;
        }
        cv_.notify_one();
    }

    // 阻塞等待所有已提交的任务执行完毕
    void wait() {
        std::unique_lock<std::mutex> lock(wait_mutex_);
        done_cv_.wait(lock, [this] { return pending_.load(std::memory_order_acquire) == 0; });
    }

private:
    void run() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (stop_ && tasks_.empty()) return;
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
            if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                done_cv_.notify_one();
            }
        }
    }

    std::vector<std::jthread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;

    std::atomic<int> pending_{0};
    std::mutex wait_mutex_;
    std::condition_variable done_cv_;

    bool stop_;
};
