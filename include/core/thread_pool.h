#pragma once
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>
#include <vector>

class ThreadPool {
private:
    void run();

    bool stop_;
    std::vector<std::jthread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<int> pending_{0};
    std::mutex wait_mutex_;
    std::condition_variable done_cv_;

public:
    ~ThreadPool();
    explicit ThreadPool(size_t num_threads);
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    void submit(std::function<void()> task);
    void wait();
};
