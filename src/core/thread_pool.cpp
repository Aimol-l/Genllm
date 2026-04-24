#include "core/thread_pool.h"


ThreadPool::ThreadPool(size_t num_threads) {
    this->stop_ = false;
    this->workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        this->workers_.emplace_back([this] { this->run(); });
    }
}

ThreadPool::~ThreadPool(){
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : this->workers_) w.join();
}

void ThreadPool::submit(std::function<void()> task) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.push(std::move(task));
        ++pending_;
    }
    cv_.notify_one();
}

void ThreadPool::wait() {
    std::unique_lock<std::mutex> lock(wait_mutex_);
    done_cv_.wait(lock, [this] { return pending_.load(std::memory_order_acquire) == 0; });
}

void ThreadPool::run() {
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
        if (pending_.fetch_sub(1, std::memory_order_acq_rel) <= 1) {
            done_cv_.notify_one();
        }
    }
}
