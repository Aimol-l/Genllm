#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <concepts>


namespace ops{


    // safe softmax for 1D vector, in-place
    template <typename T>   requires std::same_as<T, float> || std::same_as<T, double>
    void Softmax(std::vector<T>& vec,T temperature = 1.0) {
        if (vec.empty()) return;
        T max_val = *std::max_element(vec.begin(), vec.end());
        T sum = 0;

        T inv_temp = 1.0 / temperature;

        for (auto& val : vec) {
            val = std::exp((val - max_val) * inv_temp);
            sum += val;
        }
        std::transform(vec.begin(), vec.end(), vec.begin(), [sum](T val) { 
            T res = val / sum;
            return res > 0 ? res : static_cast<T>(1e-6); // 避免概率为0导致采样问题
        });
    }
}