#pragma once

#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>


class bfloat16_t {

private:
    uint16_t bits_;

    struct from_bits_t {};
    constexpr bfloat16_t(uint16_t b, from_bits_t) : bits_(b) {}

     // Round to nearest even and truncate to bfloat16
    static constexpr uint16_t float_to_bf16_bits(float f) {
        uint32_t bits = std::bit_cast<uint32_t>(f);
        uint32_t sign = (bits >> 31) & 0x1;
        uint32_t exp  = (bits >> 23) & 0xFF;
        uint32_t mant = bits & 0x7FFFFF;
        uint32_t mant_lo = bits & 0x7FFF;
        uint32_t round_bit = (mant_lo >> 14) & 1;
        uint32_t sticky_bits = mant_lo & 0x3FFF;
        uint32_t round = (round_bit && (sticky_bits || (mant & 1))) ? 1 : 0;
        uint32_t combined = ((exp << 7) | (mant >> 16)) + round;
        if (combined >= 0x100) {
            combined = 0;
            exp = 0xFF;
        }
        return (sign << 15) | static_cast<uint16_t>(combined);
    }
    static constexpr float bf16_bits_to_float(uint16_t bits) {
        uint32_t combined = (static_cast<uint32_t>(bits) << 16);
        return std::bit_cast<float>(combined); 
    }

public:
    constexpr bfloat16_t() : bits_(0) {}
    constexpr bfloat16_t(float f) : bits_(float_to_bf16_bits(f)) {} 
    constexpr bfloat16_t(double d) : bfloat16_t(static_cast<float>(d)) {}
    constexpr bfloat16_t(int8_t d) : bfloat16_t(static_cast<float>(d)) {}
    constexpr bfloat16_t(int16_t d) : bfloat16_t(static_cast<float>(d)) {}
    constexpr bfloat16_t(int32_t d) : bfloat16_t(static_cast<float>(d)) {}
    constexpr bfloat16_t(int64_t d) : bfloat16_t(static_cast<float>(d)) {}
    static constexpr bfloat16_t from_bits(uint16_t b) {
        return bfloat16_t{b, from_bits_t{}};
    }

private:
    constexpr bfloat16_t(uint16_t b, bool) : bits_(b) {}
public:

    constexpr operator float() const {
        return bf16_bits_to_float(bits_);
    }

    explicit operator double() const {
        return static_cast<float>(*this);
    }

    constexpr uint16_t bits() const { return bits_; }

    bfloat16_t operator-() const {
        return from_bits(bits_ ^ 0x8000);
    }

    bfloat16_t& operator+=(const bfloat16_t& rhs) {
        *this = static_cast<float>(*this) + static_cast<float>(rhs);
        return *this;
    }
    bfloat16_t& operator-=(const bfloat16_t& rhs) {
        *this = static_cast<float>(*this) - static_cast<float>(rhs);
        return *this;
    }
    bfloat16_t& operator*=(const bfloat16_t& rhs) {
        *this = static_cast<float>(*this) * static_cast<float>(rhs);
        return *this;
    }
    bfloat16_t& operator/=(const bfloat16_t& rhs) {
        *this = static_cast<float>(*this) / static_cast<float>(rhs);
        return *this;
    }

    bfloat16_t& operator=(float f) {
        bits_ = float_to_bf16_bits(f);
        return *this;
    }

    bool operator==(const bfloat16_t& rhs) const {
        float a = static_cast<float>(*this);
        float b = static_cast<float>(rhs);
        if (std::isnan(a) || std::isnan(b)) return false;
        return a == b;
    }
    bool operator!=(const bfloat16_t& rhs) const { return !(*this == rhs); }
    bool operator<(const bfloat16_t& rhs) const {
        if (std::isnan(*this) || std::isnan(rhs)) return false;
        return static_cast<float>(*this) < static_cast<float>(rhs);
    }
    bool operator<=(const bfloat16_t& rhs) const { return (*this < rhs) || (*this == rhs); }
    bool operator>(const bfloat16_t& rhs) const { return !(*this <= rhs); }
    bool operator>=(const bfloat16_t& rhs) const { return !(*this < rhs); }

    static constexpr bfloat16_t zero()      { return from_bits(0x0000); }
    static constexpr bfloat16_t neg_zero()  { return from_bits(0x8000); }
    static constexpr bfloat16_t infinity()  { return from_bits(0x7F80); }
    static constexpr bfloat16_t neg_infinity(){ return from_bits(0xFF80); }
    static constexpr bfloat16_t nan()       { return from_bits(0x7FC0); }

    bool is_nan() const    { return (bits_ & 0x7FFF) > 0x7F80; }
    bool is_inf() const    { return (bits_ & 0x7FFF) == 0x7F80; }
    bool is_finite() const { return !is_nan() && !is_inf(); }
    int sign() const       { return (bits_ & 0x8000) ? -1 : 1; }
};


// Non-member operators
inline bfloat16_t operator+(bfloat16_t a, bfloat16_t b) { a += b; return a; }
inline bfloat16_t operator-(bfloat16_t a, bfloat16_t b) { a -= b; return a; }
inline bfloat16_t operator*(bfloat16_t a, bfloat16_t b) { a *= b; return a; }
inline bfloat16_t operator/(bfloat16_t a, bfloat16_t b) { a /= b; return a; }

inline std::ostream& operator<<(std::ostream& os, const bfloat16_t& bf) {
    float f = static_cast<float>(bf);
    if (bf.is_nan()) {
        return os << "bfloat16(NaN)";
    } else if (bf.is_inf()) {
        return os << "bfloat16(" << (bf.sign() < 0 ? "-" : "") << "inf)";
    } else {
        return os << "bfloat16(" << f << ")";
    }
}

#if __cplusplus >= 202302L // c++23
    namespace std {
        template<>
        class numeric_limits<bfloat16_t> {
        public:
            static constexpr bool is_specialized = true;
            static constexpr bool is_signed = true;
            static constexpr bool is_integer = false;
            static constexpr bool is_exact = false;
            static constexpr bool has_infinity = true;
            static constexpr bool has_quiet_NaN = true;
            static constexpr bool has_signaling_NaN = true;
            static constexpr float_denorm_style has_denorm = denorm_present;
            static constexpr bool has_denorm_loss = false;
            static constexpr float_round_style round_style = round_to_nearest;
            static constexpr int digits = 8;
            static constexpr int digits10 = 2;
            static constexpr int max_digits10 = 4;
            static constexpr int radix = 2;
            static constexpr int min_exponent = -125;
            static constexpr int min_exponent10 = -37;
            static constexpr int max_exponent = 128;
            static constexpr int max_exponent10 = 38;

            static constexpr bfloat16_t min()        { return bfloat16_t::from_bits(0x0080); }
            static constexpr bfloat16_t lowest()     { return bfloat16_t::from_bits(0xFF7F); }
            static constexpr bfloat16_t max()        { return bfloat16_t::from_bits(0x7F7F); }
            static constexpr bfloat16_t infinity()   { return bfloat16_t::infinity(); }
            static constexpr bfloat16_t quiet_NaN()  { return bfloat16_t::nan(); }
            static constexpr bfloat16_t epsilon()    { return bfloat16_t::from_bits(0x3C00); }
            static constexpr bfloat16_t round_error(){ return bfloat16_t(0.5f); }
        };
    }
#endif
