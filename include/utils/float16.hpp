#pragma once

#include <bit>
#include <cstdint>
#include <iostream>
#include <limits>

class float16_t {
private:
    uint16_t bits_;

    struct from_bits_t {};
    constexpr float16_t(uint16_t b, from_bits_t) : bits_(b) {}

    // Convert float32 -> float16 (with rounding)
    static constexpr uint16_t float_to_f16_bits(float f) {
        uint32_t f32_bits = std::bit_cast<uint32_t>(f);  // ✅ constexpr
        uint32_t sign = (f32_bits >> 31) & 0x1;
        uint32_t exp  = (f32_bits >> 23) & 0xFF;
        uint32_t mant = f32_bits & 0x7FFFFF;

        int32_t exp_adj = static_cast<int32_t>(exp) - 127 + 15;
        if (exp_adj >= 31) {
            return (sign << 15) | 0x7C00;  // +∞ / -∞
        } else if (exp_adj <= 0) {
            // Subnormal or zero
            uint32_t mant_with_hidden = (exp != 0) ? (mant | 0x800000) : mant;
            int32_t shift = 1 - exp_adj;
            if (shift > 24) {
                return static_cast<uint16_t>(sign << 15); // underflow to zero
            }
            uint32_t mant_shifted = mant_with_hidden >> (shift + 13);
            uint32_t round_bit = (mant_with_hidden >> (shift + 12)) & 1;
            uint32_t sticky = mant_with_hidden & ((1u << (shift + 12)) - 1);
            uint32_t round = (round_bit && (sticky || (mant_shifted & 1))) ? 1 : 0;
            return (sign << 15) | static_cast<uint16_t>(mant_shifted + round);
        } else {
            // Normal
            uint32_t mant_trunc = mant >> 13;
            uint32_t round_bit = (mant >> 12) & 1;
            uint32_t sticky = mant & 0xFFF;
            uint32_t round = (round_bit && (sticky || (mant_trunc & 1))) ? 1 : 0;
            uint32_t combined_mant = mant_trunc + round;
            if (combined_mant >= 0x400) {
                combined_mant = 0;
                exp_adj++;
                if (exp_adj >= 31) return (sign << 15) | 0x7C00;
            }
            return (sign << 15) | (exp_adj << 10) | static_cast<uint16_t>(combined_mant);
        }
    }

    // Convert float16 -> float32
    static constexpr float f16_bits_to_float(uint16_t bits) {
        uint32_t sign = (bits >> 15) & 0x1;
        uint32_t exp  = (bits >> 10) & 0x1F;
        uint32_t mant = bits & 0x3FF;

        uint32_t f32_bits;
        if (exp == 0) {
            if (mant == 0) {
                f32_bits = sign << 31; // zero
            } else {
                // subnormal: normalize
                int shift = __builtin_clz(mant) - 21; // clz(10-bit mant) → normalize
                mant <<= shift;
                exp = 1 - shift;
                f32_bits = (sign << 31) | ((exp + 127 - 15) << 23) | ((mant & 0x3FF) << 13);
            }
        } else if (exp == 31) {
            f32_bits = (sign << 31) | 0x7F800000 | (mant ? 0x400000 : 0);
        } else {
            uint32_t exp32 = exp + (127 - 15);
            f32_bits = (sign << 31) | (exp32 << 23) | (mant << 13);
        }
        return std::bit_cast<float>(f32_bits);   // ✅ constexpr
    }

public:
    constexpr float16_t() : bits_(0) {}
    constexpr float16_t(float f) : bits_(float_to_f16_bits(f)) {}  // ✅ 现在是 constexpr
    constexpr float16_t(double d) : float16_t(static_cast<float>(d)) {}
    constexpr float16_t(int8_t d) : float16_t(static_cast<float>(d)) {}
    constexpr float16_t(int16_t d) : float16_t(static_cast<float>(d)) {}
    constexpr float16_t(int32_t d) : float16_t(static_cast<float>(d)) {}
    constexpr float16_t(int64_t d) : float16_t(static_cast<float>(d)) {}

    static constexpr float16_t from_bits(uint16_t b) {
        return float16_t{b, from_bits_t{}};
    }

    // Conversion
    constexpr operator float() const { return f16_bits_to_float(bits_); }
    explicit constexpr operator double() const { return static_cast<float>(*this); }

    constexpr uint16_t bits() const { return bits_; }

    // Negation
    constexpr float16_t operator-() const { return from_bits(bits_ ^ 0x8000); }

    // Arithmetic (done via float)
    float16_t& operator+=(const float16_t& rhs) {
        *this = static_cast<float>(*this) + static_cast<float>(rhs);
        return *this;
    }
    float16_t& operator-=(const float16_t& rhs) {
        *this = static_cast<float>(*this) - static_cast<float>(rhs);
        return *this;
    }
    float16_t& operator*=(const float16_t& rhs) {
        *this = static_cast<float>(*this) * static_cast<float>(rhs);
        return *this;
    }
    float16_t& operator/=(const float16_t& rhs) {
        *this = static_cast<float>(*this) / static_cast<float>(rhs);
        return *this;
    }

    // Constants
    static constexpr float16_t zero()       { return from_bits(0x0000); }
    static constexpr float16_t neg_zero()   { return from_bits(0x8000); }
    static constexpr float16_t infinity()   { return from_bits(0x7C00); }
    static constexpr float16_t neg_infinity(){ return from_bits(0xFC00); }
    static constexpr float16_t nan()        { return from_bits(0x7E00); }

    // Classification
    constexpr bool is_nan() const    { return (bits_ & 0x7FFF) > 0x7C00; }
    constexpr bool is_inf() const    { return (bits_ & 0x7FFF) == 0x7C00; }
    constexpr bool is_finite() const { return !is_nan() && !is_inf(); }
    constexpr int sign() const       { return (bits_ & 0x8000) ? -1 : 1; }
};

// Operators
inline float16_t operator+(float16_t a, float16_t b) { a += b; return a; }
inline float16_t operator-(float16_t a, float16_t b) { a -= b; return a; }
inline float16_t operator*(float16_t a, float16_t b) { a *= b; return a; }
inline float16_t operator/(float16_t a, float16_t b) { a /= b; return a; }

// Stream output
inline std::ostream& operator<<(std::ostream& os, const float16_t& f16) {
    if (f16.is_nan()) {
        return os << "float16(NaN)";
    } else if (f16.is_inf()) {
        return os << "float16(" << (f16.sign() < 0 ? "-" : "") << "inf)";
    } else {
        return os << "float16(" << static_cast<float>(f16) << ")";
    }
}


#if __cplusplus >= 202302L // c++23
    namespace std {
    template<>
    class numeric_limits<float16_t> {
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
            static constexpr int digits = 11;
            static constexpr int digits10 = 3;
            static constexpr int max_digits10 = 5;
            static constexpr int radix = 2;
            static constexpr int min_exponent = -14;
            static constexpr int min_exponent10 = -4;
            static constexpr int max_exponent = 15;
            static constexpr int max_exponent10 = 4;

            static constexpr float16_t min()        { return float16_t::from_bits(0x0400); }
            static constexpr float16_t lowest()     { return float16_t::from_bits(0xFBFF); }
            static constexpr float16_t max()        { return float16_t::from_bits(0x7BFF); }
            static constexpr float16_t infinity()   { return float16_t::infinity(); }
            static constexpr float16_t quiet_NaN()  { return float16_t::nan(); }
            static constexpr float16_t epsilon()    { return float16_t::from_bits(0x1400); }
            static constexpr float16_t round_error(){ return float16_t(0.5f); } // ✅ OK
        };
    }
#endif
