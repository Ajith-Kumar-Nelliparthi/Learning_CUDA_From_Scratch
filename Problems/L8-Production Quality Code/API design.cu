#pragma once
#include <vector>
#include <string>

namespace arithmos {
    template <typename T>
    class Vector {
        public:
            Vector(size_t size);
            Vector(const std::vector<T>& hostData);

            size_t size() const;
            void upload(const std::vector<T>& hostData);
            std::vector<T> download() const;

            T dot(const Vector<T>& other) const;
            Vector<T> add(const Vector<T>& other) const;
            Vector<T> sub(const Vector<T>& other) const;
            Vector<T> mul(const Vector<T>& other) const;
            Vector<T> div(const Vector<T>& other) const;

            // Reductions
            T sum() const;
            T min() const;
            T max() const;

            // Transformations
            Vector<T> normalize() const;
            Vector<T> pow(T exponent) const;

            // Custom operator
            template <typename Op>
            Vector<T> apply(Op op) const;
    };
}