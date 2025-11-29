#include <cmath>


template <typename T>
T dot_product(const int N, const T *w, const T *v){
    T result = 0;
    for(int i = 0; i < N; i++) {
        result += w[i] * v[i];
    }
    return result;
}

// computes new w and its norm
template <typename T>
T gemv_norm(const int N, const T beta, T *Ax, const T *y){
    T norm = 0;
    for(int i = 0; i < N; i++) {
        Ax[i] = Ax[i] + beta * y[i];
        norm += Ax[i] * Ax[i];
    }
    norm = sqrt(norm);
    return norm;
}

// normalize vector w to get new v
template <typename T>
void scale_vector(const int N, const T scalar, const T *x, T *y){
    for(int i = 0; i < N; i++) {
        y[i] = x[i] * scalar;
    }
}

template <typename T>
bool is_zero(T x) {
    if constexpr (std::is_floating_point_v<T>)
        return std::abs(x) < 1e-5;
    else
        return x == 0;
}

template <typename T>
void generate_unit_vector(const int N, T *unit_vector, int index) {
    for(int i = 0; i < N; i++) {
        unit_vector[i] = 0;
    }
    unit_vector[index] = 1;
}
