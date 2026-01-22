#pragma once

#include "armadillo"
#include <functional>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>

// Структура для хранения данных блочного метода Якоби
struct BlockJacobiData {
    std::vector<arma::mat> diag_blocks;  // Обратные диагональные блоки матрицы
    std::vector<int> block_sizes;       // Размеры блоков
    int n_blocks;                       // Количество блоков
};

// Структура для хранения данных SSOR предобуславливателя
struct SSORData {
    arma::sp_mat A;         // Исходная матрица
    arma::vec diag;         // Диагональная часть
    arma::sp_mat L;         // Нижняя треугольная часть
    arma::sp_mat U;         // Верхняя треугольная часть
    arma::sp_mat lower_op;  // D + omega * L для прямого хода
    arma::sp_mat upper_op;  // D + omega * U для обратного хода
    double omega;           // Параметр релаксации
};

// Блочный Якоби
inline std::function<arma::vec(const arma::vec &)> block_jacobi(const arma::sp_mat &A, const std::vector<int> &block_sizes) {
    int total_size = 0;
    for (int size : block_sizes) total_size += size;
    if (total_size != static_cast<int>(A.n_rows)) {
        throw std::runtime_error("Sum of block sizes does not match matrix size");
    }

    auto data = std::make_shared<BlockJacobiData>();
    data->block_sizes = block_sizes;
    data->n_blocks = static_cast<int>(block_sizes.size());

    int row_start = 0;
    for (int block_idx = 0; block_idx < data->n_blocks; ++block_idx) {
        int block_size = block_sizes[block_idx];
        int row_end = row_start + block_size;

        arma::sp_mat A_block = A.submat(arma::span(row_start, row_end - 1), arma::span(row_start, row_end - 1));
        arma::mat A_block_dense(A_block);
        arma::mat A_block_inv = arma::inv(A_block_dense);

        data->diag_blocks.push_back(A_block_inv);
        row_start = row_end;
    }

    return [data](const arma::vec &v) -> arma::vec {
        arma::vec result(v.n_elem, arma::fill::zeros);
        int row_start_local = 0;

        for (int block_idx = 0; block_idx < data->n_blocks; ++block_idx) {
            int block_size = data->block_sizes[block_idx];
            int row_end_local = row_start_local + block_size;

            arma::vec v_block = v.subvec(row_start_local, row_end_local - 1);
            arma::vec result_block = data->diag_blocks[block_idx] * v_block;
            result.subvec(row_start_local, row_end_local - 1) = result_block;

            row_start_local = row_end_local;
        }

        return result;
    };
}

// SSOR
inline std::function<arma::vec(const arma::vec &)> ssor(const arma::sp_mat &A, double omega = 1.0) {
    auto data = std::make_shared<SSORData>();
    data->A = A;
    data->omega = omega;
    const double eps = 1e-12;  // регуляризация диагонали

    data->diag.set_size(A.n_rows);
    data->L = arma::sp_mat(A.n_rows, A.n_cols);
    data->U = arma::sp_mat(A.n_rows, A.n_cols);

    arma::sp_mat::const_iterator it = A.begin();
    arma::sp_mat::const_iterator it_end = A.end();
    double val;
    for (; it != it_end; ++it) {
        val = *it;
        int i = static_cast<int>(it.row());
        int j = static_cast<int>(it.col());

        if (i == j) {
            double d = val;
            if (std::abs(d) < eps) d = (d >= 0.0) ? eps : -eps;  // защита от нуля
            data->diag(i) = d;
        } else if (i > j) {
            data->L(i, j) = val;
        } else {
            data->U(i, j) = val;
        }
    }

    // Строим разреженные операторы для треугольных систем
    data->lower_op = arma::sp_mat(A.n_rows, A.n_cols);
    data->upper_op = arma::sp_mat(A.n_rows, A.n_cols);
    data->lower_op.diag() = data->diag;
    data->upper_op.diag() = data->diag;
    data->lower_op += data->omega * data->L;
    data->upper_op += data->omega * data->U;

    return [data](const arma::vec &v) -> arma::vec {
        // Пытаемся использовать ускоренные разреженные треугольные решения
        arma::vec w;
        if (!arma::spsolve(w, data->lower_op, v, "superlu")) {
            // Фоллбек на ручной проход, если spsolve не доступен или неудачен
            w = v;
            for (int i = 0; i < static_cast<int>(data->A.n_rows); ++i) {
                double sum = 0.0;
                for (int j = 0; j < i; ++j) sum += data->omega * data->L(i, j) * w(j);
                w(i) = (v(i) - sum) / data->diag(i);
            }
        }

        arma::vec z = w / data->diag;  // масштабирование, как раньше

        arma::vec result;
        if (!arma::spsolve(result, data->upper_op, z, "superlu")) {
            result = z;
            for (int i = static_cast<int>(data->A.n_rows) - 1; i >= 0; --i) {
                double sum = 0.0;
                for (int j = i + 1; j < static_cast<int>(data->A.n_cols); ++j) sum += data->omega * data->U(i, j) * result(j);
                result(i) = (z(i) - sum) / data->diag(i);
            }
        }

        return result;
    };
}

// Диагональный (Якоби)
inline std::function<arma::vec(const arma::vec &)> diagonal_precond(const arma::sp_mat &A) {
    arma::vec diag = arma::vec(A.diag());
    return [diag](const arma::vec &v) -> arma::vec { return v / diag; };
}

// Затупленный Якоби
inline std::function<arma::vec(const arma::vec &)> damped_jacobi(const arma::sp_mat &A, double omega = 0.7) {
    arma::vec diag = arma::vec(A.diag());
    return [diag, omega](const arma::vec &v) -> arma::vec { return (1.0 / omega) * (v / diag); };
}

// Полиномиальный Якоби (k применений)
inline std::function<arma::vec(const arma::vec &)> polynomial_jacobi(const arma::sp_mat &A, int k = 2) {
    arma::vec diag = arma::vec(A.diag());
    arma::mat D_diag = arma::diagmat(1.0 / diag);
    arma::sp_mat D_inv = arma::sp_mat(D_diag);

    arma::sp_mat N = arma::speye(A.n_rows, A.n_cols) - D_inv * A;

    return [D_inv, N, k](const arma::vec &v) -> arma::vec {
        arma::vec result = D_inv * v;
        arma::vec tmp = result;
        for (int i = 1; i < k; ++i) {
            tmp = N * tmp + D_inv * v;
            result += tmp;
        }
        return result;
    };
}

// Гаусс-Зейдель (нижний треугольник с диагональю)
inline std::function<arma::vec(const arma::vec &)> gauss_seidel(const arma::sp_mat &A) {
    auto data = std::make_shared<SSORData>();
    data->A = A;
    data->diag.set_size(A.n_rows);
    data->L = arma::sp_mat(A.n_rows, A.n_cols);
    data->U = arma::sp_mat(A.n_rows, A.n_cols);

    arma::sp_mat::const_iterator it = A.begin();
    arma::sp_mat::const_iterator it_end = A.end();

    for (; it != it_end; ++it) {
        double val = *it;
        int i = static_cast<int>(it.row());
        int j = static_cast<int>(it.col());

        if (i == j) {
            data->diag(i) = val;
        } else if (i > j) {
            data->L(i, j) = val;
        } else {
            data->U(i, j) = val;
        }
    }

    return [data](const arma::vec &v) -> arma::vec {
        arma::vec result = v;
        for (int i = 0; i < static_cast<int>(data->A.n_rows); ++i) {
            double sum = 0.0;
            for (int j = 0; j < i; ++j) sum += data->L(i, j) * result(j);
            result(i) = (v(i) - sum) / data->diag(i);
        }
        return result;
    };
}

// ILU(0)
inline std::function<arma::vec(const arma::vec &)> ilu0(const arma::sp_mat &A) {
    auto data = std::make_shared<arma::sp_mat>(A);

    for (int i = 0; i < static_cast<int>(data->n_rows); ++i) {
        for (int k = 0; k < i; ++k) {
            if ((*data)(i, k) != 0.0) {
                double factor = (*data)(i, k) / (*data)(k, k);
                (*data)(i, k) = factor;

                for (int j = k + 1; j < static_cast<int>(data->n_cols); ++j) {
                    if ((*data)(i, j) != 0.0 || (*data)(k, j) != 0.0) {
                        (*data)(i, j) = (*data)(i, j) - factor * (*data)(k, j);
                    }
                }
            }
        }
    }

    return [data](const arma::vec &v) -> arma::vec {
        arma::vec y = v;
        for (int i = 0; i < static_cast<int>(data->n_rows); ++i) {
            for (int j = 0; j < i; ++j) y(i) = y(i) - (*data)(i, j) * y(j);
        }

        arma::vec x = y;
        for (int i = static_cast<int>(data->n_rows) - 1; i >= 0; --i) {
            for (int j = i + 1; j < static_cast<int>(data->n_cols); ++j) x(i) = x(i) - (*data)(i, j) * x(j);
            x(i) = x(i) / (*data)(i, i);
        }
        return x;
    };
}

// IC(0)
inline std::function<arma::vec(const arma::vec &)> ic0(const arma::sp_mat &A) {
    auto data = std::make_shared<arma::sp_mat>(A);

    for (int i = 0; i < static_cast<int>(data->n_rows); ++i) {
        double sum = 0.0;
        for (int k = 0; k < i; ++k) {
            if ((*data)(i, k) != 0.0) sum += (*data)(i, k) * (*data)(i, k);
        }
        (*data)(i, i) = std::sqrt(std::max((*data)(i, i) - sum, 1e-14));

        for (int j = i + 1; j < static_cast<int>(data->n_cols); ++j) {
            if ((*data)(j, i) != 0.0) {
                double s = (*data)(j, i);
                for (int k = 0; k < i; ++k) {
                    if ((*data)(i, k) != 0.0 && (*data)(j, k) != 0.0) {
                        s = s - (*data)(i, k) * (*data)(j, k);
                    }
                }
                (*data)(j, i) = s / (*data)(i, i);
            }
        }
    }

    return [data](const arma::vec &v) -> arma::vec {
        arma::vec y = v;
        for (int i = 0; i < static_cast<int>(data->n_rows); ++i) {
            for (int j = 0; j < i; ++j) y(i) = y(i) - (*data)(i, j) * y(j);
            y(i) = y(i) / (*data)(i, i);
        }

        arma::vec x = y;
        for (int i = static_cast<int>(data->n_rows) - 1; i >= 0; --i) {
            for (int j = i + 1; j < static_cast<int>(data->n_cols); ++j) x(i) = x(i) - (*data)(j, i) * x(j);
            x(i) = x(i) / (*data)(i, i);
        }
        return x;
    };
}


