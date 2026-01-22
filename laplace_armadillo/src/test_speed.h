#pragma once

#include "solvers.h"
#include "preconditioners.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <functional>
#include <map>

// Вспомогательные функции для получения решателя и предобуславливателя

std::function<arma::vec(const arma::sp_mat&, const arma::vec&, arma::vec, int&)> 
get_solver(const std::string& solver_name) {
    if (solver_name == "gmres") {
        return [](const arma::sp_mat& A, const arma::vec& b, arma::vec x, int& iters) {
            return solvers::gmres_solver(A, b, std::move(x), iters, 30, 300, 1e-6);
        };
    } else if (solver_name == "bicgstab") {
        return [](const arma::sp_mat& A, const arma::vec& b, arma::vec x, int& iters) {
            return solvers::bicgstab_solver(A, b, std::move(x), iters, 300, 1e-6);
        };
    } else if (solver_name == "cg") {
        return [](const arma::sp_mat& A, const arma::vec& b, arma::vec x, int& iters) {
            return solvers::cg_solver(A, b, std::move(x), iters, 300, 1e-6);
        };
    } else if (solver_name == "cr") {
        return [](const arma::sp_mat& A, const arma::vec& b, arma::vec x, int& iters) {
            return solvers::cr_solver(A, b, std::move(x), iters, 300, 1e-6);
        };
    }
    throw std::runtime_error("Unknown solver: " + solver_name);
}

solvers::Precond get_preconditioner(const std::string& precond_name, const arma::sp_mat& A) {
    if (precond_name == "identity") {
        return solvers::identity_precond();
    } else if (precond_name == "diagonal") {
        return diagonal_precond(A);
    } else if (precond_name == "damped_jacobi") {
        return damped_jacobi(A, 0.7);
    } else if (precond_name == "polynomial_jacobi") {
        return polynomial_jacobi(A, 2);
    } else if (precond_name == "gauss_seidel") {
        return gauss_seidel(A);
    } else if (precond_name == "ilu0") {
        return ilu0(A);
    } else if (precond_name == "ssor") {
        return ssor(A, 1.0);
    }
    throw std::runtime_error("Unknown preconditioner: " + precond_name);
}

// Вывод результатов в логи
void log_test_result(int n, const std::string& test_type, const std::string& solver_name, 
                     const std::string& precond_name, double time_ms, int iterations, 
                     double residual_norm) {
    std::cout << std::fixed << std::setprecision(6)
              << "[" << test_type << "] n=" << std::setw(5) << n 
              << " | solver=" << std::setw(12) << solver_name 
              << " | precond=" << std::setw(15) << precond_name
              << " | time=" << std::setw(10) << time_ms << " ms"
              << " | iters=" << std::setw(3) << iterations
              << " | residual=" << std::scientific << std::setprecision(3) << residual_norm
              << std::endl;
}

// Функция для решения системы и измерения времени
bool solve_and_measure(const arma::sp_mat& A, const arma::vec& b, const arma::vec& x_init,
                       const std::string& solver_name, const std::string& precond_name,
                       int n, const std::string& test_type) {
    try {
        auto solver_func = get_solver(solver_name);
        auto precond = get_preconditioner(precond_name, A);

        auto start = std::chrono::high_resolution_clock::now();
        
        int iterations = 0;
        arma::vec x;
        if (solver_name == "gmres") {
            x = solvers::gmres_solver(A, b, x_init, iterations, 30, 300, 1e-6, precond);
        } else if (solver_name == "bicgstab") {
            x = solvers::bicgstab_solver(A, b, x_init, iterations, 300, 1e-6, precond);
        } else if (solver_name == "cg") {
            x = solvers::cg_solver(A, b, x_init, iterations, 300, 1e-6, precond);
        } else if (solver_name == "cr") {
            x = solvers::cr_solver(A, b, x_init, iterations, 300, 1e-6, precond);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        arma::vec residual = b - A * x;
        double residual_norm = arma::norm(residual);

        log_test_result(n, test_type, solver_name, precond_name, time_ms, iterations, residual_norm);
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error in solve_and_measure: " << e.what() << std::endl;
        return false;
    }
}

// int test_diagonal() - семидиагональные матрицы с диагональным преобладанием
int test_diagonal(const std::string& solver_name = "gmres", 
                  const std::string& precond_name = "diagonal") {
    std::vector<int> sizes = {10, 100, 1000, 10000};

    std::cout << "\n=== TEST_DIAGONAL: Семидиагональные матрицы ===" << std::endl;

    for (int n : sizes) {
        // Создание семидиагональной матрицы (диагональ, +/-1, +/-2)
        std::vector<unsigned long long> mat_i, mat_j;
        std::vector<double> values;

        for (int i = 0; i < n; ++i) {
            // Диагональный элемент (преобладание)
            mat_i.push_back(i);
            mat_j.push_back(i);
            values.push_back(6.0);

            // Элемент на диагонали-1 (верхняя)
            if (i > 0) {
                mat_i.push_back(i);
                mat_j.push_back(i - 1);
                values.push_back(-1.0);
            }

            // Элемент на диагонали+1 (нижняя)
            if (i < n - 1) {
                mat_i.push_back(i);
                mat_j.push_back(i + 1);
                values.push_back(-1.0);
            }

            // Элемент на диагонали-2 (верхняя)
            if (i > 1) {
                mat_i.push_back(i);
                mat_j.push_back(i - 2);
                values.push_back(-0.5);
            }

            // Элемент на диагонали+2 (нижняя)
            if (i < n - 2) {
                mat_i.push_back(i);
                mat_j.push_back(i + 2);
                values.push_back(-0.5);
            }
        }

        arma::umat locations(2, mat_i.size());
        for (int i = 0; i < static_cast<int>(mat_i.size()); ++i) {
            locations(0, i) = mat_i[i];
            locations(1, i) = mat_j[i];
        }
        arma::vec values_arma(values);
        arma::sp_mat A(locations, values_arma, n, n);
        mat_i.clear(); mat_j.clear(); values.clear();

        // Создание правой части и начального приближения
        arma::vec b = arma::randu<arma::vec>(n);
        arma::vec x_init = arma::zeros<arma::vec>(n);

        // Решение системы
        solve_and_measure(A, b, x_init, solver_name, precond_name, n, "DIAGONAL");
    }

    return 0;
}

// int test_symmetric() - симметричные разреженные матрицы
int test_symmetric(const std::string& solver_name = "cg", 
                   const std::string& precond_name = "diagonal") {
    std::vector<int> sizes = {10, 100, 1000, 10000};

    std::cout << "\n=== TEST_SYMMETRIC: Симметричные разреженные матрицы ===" << std::endl;

    for (int n : sizes) {
        std::vector<unsigned long long> mat_i, mat_j;
        std::vector<double> values;

        arma::arma_rng::set_seed(42); // Для воспроизводимости

        for (int i = 0; i < n; ++i) {
            // Диагональный элемент
            mat_i.push_back(i);
            mat_j.push_back(i);
            values.push_back(static_cast<double>(n) + 1.0);

            // Добавляем симметричные элементы вне диагонали
            for (int j = i + 1; j < std::min(i + 4, n); ++j) {
                double val = arma::randu() - 0.5; // [-0.5, 0.5]
                
                // Верхняя треугольная часть
                mat_i.push_back(i);
                mat_j.push_back(j);
                values.push_back(val);

                // Нижняя треугольная часть (для симметрии)
                mat_i.push_back(j);
                mat_j.push_back(i);
                values.push_back(val);
            }
        }

        arma::umat locations(2, mat_i.size());
        for (int i = 0; i < static_cast<int>(mat_i.size()); ++i) {
            locations(0, i) = mat_i[i];
            locations(1, i) = mat_j[i];
        }
        arma::vec values_arma(values);
        arma::sp_mat A(locations, values_arma, n, n);
        mat_i.clear(); mat_j.clear(); values.clear();

        // Создание правой части и начального приближения
        arma::vec b = arma::randu<arma::vec>(n);
        arma::vec x_init = arma::zeros<arma::vec>(n);

        // Решение системы
        solve_and_measure(A, b, x_init, solver_name, precond_name, n, "SYMMETRIC");
    }

    return 0;
}

// int test_spd() - симметричная положительно определённая матрица
int test_spd(const std::string& solver_name = "cg", 
             const std::string& precond_name = "diagonal") {
    std::vector<int> sizes = {10, 100, 1000, 10000};

    std::cout << "\n=== TEST_SPD: Симметричные положительно определённые матрицы ===" << std::endl;

    for (int n : sizes) {
        std::vector<unsigned long long> mat_i, mat_j;
        std::vector<double> values;

        arma::arma_rng::set_seed(42);

        for (int i = 0; i < n; ++i) {
            // Диагональный элемент (больший, чтобы гарантировать SPD)
            mat_i.push_back(i);
            mat_j.push_back(i);
            values.push_back(static_cast<double>(n) + 2.0);

            // Добавляем симметричные элементы вне диагонали (малые)
            for (int j = i + 1; j < std::min(i + 4, n); ++j) {
                double val = (arma::randu() - 0.5) * 0.1; // [-0.05, 0.05]
                
                // Верхняя треугольная часть
                mat_i.push_back(i);
                mat_j.push_back(j);
                values.push_back(val);

                // Нижняя треугольная часть (для симметрии)
                mat_i.push_back(j);
                mat_j.push_back(i);
                values.push_back(val);
            }
        }

        arma::umat locations(2, mat_i.size());
        for (int i = 0; i < static_cast<int>(mat_i.size()); ++i) {
            locations(0, i) = mat_i[i];
            locations(1, i) = mat_j[i];
        }
        arma::vec values_arma(values);
        arma::sp_mat A(locations, values_arma, n, n);
        mat_i.clear(); mat_j.clear(); values.clear();

        // Создание правой части и начального приближения
        arma::vec b = arma::randu<arma::vec>(n);
        arma::vec x_init = arma::zeros<arma::vec>(n);

        // Решение системы
        solve_and_measure(A, b, x_init, solver_name, precond_name, n, "SPD");
    }

    return 0;
}

// int test_other() - разреженная матрица, не соответствующая предыдущим условиям
int test_other(const std::string& solver_name = "gmres", 
               const std::string& precond_name = "diagonal") {
    std::vector<int> sizes = {10, 100, 1000, 10000};

    std::cout << "\n=== TEST_OTHER: Разреженные несимметричные матрицы ===" << std::endl;

    for (int n : sizes) {
        std::vector<unsigned long long> mat_i, mat_j;
        std::vector<double> values;

        arma::arma_rng::set_seed(42);

        for (int i = 0; i < n; ++i) {
            // Диагональный элемент
            mat_i.push_back(i);
            mat_j.push_back(i);
            values.push_back(static_cast<double>(n) + 1.0);

            // Несимметричные элементы вне диагонали
            for (int j = 0; j < n; ++j) {
                if (i != j && std::abs(i - j) <= 3) {
                    double val = arma::randu() - 0.5; // Случайные значения
                    mat_i.push_back(i);
                    mat_j.push_back(j);
                    values.push_back(val);
                }
            }
        }

        arma::umat locations(2, mat_i.size());
        for (int i = 0; i < static_cast<int>(mat_i.size()); ++i) {
            locations(0, i) = mat_i[i];
            locations(1, i) = mat_j[i];
        }
        arma::vec values_arma(values);
        arma::sp_mat A(locations, values_arma, n, n);
        mat_i.clear(); mat_j.clear(); values.clear();

        // Создание правой части и начального приближения
        arma::vec b = arma::randu<arma::vec>(n);
        arma::vec x_init = arma::zeros<arma::vec>(n);

        // Решение системы
        solve_and_measure(A, b, x_init, solver_name, precond_name, n, "OTHER");
    }

    return 0;
}