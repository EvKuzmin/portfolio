#pragma once
#include <armadillo>
#include <map>
#include <variant>
#include <vector>
#include <string>
#include <stdexcept>
#include <iostream>
#include <set>

using MapVariant = std::map< std::string, std::variant<int, arma::uword , std::string, double, std::vector<double>, std::vector<std::string>, arma::mat, arma::sp_mat, bool > >;

arma::sp_mat get_compensator(const arma::sp_mat &A_ex, const arma::sp_mat &A_ap, 
                             const arma::field<arma::vec> &vectors, const arma::sp_mat &pattern,const MapVariant &parameters)
{
    bool debug_mode = std::get<bool>(parameters.at("debug_mode"));
    bool pattern_addition_mode = std::get<bool>(parameters.at("pattern_addition_mode"));

    arma::uword M = vectors.n_elem;
    arma::uword N = A_ex.n_rows;

    // Если паттерн не задан (пустые размеры), используем A_ap как паттерн
    const arma::sp_mat& pattern_used = (pattern.n_rows == 0 || pattern.n_cols == 0) ? A_ap : pattern;

    // Проверки при debug_mode = true
    if (debug_mode) {
        
        // Проверка: A_ex и A_ap должны иметь одинаковые размеры
        if (A_ex.n_rows != A_ap.n_rows || A_ex.n_cols != A_ap.n_cols) {
            throw std::invalid_argument("Ошибка: A_ex и A_ap должны иметь одинаковые размеры");
        }

        // Проверка: A_ex и pattern должны иметь одинаковые размеры
        if (A_ex.n_rows != pattern_used.n_rows || A_ex.n_cols != pattern_used.n_cols) {
            throw std::invalid_argument("Ошибка: A_ex и pattern должны иметь одинаковые размеры");
        }

        // Проверка: число столбцов A_ex равно числу элементов любого из векторов
        if (M > 0) {
            arma::uword vec_size = vectors(0).n_elem;
            
            if (A_ex.n_cols != vec_size) {
                throw std::invalid_argument("Ошибка: число столбцов A_ex (" + std::to_string(A_ex.n_cols) + 
                                          ") не равно размеру векторов (" + std::to_string(vec_size) + ")");
            }

            // Дополнительная проверка: все векторы должны иметь одинаковый размер
            for (int i = 1; i < M; ++i) {
                if (vectors(i).n_elem != vec_size) {
                    throw std::invalid_argument("Ошибка: все векторы должны иметь одинаковый размер");
                }
            }
        } else {
            throw std::invalid_argument("Ошибка: field векторов пусто (M = 0)");
        }

        std::cout << "[DEBUG] Все проверки пройдены успешно:\n"
                  << "  - A_ex размер: " << A_ex.n_rows << "x" << A_ex.n_cols << "\n"
                  << "  - A_ap размер: " << A_ap.n_rows << "x" << A_ap.n_cols << "\n"
              << "  - pattern размер: " << pattern_used.n_rows << "x" << pattern_used.n_cols << "\n"
                  << "  - Количество векторов: " << M << "\n"
                  << "  - Размер каждого вектора: " << vectors(0).n_elem << "\n";
    }

    
    //создать матрицу С из векторов vectors
    arma::mat C(N, M);
    for (arma::uword j = 0; j < M; ++j) {
        for (arma::uword i = 0; i < N; ++i) {
            C(i, j) = vectors(j)(i);
        }
    }
    arma::sp_mat A = A_ex - A_ap;
    arma::sp_mat A_T = A.t();
    arma::sp_mat pattern_T = pattern_used.t();
    arma::sp_mat com(N, N);
    for (arma::uword i = 0; i< N; i++){
        std::set<int> I;
        for (auto it = pattern_T.begin_col(i); it != pattern_T.end_col(i); ++it) {
            I.insert(it.row());
        }
        if (pattern_addition_mode) {
            for (auto it = A_T.begin_col(i); it != A_T.end_col(i); ++it) {
                if (I.size() >= M) break;
                    
                
                I.insert(it.row());

            }
        }
        if (I.size() < M) throw std::invalid_argument("Ошибка: система переопределена на строке " + std::to_string(i));
        // МНК arma::vec x = arma::solve(A, b, arma::solve_opts::fast);  // A: m×n, m>n
        arma::vec b = arma::vec(A_T.col(i)) * C; //проверить правильность операции(расписать)
        
        // Сформировать индексы из множества I
        arma::uvec idx(I.size());
        arma::uword k = 0;
        for (auto v : I) idx(k++) = v;

        // Подматрица C: все строки, столбцы из множества I, проверить правильность операции(расписать) (мб rows вместо cols)
        arma::mat C_submat = C.cols(idx);
        arma::vec x;
        if (I.size() == M){
        x = arma::solve(C_submat, b);  // A: n×n, b: n
        //решить слау с подматрицей матрицы С и полученной правой частью 
        
        

        }
        else{
            // Недоопределённая (минимум нормы)
            x = arma::pinv(C_submat) * b; //проверить правильность операции(расписать)
        }

        //добавить решение в матрицу компенсации
        for (arma::uword j = 0; j < I.size(); ++j) { //строго проверить
            com(idx(j), i) = x(j);
        }
        
    }
    
    // Проверка главного свойства при debug_mode:
    // Для всех столбцов C: com * C = A_ex * C - A_ap * C
    if (debug_mode) {
        // Допуск для сравнения свойства
        double prop_tol = 1e-10;
        if (auto it = parameters.find("property_tol"); it != parameters.end()) {
            try { prop_tol = std::get<double>(it->second); } catch (...) {}
        }

        // Разрешить безматричный режим для A_ex: используем предвычисленное A_ex(C)
        bool use_operator_A_ex = false;
        if (auto it = parameters.find("use_operator_A_ex"); it != parameters.end()) {
            try { use_operator_A_ex = std::get<bool>(it->second); } catch (...) {}
        }

        arma::mat leftM = arma::mat(com * C); // N x M
        arma::mat rightM;
        if (use_operator_A_ex) {
            bool ok = false;
            // Предпочтительно: arma::mat N x M
            if (auto it = parameters.find("A_ex_C_mat"); it != parameters.end()) {
                try {
                    const auto &MC = std::get<arma::mat>(it->second);
                    if (MC.n_rows == N && MC.n_cols == M) {
                        rightM = MC - arma::mat(A_ap * C);
                        ok = true;
                    }
                } catch (...) {}
            }
            // Альтернатива: плоский вектор длины N*M (по столбцам)
            if (!ok) {
                if (auto it2 = parameters.find("A_ex_C_flat"); it2 != parameters.end()) {
                    try {
                        const auto &vf = std::get<std::vector<double>>(it2->second);
                        if (vf.size() == static_cast<size_t>(N * M)) {
                            rightM.set_size(N, M);
                            // Заполняем по столбцам
                            size_t pos = 0;
                            for (arma::uword j = 0; j < M; ++j) {
                                for (arma::uword i = 0; i < N; ++i) {
                                    rightM(i, j) = vf[pos++];
                                }
                            }
                            rightM -= arma::mat(A_ap * C);
                            ok = true;
                        }
                    } catch (...) {}
                }
            }
            if (!ok) {
                throw std::invalid_argument("[DEBUG] use_operator_A_ex=true, но не найден параметр 'A_ex_C_mat' (arma::mat NxM) или 'A_ex_C_flat' (vector<double> длины N*M)");
            }
        } else {
            rightM = arma::mat(A_ex * C) - arma::mat(A_ap * C);
        }

        arma::mat R = rightM - leftM;
        double frob = arma::norm(R, "fro");
        if (frob > prop_tol) {
            std::cout << "[DEBUG] Нарушено главное свойство (все векторы): ||R||_F = "
                      << std::scientific << frob << ", tol = " << prop_tol << "\n";
            // Невязка по столбцам (для каждого входного вектора)
            for (arma::uword j = 0; j < M; ++j) {
                double cj = arma::norm(R.col(j));
                std::cout << "[DEBUG]  |r(v" << j << ")|_2 = " << std::scientific << cj << "\n";
            }
            // Невязка по строкам (в духе исходного требования)
            arma::vec row_res(N, arma::fill::zeros);
            for (arma::uword i = 0; i < N; ++i) row_res(i) = arma::norm(R.row(i));
            std::cout << "[DEBUG]  row-wise residuals^T = " << row_res.t();
        } else {
            std::cout << "[DEBUG] Главное свойство выполнено (все векторы): ||R||_F = "
                      << std::scientific << frob << " <= tol = " << prop_tol << "\n";
        }
    }

    return com;
}

arma::sp_mat diag_compensator(const arma::sp_mat &A_ex, const arma::sp_mat &A_ap, const MapVariant &parameters)
{
    arma::uword N = A_ex.n_rows;

    // Диагональный паттерн (единицы на диагонали)
    arma::sp_mat pattern(N, N);
    pattern.diag().ones();

    // Вектор единиц в field из одного столбца
    arma::field<arma::vec> vectors(1);
    vectors(0) = arma::vec(N, arma::fill::ones);

    return get_compensator(A_ex, A_ap, vectors, pattern, parameters);

}