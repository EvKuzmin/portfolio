#include<vector>
#include<string>
#include<fstream>
#include<regex>
#include<map>
#include<iostream>
#include<variant>
#include<filesystem>
#include<armadillo>
#include<any>
#include<iomanip>
#include<chrono>

#include"include/unordered_dense.h"

#include"solvers.h"
#include"preconditioners.h"
#include "test_speed.h"
#include"unsorted.hpp"


using namespace solvers;

int lap_mg() {
    int N1 = 4;
    int N2 = 4;
    int N3 = 4;

    ankerl::unordered_dense::map< std::tuple<int, int, int>, int> ijk_to_cell_id;

    for( int k = 0 ; k < N3 ; k++ ) {
        for( int j = 0 ; j < N2 ; j++ ) {
            for( int i = 0 ; i < N1 ; i++ ) {
                ijk_to_cell_id[std::make_tuple(i,j,k)] = i + j*N1 + k*N1*N2;
            }
        }
    }

    arma::sp_mat A;

    {

        std::vector<unsigned long long> mat_i, mat_j;
        std::vector<double> values;

        for( int k = 0 ; k < N3 ; k++ ) {
            for( int j = 0 ; j < N2 ; j++ ) {
                for( int i = 0 ; i < N1 ; i++ ) {
                    int c_id = i + j*N1 + k*N1*N2;
                    mat_i.push_back(c_id);
                    mat_j.push_back(c_id);
                    values.push_back(6);
                    std::array<std::tuple<int,int,int>, 6> o_id{
                        std::make_tuple(i-1,j,k),
                        std::make_tuple(i+1,j,k),
                        std::make_tuple(i,j-1,k),
                        std::make_tuple(i,j+1,k),
                        std::make_tuple(i,j,k-1),
                        std::make_tuple(i,j,k+1)
                    };
                    for(auto o : o_id) {
                        auto it = ijk_to_cell_id.find(o);
                        if( it != ijk_to_cell_id.end() ) {
                            int o_id = it->second;
                            mat_i.push_back(c_id);
                            mat_j.push_back(o_id);
                            values.push_back(-1);
                        }
                    }
                }
            }
        }

    
        // arma::umat locations1 = { { 1, 7, 9 },
                    //    { 2, 8, 9 } };
                    

        arma::umat locations1(2, mat_i.size());
        // locations1 = { 1, 7, 9, 2, 8, 9 };

        for(int i = 0; i < mat_i.size() ; i++) {
            locations1(0, i) = mat_i[i];
            locations1(1, i) = mat_j[i];
        }

        arma::vec values1{values};

        // {
        //     mat_i = std::vector<unsigned long long>{};
        //     mat_j = std::vector<unsigned long long>{};
        //     values = std::vector<double>{};
        // }
        mat_i.clear();
        mat_j.clear();
        values.clear();

        A = arma::sp_mat(locations1, values1, N1*N2*N3, N1*N2*N3);
    }

    arma::vec b(N1*N2*N3);

    for( int k = 0 ; k < N3 ; k++ ) {
        for( int j = 0 ; j < N2 ; j++ ) {
            for( int i = 0 ; i < N1 ; i++ ) {
                int c_id = i + j*N1 + k*N1*N2;
                std::array<std::tuple<int,int,int>, 6> o_id{
                    std::make_tuple(i-1,j,k),
                    std::make_tuple(i+1,j,k),
                    std::make_tuple(i,j-1,k),
                    std::make_tuple(i,j+1,k),
                    std::make_tuple(i,j,k-1),
                    std::make_tuple(i,j,k+1)
                };
                double vals[6] = { 1.0, 1.0, 2.0, 2.0, 3.0, 3.0 };
                for(int l = 0 ; l < 6 ; l++) {
                    if( !ijk_to_cell_id.contains(o_id[l]) ) {
                        b[c_id] = vals[l];
                    }
                }
            }
        }
    }

    // arma::mat A_dense(A);

    // arma::vec x = arma::solve( A_dense, b );

    arma::vec x(N1*N2*N3);

    x = gmres(A, b, x);


    // x.print();

    std::vector<int> coarse_elements = coarsening_MIS(A, 2);

    // Преобразуем std::vector в arma::uvec для использования с extract_sparse_block
    arma::uvec coarse_elements_b(N1*N2*N3, arma::fill::zeros);
    for( auto v : coarse_elements ) {
        coarse_elements_b[v] = 1;
    }

    arma::uvec fine_elements_b(N1*N2*N3, arma::fill::zeros);
    arma::uvec coarse_elements_prefix_sum(N1*N2*N3, arma::fill::zeros);
    arma::uvec fine_elements_prefix_sum(N1*N2*N3, arma::fill::zeros);
    int N_coarse = 0;
    int N_fine = 0;
    for(arma::uword i = 0 ; i < coarse_elements_b.n_elem ; i++) {
        if(coarse_elements_b[i] != 1) {
            fine_elements_b[i] = 1;
        }
        coarse_elements_prefix_sum[i] = N_coarse;
        fine_elements_prefix_sum[i] = N_fine;
        N_coarse += static_cast<int>(coarse_elements_b[i]);
        N_fine   += static_cast<int>(fine_elements_b[i]);
    }

    // Извлекаем все четыре блока матрицы за один вызов
    MatrixBlocks blocks = extract_sparse_block(A, fine_elements_b, coarse_elements_b, 
                                                fine_elements_prefix_sum, coarse_elements_prefix_sum, 
                                                N_fine, N_coarse);

    arma::sp_mat& Aff = blocks.Aff;
    arma::sp_mat& Afc = blocks.Afc;
    arma::sp_mat& Acf = blocks.Acf;
    arma::sp_mat& Acc = blocks.Acc;

    arma::sp_mat Aff_SPAI = SPAI(Aff);
    Aff_SPAI.print();

    (Aff_SPAI.t() * Afc).print();
    
    return 0;
}

// Структура для хранения результатов
struct SolverResult {
    std::string test_type;           // Тип матрицы (symmetric, spd, arbitrary)
    std::string precond_name;        // Название предобуславливателя
    std::string solver_name;         // Название решателя (gmres/bicgstab)
    int iterations;                  // Количество итераций
    double residual_norm;            // Норма финальной невязки
    double solution_error;           // Ошибка решения
    double time_ms;                  // Время выполнения в миллисекундах
    bool converged;                  // Сошлась ли система
};

void check_arm_ksp() {
    std::vector<SolverResult> results;
    std::ofstream report("ksp_report.txt");
    
    // Размер матриц для тестирования
    const int N = 50;
    
    report << "=== Iterative Solvers and Preconditioners Benchmark ===" << std::endl;
    report << "Matrix size: " << N << "x" << N << std::endl;
    report << std::string(100, '=') << std::endl << std::endl;

    // ====== Тест 1: Симметричная матрица ======
    {
        report << "TEST 1: SYMMETRIC MATRIX" << std::endl;
        report << std::string(100, '-') << std::endl;
        
        arma::sp_mat A_sym(N, N);
        arma::vec b_sym(N, arma::fill::randn);
        arma::vec x_exact(N, arma::fill::randn);
        
        // Строим симметричную матрицу: A = D + L + L^T
        for(int i = 0; i < N; i++) {
            A_sym(i, i) = 20.0 + i * 0.1;  // Диагональное доминирование
            if(i + 1 < N) {
                A_sym(i, i+1) = -1.0;
                A_sym(i+1, i) = -1.0;
            }
            if(i + 2 < N) {
                A_sym(i, i+2) = -0.5;
                A_sym(i+2, i) = -0.5;
            }
        }
        
        b_sym = A_sym * x_exact;
        arma::vec x_init(N, arma::fill::zeros);
        
        report << "Matrix properties: symmetric, diagonally dominant" << std::endl;
        report << "Condition number estimate: " << A_sym.n_nonzero << " non-zeros" << std::endl << std::endl;
        
        // Тестируем разные предобуславливатели
        std::vector<std::pair<std::string, std::function<arma::vec(const arma::vec&)>>> preconditioners = {
            {"diagonal", diagonal_precond(A_sym)},
            {"damped_jacobi", damped_jacobi(A_sym, 0.7)},
            {"gauss_seidel", gauss_seidel(A_sym)},
            {"ssor", ssor(A_sym, 1.2)},
            {"ilu0", ilu0(A_sym)},
            {"ic0", ic0(A_sym)}
        };
        
        for(auto& [precond_name, M_inv] : preconditioners) {
            // GMRES с предобуславливателем
            {
                int iter_count = 0;
                auto start = std::chrono::high_resolution_clock::now();
                arma::vec x_sol = gmres_p(A_sym, b_sym, x_init, M_inv, iter_count, 30);
                auto end = std::chrono::high_resolution_clock::now();
                double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                
                arma::vec residual = b_sym - A_sym * x_sol;
                double residual_norm = arma::norm(residual);
                double error = arma::norm(x_sol - x_exact);
                
                SolverResult res;
                res.test_type = "symmetric";
                res.precond_name = precond_name;
                res.solver_name = "gmres_p";
                res.iterations = iter_count;
                res.residual_norm = residual_norm;
                res.solution_error = error;
                res.time_ms = time_ms;
                res.converged = (residual_norm < 1e-6);
                
                results.push_back(res);
                
                report << "GMRES + " << std::setw(15) << precond_name 
                       << " | iterations: " << std::setw(2) << iter_count
                       << " | residual: " << std::scientific << std::setprecision(3) << residual_norm
                       << " | error: " << error << " | time: " << std::fixed << std::setprecision(3) 
                       << time_ms << " ms" << std::endl;
            }
            
            // BiCGSTAB с предобуславливателем
            {
                int iter_count = 0;
                auto start = std::chrono::high_resolution_clock::now();
                arma::vec x_sol = bicgstab_p(A_sym, b_sym, x_init, M_inv, iter_count);
                auto end = std::chrono::high_resolution_clock::now();
                double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                
                arma::vec residual = b_sym - A_sym * x_sol;
                double residual_norm = arma::norm(residual);
                double error = arma::norm(x_sol - x_exact);
                
                SolverResult res;
                res.test_type = "symmetric";
                res.precond_name = precond_name;
                res.solver_name = "bicgstab_p";
                res.iterations = iter_count;
                res.residual_norm = residual_norm;
                res.solution_error = error;
                res.time_ms = time_ms;
                res.converged = (residual_norm < 1e-6);
                
                results.push_back(res);
                
                report << "BiCGSTAB + " << std::setw(13) << precond_name 
                       << " | iterations: " << std::setw(2) << iter_count
                       << " | residual: " << std::scientific << std::setprecision(3) << residual_norm
                       << " | error: " << error << " | time: " << std::fixed << std::setprecision(3) 
                       << time_ms << " ms" << std::endl;
            }
        }
        report << std::endl << std::endl;
    }

    // ====== Тест 2: Положительно-определённая (SPD) матрица ======
    {
        report << "TEST 2: SYMMETRIC POSITIVE DEFINITE (SPD) MATRIX" << std::endl;
        report << std::string(100, '-') << std::endl;
        
        // Строим SPD матрицу: A = B^T * B + diag(lambda)
        arma::sp_mat B(N, N);
        for(int i = 0; i < N; i++) {
            B(i, i) = 2.0;
            if(i + 1 < N) B(i, i+1) = -1.0;
        }
        
        arma::sp_mat A_spd = B.t() * B;
        for(int i = 0; i < N; i++) {
            A_spd(i, i) += 10.0;  // Добавляем для гарантированной SPD
        }
        
        arma::vec x_exact(N, arma::fill::randn);
        arma::vec b_spd = A_spd * x_exact;
        arma::vec x_init(N, arma::fill::zeros);
        
        report << "Matrix properties: SPD, from B^T*B + lambda*I" << std::endl;
        report << "Condition number estimate: " << A_spd.n_nonzero << " non-zeros" << std::endl << std::endl;
        
        // Тестируем все предобуславливатели, включая ic0 (специально для SPD)
        std::vector<std::pair<std::string, std::function<arma::vec(const arma::vec&)>>> preconditioners = {
            {"diagonal", diagonal_precond(A_spd)},
            {"damped_jacobi", damped_jacobi(A_spd, 0.8)},
            {"gauss_seidel", gauss_seidel(A_spd)},
            {"ssor", ssor(A_spd, 1.5)},
            {"polynomial_jacobi(2)", polynomial_jacobi(A_spd, 2)},
            {"ilu0", ilu0(A_spd)},
            {"ic0", ic0(A_spd)}
        };
        
        for(auto& [precond_name, M_inv] : preconditioners) {
            // GMRES с предобуславливателем
            {
                int iter_count = 0;
                auto start = std::chrono::high_resolution_clock::now();
                arma::vec x_sol = gmres_p(A_spd, b_spd, x_init, M_inv, iter_count, 30);
                auto end = std::chrono::high_resolution_clock::now();
                double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                
                arma::vec residual = b_spd - A_spd * x_sol;
                double residual_norm = arma::norm(residual);
                double error = arma::norm(x_sol - x_exact);
                
                SolverResult res;
                res.test_type = "spd";
                res.precond_name = precond_name;
                res.solver_name = "gmres_p";
                res.iterations = iter_count;
                res.residual_norm = residual_norm;
                res.solution_error = error;
                res.time_ms = time_ms;
                res.converged = (residual_norm < 1e-6);
                
                results.push_back(res);
                
                report << "GMRES + " << std::setw(20) << precond_name 
                       << " | iterations: " << std::setw(2) << iter_count
                       << " | residual: " << std::scientific << std::setprecision(3) << residual_norm
                       << " | error: " << error << " | time: " << std::fixed << std::setprecision(3) 
                       << time_ms << " ms" << std::endl;
            }
            
            // BiCGSTAB с предобуславливателем
            {
                int iter_count = 0;
                auto start = std::chrono::high_resolution_clock::now();
                arma::vec x_sol = bicgstab_p(A_spd, b_spd, x_init, M_inv, iter_count);
                auto end = std::chrono::high_resolution_clock::now();
                double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                
                arma::vec residual = b_spd - A_spd * x_sol;
                double residual_norm = arma::norm(residual);
                double error = arma::norm(x_sol - x_exact);
                
                SolverResult res;
                res.test_type = "spd";
                res.precond_name = precond_name;
                res.solver_name = "bicgstab_p";
                res.iterations = iter_count;
                res.residual_norm = residual_norm;
                res.solution_error = error;
                res.time_ms = time_ms;
                res.converged = (residual_norm < 1e-6);
                
                results.push_back(res);
                
                report << "BiCGSTAB + " << std::setw(18) << precond_name 
                       << " | iterations: " << std::setw(2) << iter_count
                       << " | residual: " << std::scientific << std::setprecision(3) << residual_norm
                       << " | error: " << error << " | time: " << std::fixed << std::setprecision(3) 
                       << time_ms << " ms" << std::endl;
            }
        }
        report << std::endl << std::endl;
    }

    // ====== Тест 3: Произвольная матрица ======
    {
        report << "TEST 3: ARBITRARY (GENERAL) MATRIX" << std::endl;
        report << std::string(100, '-') << std::endl;
        
        arma::sp_mat A_gen(N, N);
        for(int i = 0; i < N; i++) {
            A_gen(i, i) = 15.0 + i * 0.05;
            if(i + 1 < N) {
                A_gen(i, i+1) = -0.7;
                A_gen(i+1, i) = -0.3;
            }
            if(i + 2 < N) {
                A_gen(i, i+2) = -0.1;
            }
        }
        
        arma::vec x_exact(N, arma::fill::randn);
        arma::vec b_gen = A_gen * x_exact;
        arma::vec x_init(N, arma::fill::zeros);
        
        report << "Matrix properties: general (non-symmetric)" << std::endl;
        report << "Condition number estimate: " << A_gen.n_nonzero << " non-zeros" << std::endl << std::endl;
        
        // Тестируем подходящие предобуславливатели для общего случая
        std::vector<std::pair<std::string, std::function<arma::vec(const arma::vec&)>>> preconditioners = {
            {"diagonal", diagonal_precond(A_gen)},
            {"damped_jacobi", damped_jacobi(A_gen, 0.6)},
            {"gauss_seidel", gauss_seidel(A_gen)},
            {"polynomial_jacobi(2)", polynomial_jacobi(A_gen, 2)},
            {"ilu0", ilu0(A_gen)}
        };
        
        for(auto& [precond_name, M_inv] : preconditioners) {
            // GMRES с предобуславливателем
            {
                int iter_count = 0;
                auto start = std::chrono::high_resolution_clock::now();
                arma::vec x_sol = gmres_p(A_gen, b_gen, x_init, M_inv, iter_count, 30);
                auto end = std::chrono::high_resolution_clock::now();
                double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                
                arma::vec residual = b_gen - A_gen * x_sol;
                double residual_norm = arma::norm(residual);
                double error = arma::norm(x_sol - x_exact);
                
                SolverResult res;
                res.test_type = "arbitrary";
                res.precond_name = precond_name;
                res.solver_name = "gmres_p";
                res.iterations = iter_count;
                res.residual_norm = residual_norm;
                res.solution_error = error;
                res.time_ms = time_ms;
                res.converged = (residual_norm < 1e-6);
                
                results.push_back(res);
                
                report << "GMRES + " << std::setw(20) << precond_name 
                       << " | iterations: " << std::setw(2) << iter_count
                       << " | residual: " << std::scientific << std::setprecision(3) << residual_norm
                       << " | error: " << error << " | time: " << std::fixed << std::setprecision(3) 
                       << time_ms << " ms" << std::endl;
            }
            
            // BiCGSTAB с предобуславливателем
            {
                int iter_count = 0;
                auto start = std::chrono::high_resolution_clock::now();
                arma::vec x_sol = bicgstab_p(A_gen, b_gen, x_init, M_inv, iter_count);
                auto end = std::chrono::high_resolution_clock::now();
                double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                
                arma::vec residual = b_gen - A_gen * x_sol;
                double residual_norm = arma::norm(residual);
                double error = arma::norm(x_sol - x_exact);
                
                SolverResult res;
                res.test_type = "arbitrary";
                res.precond_name = precond_name;
                res.solver_name = "bicgstab_p";
                res.iterations = iter_count;
                res.residual_norm = residual_norm;
                res.solution_error = error;
                res.time_ms = time_ms;
                res.converged = (residual_norm < 1e-6);
                
                results.push_back(res);
                
                report << "BiCGSTAB + " << std::setw(18) << precond_name 
                       << " | iterations: " << std::setw(2) << iter_count
                       << " | residual: " << std::scientific << std::setprecision(3) << residual_norm
                       << " | error: " << error << " | time: " << std::fixed << std::setprecision(3) 
                       << time_ms << " ms" << std::endl;
            }
        }
        report << std::endl << std::endl;
    }

    // ====== Итоговая статистика ======
    {
        report << std::string(100, '=') << std::endl;
        report << "SUMMARY STATISTICS" << std::endl;
        report << std::string(100, '=') << std::endl << std::endl;
        
        // Группируем по типам матриц
        for(const auto& test_type : {"symmetric", "spd", "arbitrary"}) {
            report << "Test type: " << test_type << std::endl;
            report << std::setw(20) << "Preconditioner" << " | "
                   << std::setw(10) << "GMRES" << " | "
                   << std::setw(10) << "BiCGSTAB" << " | "
                   << std::setw(15) << "Better" << std::endl;
            report << std::string(70, '-') << std::endl;
            
            std::map<std::string, std::pair<double, double>> precond_times;  // gmres_time, bicgstab_time
            
            for(const auto& res : results) {
                if(res.test_type == test_type) {
                    if(precond_times.find(res.precond_name) == precond_times.end()) {
                        precond_times[res.precond_name] = {0, 0};
                    }
                    if(res.solver_name == "gmres_p") {
                        precond_times[res.precond_name].first = res.time_ms;
                    } else {
                        precond_times[res.precond_name].second = res.time_ms;
                    }
                }
            }
            
            for(const auto& [precond, times] : precond_times) {
                std::string better = (times.first < times.second) ? "GMRES" : "BiCGSTAB";
                report << std::setw(20) << precond << " | "
                       << std::setw(10) << std::fixed << std::setprecision(3) << times.first << " ms | "
                       << std::setw(10) << std::fixed << std::setprecision(3) << times.second << " ms | "
                       << std::setw(15) << better << std::endl;
            }
            report << std::endl;
        }
    }
    
    report << "Report generated: ksp_report.txt" << std::endl;
    report.close();
    
    std::cout << "Benchmark completed! Results saved to ksp_report.txt" << std::endl;
}

void check_arm_ksp2() {
    std::ofstream report("ksp_report2.txt");
    const int N = 50;

    report << "=== Benchmark 2: CG / CR with Preconditioning ===" << std::endl;
    report << "Matrix size: " << N << "x" << N << std::endl;
    report << std::string(100, '=') << std::endl << std::endl;

    auto run_case = [&](const std::string& title,
                        arma::sp_mat& A,
                        arma::vec& b,
                        const arma::vec& x_exact,
                        const std::vector<std::pair<std::string, std::function<arma::vec(const arma::vec&)>>>& preconds,
                        bool allow_cg) {
        report << title << std::endl;
        report << std::string(100, '-') << std::endl;

        arma::vec x_init(b.n_rows, arma::fill::zeros);

        for (auto& [name, M_inv] : preconds) {
            // CG (только если SPD/симметричная часть теста позволяет)
            if (allow_cg) {
                int iters = 0;
                auto start = std::chrono::high_resolution_clock::now();
                arma::vec x_sol = cg_solver(A, b, x_init, iters, 300, 1e-6, M_inv);
                auto end = std::chrono::high_resolution_clock::now();
                double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

                arma::vec residual = b - A * x_sol;
                double residual_norm = arma::norm(residual);
                double error = arma::norm(x_sol - x_exact);

                report << "CG + " << std::setw(18) << name
                       << " | it: " << std::setw(3) << iters
                       << " | res: " << std::scientific << std::setprecision(3) << residual_norm
                       << " | err: " << error
                       << " | time: " << std::fixed << std::setprecision(3) << time_ms << " ms" << std::endl;
            }

            // CR (работает и для общего случая)
            {
                int iters = 0;
                auto start = std::chrono::high_resolution_clock::now();
                arma::vec x_sol = cr_solver(A, b, x_init, iters, 300, 1e-6, M_inv);
                auto end = std::chrono::high_resolution_clock::now();
                double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

                arma::vec residual = b - A * x_sol;
                double residual_norm = arma::norm(residual);
                double error = arma::norm(x_sol - x_exact);

                report << "CR + " << std::setw(19) << name
                       << " | it: " << std::setw(3) << iters
                       << " | res: " << std::scientific << std::setprecision(3) << residual_norm
                       << " | err: " << error
                       << " | time: " << std::fixed << std::setprecision(3) << time_ms << " ms" << std::endl;
            }
        }

        report << std::endl;
    };

    // Тест 1: Симметричная матрица (диагонально доминантная)
    {
        arma::sp_mat A_sym(N, N);
        arma::vec b_sym(N, arma::fill::randn);
        arma::vec x_exact(N, arma::fill::randn);

        for (int i = 0; i < N; ++i) {
            A_sym(i, i) = 20.0 + i * 0.1;
            if (i + 1 < N) { A_sym(i, i + 1) = -1.0; A_sym(i + 1, i) = -1.0; }
            if (i + 2 < N) { A_sym(i, i + 2) = -0.5; A_sym(i + 2, i) = -0.5; }
        }
        b_sym = A_sym * x_exact;

        std::vector<std::pair<std::string, std::function<arma::vec(const arma::vec&)>>> preconds = {
            {"diagonal", diagonal_precond(A_sym)},
            {"damped_jacobi", damped_jacobi(A_sym, 0.7)},
            {"gauss_seidel", gauss_seidel(A_sym)},
            {"ssor", ssor(A_sym, 1.2)},
            {"polynomial_jacobi(2)", polynomial_jacobi(A_sym, 2)}
        };

        run_case("TEST 1: SYMMETRIC", A_sym, b_sym, x_exact, preconds, /*allow_cg=*/true);
    }

    // Тест 2: SPD
    {
        arma::sp_mat B(N, N);
        for (int i = 0; i < N; ++i) {
            B(i, i) = 2.0;
            if (i + 1 < N) B(i, i + 1) = -1.0;
        }

        arma::sp_mat A_spd = B.t() * B;
        for (int i = 0; i < N; ++i) A_spd(i, i) += 10.0;

        arma::vec x_exact(N, arma::fill::randn);
        arma::vec b_spd = A_spd * x_exact;

        std::vector<std::pair<std::string, std::function<arma::vec(const arma::vec&)>>> preconds = {
            {"diagonal", diagonal_precond(A_spd)},
            {"damped_jacobi", damped_jacobi(A_spd, 0.8)},
            {"ssor", ssor(A_spd, 1.5)},
            {"polynomial_jacobi(2)", polynomial_jacobi(A_spd, 2)},
            {"ic0", ic0(A_spd)}
        };

        run_case("TEST 2: SPD", A_spd, b_spd, x_exact, preconds, /*allow_cg=*/true);
    }

    // Тест 3: Общая нессиметричная
    {
        arma::sp_mat A_gen(N, N);
        for (int i = 0; i < N; ++i) {
            A_gen(i, i) = 15.0 + i * 0.05;
            if (i + 1 < N) { A_gen(i, i + 1) = -0.7; A_gen(i + 1, i) = -0.3; }
            if (i + 2 < N) { A_gen(i, i + 2) = -0.1; }
        }

        arma::vec x_exact(N, arma::fill::randn);
        arma::vec b_gen = A_gen * x_exact;

        std::vector<std::pair<std::string, std::function<arma::vec(const arma::vec&)>>> preconds = {
            {"diagonal", diagonal_precond(A_gen)},
            {"damped_jacobi", damped_jacobi(A_gen, 0.6)},
            {"gauss_seidel", gauss_seidel(A_gen)},
            {"polynomial_jacobi(2)", polynomial_jacobi(A_gen, 2)},
            {"ilu0", ilu0(A_gen)}
        };

        run_case("TEST 3: GENERAL (CR only)", A_gen, b_gen, x_exact, preconds, /*allow_cg=*/false);
    }

    report << "Report generated: ksp_report2.txt" << std::endl;
    report.close();
    std::cout << "Benchmark 2 completed! Results saved to ksp_report2.txt" << std::endl;
}





int main() {
    lap_mg();
    return 0;
}