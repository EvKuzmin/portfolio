#pragma once

#include "armadillo"
#include <functional>
#include <vector>
#include <cmath>

namespace solvers {

using Precond = std::function<arma::vec(const arma::vec&)>;

// Единичный предобуславливатель (тождественная функция)
inline Precond identity_precond() {
    return [](const arma::vec& v) -> arma::vec { return v; };
}

// Прямые решатели
inline arma::vec solve_dense(const arma::mat& A, const arma::vec& b) {
    return arma::solve(A, b);
}

inline arma::vec solve_sparse(const arma::sp_mat& A, const arma::vec& b) {
    arma::vec x;
    if (arma::spsolve(x, A, b)) {
        return x;
    }
    // Fallback: плотное решение, если разреженный решатель не справился
    return arma::solve(arma::mat(A), b);
}

// Итерационные решатели (обёртки) с предобуславливателем по умолчанию (I)
inline arma::vec gmres_solver(const arma::sp_mat& A,
                              const arma::vec& b,
                              arma::vec x0,
                              int& iterations_count,
                              int restart = 30,
                              int max_it = 300,
                              double tol = 1e-6,
                              Precond M = Precond()) {
    if (!M) M = identity_precond();

    const arma::uword n = b.n_rows;
    arma::vec x = std::move(x0);
    iterations_count = 0;

    auto apply_A = [&A](const arma::vec& v) -> arma::vec { return A * v; };
    auto apply_Minv = [&M](const arma::vec& v) -> arma::vec { return M(v); };

    arma::vec r = b - apply_A(x);
    arma::vec z = apply_Minv(r);
    double beta = arma::norm(z);
    if (beta <= tol) return x;

    const int m = std::max(1, restart);

    // Память для Хессенберга и Givens
    arma::mat H(m + 1, m, arma::fill::zeros);
    arma::vec cs(m, arma::fill::zeros);
    arma::vec sn(m, arma::fill::zeros);

    std::vector<arma::vec> V(m + 1, arma::vec(n, arma::fill::zeros));

    auto apply_givens = [&](double& h1, double& h2, double c, double s) {
        double tmp = c * h1 + s * h2;
        h2 = -s * h1 + c * h2;
        h1 = tmp;
    };

    while (iterations_count < max_it) {
        r = b - apply_A(x);
        z = apply_Minv(r);
        beta = arma::norm(z);
        if (beta <= tol) break;

        V[0] = z / beta;
        arma::vec g(m + 1, arma::fill::zeros);
        g(0) = beta;
        H.zeros();

        int j_converged = -1;

        for (int j = 0; j < m && iterations_count < max_it; ++j) {
            arma::vec w = apply_Minv(apply_A(V[j]));

            for (int i = 0; i <= j; ++i) {
                H(i, j) = arma::dot(w, V[i]);
                w -= H(i, j) * V[i];
            }
            H(j + 1, j) = arma::norm(w);
            if (H(j + 1, j) > 0.0) {
                V[j + 1] = w / H(j + 1, j);
            } else {
                // Преждевременная остановка при вырожденном направлении
                for (int i = 0; i < j; ++i) apply_givens(H(i, j), H(i + 1, j), cs(i), sn(i));
                double delta = std::hypot(H(j, j), H(j + 1, j));
                cs(j) = (delta == 0.0) ? 1.0 : H(j, j) / delta;
                sn(j) = (delta == 0.0) ? 0.0 : H(j + 1, j) / delta;
                apply_givens(H(j, j), H(j + 1, j), cs(j), sn(j));
                apply_givens(g(j), g(j + 1), cs(j), sn(j));
                j_converged = j;
                ++iterations_count;
                break;
            }

            // Применяем существующие Givens к столбцу j
            for (int i = 0; i < j; ++i) apply_givens(H(i, j), H(i + 1, j), cs(i), sn(i));

            // Вычисляем новую ротацию для зануления H(j+1,j)
            double delta = std::hypot(H(j, j), H(j + 1, j));
            cs(j) = (delta == 0.0) ? 1.0 : H(j, j) / delta;
            sn(j) = (delta == 0.0) ? 0.0 : H(j + 1, j) / delta;
            apply_givens(H(j, j), H(j + 1, j), cs(j), sn(j));
            apply_givens(g(j), g(j + 1), cs(j), sn(j));

            ++iterations_count;

            if (std::abs(g(j + 1)) <= tol) {
                j_converged = j;
                break;
            }
        }

        int j_last = (j_converged >= 0) ? j_converged : std::min(m - 1, (int)H.n_cols - 1);
        // Верхнетреугольная система R y = g(0:j_last)
        arma::vec y(j_last + 1, arma::fill::zeros);
        for (int i = j_last; i >= 0; --i) {
            double sum = 0.0;
            for (int k = i + 1; k <= j_last; ++k) sum += H(i, k) * y(k);
            y(i) = (g(i) - sum) / H(i, i);
        }

        // Обновление решения: x += V(:,0..j_last) * y
        for (int i = 0; i <= j_last; ++i) x += y(i) * V[i];

        // Проверка сходимости перед следующим рестартом
        r = b - apply_A(x);
        if (arma::norm(apply_Minv(r)) <= tol) break;
    }

    return x;
}

inline arma::vec gmres_solver(const arma::sp_mat& A,
                              const arma::vec& b,
                              arma::vec x0,
                              int restart = 30,
                              int max_it = 300,
                              double tol = 1e-6,
                              Precond M = Precond()) {
    int iters = 0;
    return gmres_solver(A, b, std::move(x0), iters, restart, max_it, tol, std::move(M));
}

inline arma::vec bicgstab_solver(const arma::sp_mat& A,
                                 const arma::vec& b,
                                 arma::vec x0,
                                 int& iterations_count,
                                 int max_it = 300,
                                 double tol = 1e-6,
                                 Precond M = Precond()) {
    if (!M) M = identity_precond();

    arma::vec x = std::move(x0);
    arma::vec r = b - A * x;
    arma::vec r_tld = r; // фиксированный тензиор

    arma::vec p = arma::zeros<arma::vec>(b.n_rows);
    arma::vec v = arma::zeros<arma::vec>(b.n_rows);
    arma::vec s, t;

    double alpha = 1.0, omega = 1.0, rho = 1.0, rho_1 = 1.0;
    iterations_count = 0;

    for (int it = 0; it < max_it; ++it) {
        rho_1 = arma::dot(r_tld, r);
        if (std::abs(rho_1) < 1e-30) break; // breakdown

        if (it == 0) {
            p = r;
        } else {
            double beta = (rho_1 / rho) * (alpha / omega);
            p = r + beta * (p - omega * v);
        }

        arma::vec phat = M(p);           // левое предобуславливание
        v = A * phat;
        alpha = rho_1 / arma::dot(r_tld, v);
        s = r - alpha * v;

        if (arma::norm(s) <= tol) {
            x += alpha * phat;
            ++iterations_count;
            break;
        }

        arma::vec shat = M(s);
        t = A * shat;
        double tt = arma::dot(t, t);
        if (tt == 0.0) break; // degenerate
        omega = arma::dot(t, s) / tt;

        x += alpha * phat + omega * shat;
        r = s - omega * t;
        ++iterations_count;

        if (arma::norm(r) <= tol || std::abs(omega) < 1e-30) break;

        rho = rho_1;
    }

    return x;
}

inline arma::vec bicgstab_solver(const arma::sp_mat& A,
                                 const arma::vec& b,
                                 arma::vec x0,
                                 int max_it = 300,
                                 double tol = 1e-6,
                                 Precond M = Precond()) {
    int iters = 0;
    return bicgstab_solver(A, b, std::move(x0), iters, max_it, tol, std::move(M));
}

// Conjugate Gradient (SPD) с левым предобуславливанием
inline arma::vec cg_solver(const arma::sp_mat& A,
                           const arma::vec& b,
                           arma::vec x0,
                           int& iterations_count,
                           int max_it = 300,
                           double tol = 1e-6,
                           Precond M = Precond()) {
    if (!M) M = identity_precond();

    arma::vec x = std::move(x0);
    arma::vec r = b - A * x;
    arma::vec z = M(r);
    arma::vec p = z;

    double rho = arma::dot(r, z);
    iterations_count = 0;

    for (int it = 0; it < max_it; ++it) {
        arma::vec Ap = A * p;
        double denom = arma::dot(p, Ap);
        if (std::abs(denom) < 1e-30) break;

        double alpha = rho / denom;
        x += alpha * p;
        r -= alpha * Ap;

        if (arma::norm(r) <= tol) { ++iterations_count; break; }

        z = M(r);
        double rho_new = arma::dot(r, z);
        if (std::abs(rho) < 1e-30) break;
        double beta = rho_new / rho;
        p = z + beta * p;
        rho = rho_new;
        ++iterations_count;
    }

    return x;
}

inline arma::vec cg_solver(const arma::sp_mat& A,
                           const arma::vec& b,
                           arma::vec x0,
                           int max_it = 300,
                           double tol = 1e-6,
                           Precond M = Precond()) {
    int iters = 0;
    return cg_solver(A, b, std::move(x0), iters, max_it, tol, std::move(M));
}

// Conjugate Residual с левым предобуславливанием
inline arma::vec cr_solver(const arma::sp_mat& A,
                           const arma::vec& b,
                           arma::vec x0,
                           int& iterations_count,
                           int max_it = 300,
                           double tol = 1e-6,
                           Precond M = Precond()) {
    if (!M) M = identity_precond();

    arma::vec x = std::move(x0);
    arma::vec r = b - A * x;
    arma::vec z = M(r);
    arma::vec p = z;
    arma::vec Ap = A * p;

    iterations_count = 0;

    for (int it = 0; it < max_it; ++it) {
        double ApAp = arma::dot(Ap, Ap);
        if (std::abs(ApAp) < 1e-30) break;

        double alpha = arma::dot(r, Ap) / ApAp;
        x += alpha * p;
        r -= alpha * Ap;

        if (arma::norm(r) <= tol) { ++iterations_count; break; }

        z = M(r);
        arma::vec Az = A * z;
        double beta = arma::dot(Az, Ap) / ApAp;

        p = z - beta * p;
        Ap = Az - beta * Ap;

        ++iterations_count;
    }

    return x;
}

inline arma::vec cr_solver(const arma::sp_mat& A,
                           const arma::vec& b,
                           arma::vec x0,
                           int max_it = 300,
                           double tol = 1e-6,
                           Precond M = Precond()) {
    int iters = 0;
    return cr_solver(A, b, std::move(x0), iters, max_it, tol, std::move(M));
}

// --- Совместимость с прежними именами ---
inline arma::vec gmres(const arma::sp_mat& A, const arma::vec& rhs, arma::vec x,
                       int restart = 30, int max_it = 300, double tol = 1e-6) {
    return gmres_solver(A, rhs, std::move(x), restart, max_it, tol);
}

inline arma::vec gmres_p(const arma::sp_mat& A, const arma::vec& rhs, arma::vec x,
                         Precond& M_inv,
                         int& iterations_count,
                         int restart = 30,
                         int max_it = 300,
                         double tol = 1e-6) {
    return gmres_solver(A, rhs, std::move(x), iterations_count, restart, max_it, tol, M_inv);
}

inline arma::vec bicgstab(const arma::sp_mat& A, const arma::vec& rhs, arma::vec x,
                          int max_it = 300, double tol = 1e-6) {
    return bicgstab_solver(A, rhs, std::move(x), max_it, tol);
}

inline arma::vec bicgstab_p(const arma::sp_mat& A, const arma::vec& rhs, arma::vec x,
                            Precond& M_inv,
                            int& iterations_count,
                            int max_it = 300,
                            double tol = 1e-6) {
    return bicgstab_solver(A, rhs, std::move(x), iterations_count, max_it, tol, M_inv);
}

} // namespace solvers
