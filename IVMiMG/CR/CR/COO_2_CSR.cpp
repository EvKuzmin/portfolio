#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <mkl.h>
#include <iomanip>
#include <sstream>
#include "COO_2_CSR.h"
#include<algorithm>
#include<vector>
#include<array>

using namespace std;


/* ############################################################
Sort elements in line of CSR matrix

(IN) n - Number of elements in line
(IN/OUT) a   - values in line
(IN/OUT) ja  - index in line

############################################################ */
void ex_csrsort(MKL_INT n, double* a, MKL_INT* ja) {
    MKL_INT i, j;
    double tt;
    MKL_INT it;
    for (j = 1; j < n; j++) {
        for (i = 1; i < n; i++) {
            if (ja[i] < ja[i - 1]) {
                tt = a[i];
                a[i] = a[i - 1];
                a[i - 1] = tt;

                it = ja[i];
                ja[i] = ja[i - 1];
                ja[i - 1] = it;
            }
        }
    }

}



/* ############################################################
Convert from COO real to CSR real

(IN) n - number of rows,
(IN) m - number of columns,
(IN) nnz - number of nonzeros,

(IN) coo_ia, coo_ja, coo_a - COO matrix

(OUT) csr_ia, csr_ja, csr_a - "pointers" to CSR matrix

############################################################ */
void ex_convert_COO_2_CSR(MKL_INT n, MKL_INT m, MKL_INT nnz, MKL_INT* coo_ia, MKL_INT* coo_ja,
    double* coo_a, MKL_INT** csr_ia, MKL_INT** csr_ja, double** csr_a) {

    MKL_INT* pbegin;

    MKL_INT* ia, * ja;
    double* a;
    MKL_INT i;

    ia = (MKL_INT*)malloc((n + 1) * sizeof(MKL_INT));
    ja = (MKL_INT*)malloc(nnz * sizeof(MKL_INT));
    a = (double*)malloc(nnz * sizeof(double));
    if ((ia == NULL) || (ja == NULL) || (a == NULL)) {
        printf("ERROR   : Could not allocate arrays for CSR matrix \n");
        exit(1);
    }

    // Convert from COO -> CSR: Find array ia
    for (i = 0; i < n + 1; i++) {
        ia[i] = 0;
    }
    for (i = 0; i < nnz; i++) {
        ia[coo_ia[i] + 1]++;
    }
    for (i = 1; i < n + 1; i++) {
        ia[i] += ia[i - 1];
    }

    // Convert from COO -> CSR: Find arrays ja, a
    pbegin = (MKL_INT*)malloc(n * sizeof(MKL_INT));
    if (pbegin == NULL) {
        printf("ERROR   : Could not allocate arrays for converter COO->CSR \n");
        exit(1);
    }
    for (i = 0; i < n; i++) {
        pbegin[i] = ia[i];
    }
    for (i = 0; i < nnz; i++) {
        ja[pbegin[coo_ia[i]]] = coo_ja[i];
        a[pbegin[coo_ia[i]]] = coo_a[i];
        pbegin[coo_ia[i]]++;
    }

    // Convert from COO -> CSR: modify lines to be align with CSR spesification
    for (i = 0; i < n; i++) {
        ex_csrsort(ia[i + 1] - ia[i], &(a[ia[i]]), &(ja[ia[i]]));
    }

    *csr_ia = ia;
    *csr_ja = ja;
    *csr_a = a;
}

double eucl_norm(
    double* vec,
    MKL_INT size
) {
    double res = cblas_ddot(size, vec, 1, vec, 1);
    res = sqrt(res);
    return res;
}

double cheb_norm(
    double* vec,
    MKL_INT size
) {
    double res = 0;
    for (int i = 0; i < size; i++)
    {
        if (res < abs(vec[i]))
            res = abs(vec[i]);
    }
    return res;
}

void print(double* vec,
    MKL_INT size
)
{
    std::cout << " = ";
    std::cout << "{";
    for (int i = 0; i < size; i++) {
        std::cout << vec[i] << ", " << endl;
    }

    std::cout << "\n";
}

void CR(
    MKL_INT nrows,
    sparse_matrix_t& A,
    double* u,
    double* f,
    double eps,
    MKL_INT itermax,
    MKL_INT* number_of_operations
)
{
    ofstream plot_outf("Residual_plot.txt"); //Файл для отрисовки графика  |r^n|/|f|
    /*std::ofstream outf("CR_u, r, p.txt");*/

    double* Ap = new double[nrows];
    double* Ar = new double[nrows];
    double* p = new double[nrows];
    double* r = new double[nrows];

    int iter = 0;
    double alpha = 0.0;
    double beta = 0.0;
    double rr = 0;
    double rAr = 0;

    matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_UNIT };

    copy(f, f + nrows, r);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., A, descr, u, 1, r);

    eps *= eps * cblas_ddot(nrows, f, 1, f, 1);
    copy(r, r + nrows, p);
    rr = cblas_ddot(nrows, r, 1, r, 1);

    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, r, 0, Ar);
    rAr = cblas_ddot(nrows, r, 1, Ar, 1);

    /*outf << "Iter: " << iter << "\n" << "u: ";
    for (int i = 0; i < nrows; i++) outf << u[i] << "; ";
    outf << "\n" << "r: ";
    for (int i = 0; i < nrows; i++) outf << r[i] << "; ";
    outf << "\n" << "p: ";
    for (int i = 0; i < nrows; i++) outf << p[i] << "; ";*/
    ofstream report("report_CR_precond.txt");
    report << "iter" << '\t' << "sqrt(rr)" << endl;
    while (rr > eps && iter < itermax)
    {
        report << iter << '\t' << sqrt(rr) << endl;
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, p, 0, Ap);

        alpha = rAr / cblas_ddot(nrows, Ap, 1, Ap, 1);

        iter++;

        cblas_daxpy(nrows, alpha, p, 1, u, 1);
        cblas_daxpy(nrows, -alpha, Ap, 1, r, 1);

        rr = cblas_ddot(nrows, r, 1, r, 1);
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, r, 0, Ar);

        beta = rAr;
        rAr = cblas_ddot(nrows, r, 1, Ar, 1);
        beta = rAr / beta;

        cblas_daxpby(nrows, 1, r, 1, beta, p, 1);
        cblas_daxpby(nrows, 1, Ar, 1, beta, Ap, 1);

        /*outf << "\n\n" << "Iter: " << iter << "\n" << "u: ";
        for (int i = 0; i < nrows; i++) outf << u[i] << "; ";
        outf << "\n" << "r: ";
        for (int i = 0; i < nrows; i++) outf << r[i] << "; ";
        outf << "\n" << "p: ";
        for (int i = 0; i < nrows; i++) outf << p[i] << "; ";*/


        plot_outf << iter << "\t" << eucl_norm(r, nrows) / eucl_norm(f, nrows) << "\n";

        *number_of_operations = iter;
    }
    plot_outf.close();

    delete[] Ap;
    delete[] Ar;
    delete[] p;
    delete[] r;
}



void CR_AT(  //Метод сопряжённых невязок с предобуславливанием А^T
    MKL_INT nrows,                //Размер матрицы
    sparse_matrix_t& A,       //Матрица в формате CSR
    double* u,                //Начальное приближение
    double* f,                //Правая часть
    double eps,               //Точность, критерий остановки
    MKL_INT itermax,              //Максимальное число итераций, критерий остановки
    MKL_INT* number_of_operations //Выполненное число итераций
)
{
    ofstream plot_outf("Residual_plot.txt"); //Файл для отрисовки графика |r^n|/|f|

    double* Ap = new double[nrows];
    double* Atr = new double[nrows]; // A^T*r^n
    double* p = new double[nrows];
    double* r = new double[nrows];

    int iter = 0;
    double alpha = 0.0;
    double beta = 0.0;
    double rr = 0;       // (r^n, r^n)
    double Atr_p = 0;    // (A^T*r^n, p^n)
    double Atr_Atr = 0;  // (A^T*r^n, A^T*r^n)

    matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_UNIT };

    eps *= eps * cblas_ddot(nrows, f, 1, f, 1);
    copy(f, f + nrows, r);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., A, descr, u, 1, r);
    rr = cblas_ddot(nrows, r, 1, r, 1);

    mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1., A, descr, r, 0, Atr);
    copy(Atr, Atr + nrows, p);
    Atr_p = cblas_ddot(nrows, Atr, 1, p, 1);
    Atr_Atr = cblas_ddot(nrows, Atr, 1, Atr, 1);

    while (rr > eps && iter < itermax) {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, p, 0, Ap);
        alpha = Atr_p / cblas_ddot(nrows, Ap, 1, Ap, 1);

        iter++;

        cblas_daxpy(nrows, alpha, p, 1, u, 1);
        cblas_daxpy(nrows, -alpha, Ap, 1, r, 1);

        rr = cblas_ddot(nrows, r, 1, r, 1);
        mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1., A, descr, r, 0, Atr);

        beta = Atr_Atr;
        Atr_Atr = cblas_ddot(nrows, Atr, 1, Atr, 1);
        beta = Atr_Atr / beta;

        cblas_daxpby(nrows, 1, Atr, 1, beta, p, 1);
        Atr_p = cblas_ddot(nrows, Atr, 1, p, 1);

        plot_outf << iter << "\t" << eucl_norm(r, nrows) / eucl_norm(f, nrows) << "\n";

        *number_of_operations = iter;
    }
    plot_outf.close();


    delete[] Ap;
    delete[] Atr;
    delete[] p;
    delete[] r;
}


CRSArrays get_csr_submatrix(const CRSArrays& A, MKL_INT* rows, MKL_INT* cols, MKL_INT N, MKL_INT M) {
    CRSArrays res;
    std::vector<MKL_INT> ia_new(N + 1, 0);
    std::vector<MKL_INT> ja_new;
    std::vector<double> a_new;
    MKL_INT n = 0;
    for (int i = 0; i < N; i++) {
        MKL_INT row = rows[i];
        for (int j = A.ia[row]; j < A.ia[row + 1]; j++) {
            MKL_INT col = A.ja[j];
            auto it = std::lower_bound(cols, cols + M, col);
            if (it != cols + M)
                if ((*it) == col) {
                    n++;
                    ja_new.push_back(it - cols);
                    a_new.push_back(A.a[j]);
                }
        }
        ia_new[i + 1] = n;
    }
    res.n = N;
    res.m = M;
    res.nnz = ja_new.size();
    res.ia = new MKL_INT[N + 1];
    res.ja = new MKL_INT[res.nnz];
    res.a = new double[res.nnz];
    std::copy(ia_new.begin(), ia_new.end(), res.ia);
    std::copy(ja_new.begin(), ja_new.end(), res.ja);
    std::copy(a_new.begin(), a_new.end(), res.a);
    return res;
}

void CRSArrays::print() const {
    for (int i = 0; i < n; i++) {
        std::cout << "        ";
        for (int j = ia[i]; j < ia[i + 1]; j++) {
            std::cout << " " << std::setw(8) << ja[j];
        }
        std::cout << "\n";
        std::cout << std::setw(8) << i;
        for (int j = ia[i]; j < ia[i + 1]; j++) {
            std::cout << " " << std::setw(8) << std::setprecision(3) << a[j];
        }
        std::cout << "\n\n";
    }






}




int len_file(string filename) { //no more 1024 symbol in string
    char* str = new char[1024];
    int i = 0;
    ifstream base(filename);
    while (!base.eof())
    {
        base.getline(str, 1024, '\n');
        i++;
    }
    base.close();
    delete[] str;
    return i;
}


void CR_precond( //only symmetric matrix A
    CRSArrays& A_crs,
    double* u,
    double* f,
    double eps,
    MKL_INT itermax,
    MKL_INT* number_of_operations,
    double gamma,
    MKL_INT* p_ind,
    MKL_INT* u_ind,
    MKL_INT len_p,
    MKL_INT len_u

)
{
    matrix_descr descr = {SPARSE_MATRIX_TYPE_GENERAL};

    CRSArrays C_CRS = get_csr_submatrix(A_crs, u_ind, p_ind, len_u, len_p);
    sparse_matrix_t C;
    mkl_sparse_d_create_csr(&C, SPARSE_INDEX_BASE_ZERO, C_CRS.n, C_CRS.m, C_CRS.ia, (C_CRS.ia + 1), C_CRS.ja, C_CRS.a);

    CRSArrays CT_CRS = get_csr_submatrix(A_crs, p_ind, u_ind, len_p, len_u);
    sparse_matrix_t CT;
    mkl_sparse_d_create_csr(&CT, SPARSE_INDEX_BASE_ZERO, CT_CRS.n, CT_CRS.m, CT_CRS.ia, (CT_CRS.ia + 1), CT_CRS.ja, CT_CRS.a);

    CRSArrays D_CRS = get_csr_submatrix(A_crs, u_ind, u_ind, len_u, len_u);
    sparse_matrix_t D;
    mkl_sparse_d_create_csr(&D, SPARSE_INDEX_BASE_ZERO, D_CRS.n, D_CRS.m, D_CRS.ia, (D_CRS.ia + 1), D_CRS.ja, D_CRS.a);

    sparse_matrix_t CCT;
    mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, C, CT, &CCT);

    sparse_matrix_t D_;
    mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, CCT, gamma, D, &D_);

    double* u1 = new double[len_u];
    for (int i = 0; i < len_u; i++) u1[i] = u[i];
    double* u2 = new double[len_p];
    for (int i = 0; i < len_p; i++) u2[i] = u[len_u + i];
    double* f1 = new double[len_u];
    for (int i = 0; i < len_u; i++) f1[i] = f[i];
    double* f2 = new double[len_p];
    for (int i = 0; i < len_p; i++) f2[i] = f[len_u + i];
    double* f1_ = new double[len_u];

    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, gamma, C, descr, f2, 1., f1);
    for (int i = 0; i < len_u; i++) f1_[i] = f1[i];

    double* r1 = new double[len_u];
    double* D_u1 = new double[len_u];
    double* Cu2 = new double[len_u];
    for (int i = 0; i < len_u; i++) Cu2[i] = 0;
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., D_, descr, u1, 0, D_u1);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., C, descr, u2, 0, Cu2);
    for (int i = 0; i < len_u; i++) r1[i] = D_u1[i] + Cu2[i] - f1_[i];

    double* r2 = new double[len_p];
    double* CTu1 = new double[len_p];
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., CT, descr, u1, 0, CTu1);
    for (int i = 0; i < len_p; i++) r2[i] = CTu1[i] - f2[i];

    double* r = new double[A_crs.n];
    for (int i = 0; i < len_u; i++) r[i] = r1[i];
    for (int i = 0; i < len_p; i++) r[len_u + i] = r2[i];
    double rr = cblas_ddot(A_crs.n, r, 1, r, 1);

    int iter = 0;
    eps *= eps;

    ofstream report("report_CR_precond.txt");
    report << "iter" << '\t' << "sqrt(rr)" << endl;

    while (rr > eps && iter < itermax)
    {   
        report << iter << '\t' << sqrt(rr) << endl;

        for (int i = 0; i < len_u; i++) u1[i] = D_u1[i] + Cu2[i];
        for (int i = 0; i < len_p; i++) u2[i] = CTu1[i];
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., D_, descr, u1, 0, D_u1);
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., C, descr, u2, 0, Cu2);
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., CT, descr, u1, 0, CTu1);
        for (int i = 0; i < len_u; i++) r1[i] = D_u1[i] + Cu2[i] - f1_[i];
        for (int i = 0; i < len_p; i++) r2[i] = CTu1[i] - f2[i];
        for (int i = 0; i < len_u; i++) r[i] = r1[i];
        for (int i = 0; i < len_p; i++) r[len_u + i] = r2[i];
        rr = cblas_ddot(A_crs.n, r, 1, r, 1);
        iter++;
        
    }
    *number_of_operations = iter;
    delete[] r;
    delete[] CTu1;
    delete[] r2;
    delete[] r1;
    delete[] Cu2;
    delete[] D_u1;
    delete[] f1_;
    delete[] f1;
    delete[] f2;
    delete[] u1;
    delete[] u2;

}


void CR_precond_2( //only symmetric matrix A
    MKL_INT nrows,
    sparse_matrix_t& A,
    double* u,
    double* f,
    double eps,
    MKL_INT itermax,
    MKL_INT* number_of_operations,
    CRSArrays& A_crs,
    double gamma,
    MKL_INT* p_ind,
    MKL_INT* u_ind,
    MKL_INT len_p,
    MKL_INT len_u
)
{
    
    CRSArrays C_CRS = get_csr_submatrix(A_crs, u_ind, p_ind, len_u, len_p);
    sparse_matrix_t C;
    mkl_sparse_d_create_csr(&C, SPARSE_INDEX_BASE_ZERO, C_CRS.n, C_CRS.m, C_CRS.ia, (C_CRS.ia + 1), C_CRS.ja, C_CRS.a);

    CRSArrays CT_CRS = get_csr_submatrix(A_crs, p_ind, u_ind, len_p, len_u);
    sparse_matrix_t CT;
    mkl_sparse_d_create_csr(&CT, SPARSE_INDEX_BASE_ZERO, CT_CRS.n, CT_CRS.m, CT_CRS.ia, (CT_CRS.ia + 1), CT_CRS.ja, CT_CRS.a);
    sparse_matrix_t CCT;
    mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, C, CT, &CCT);
    double* CTCp1 = new double[len_u];
    double* CTCu1 = new double[len_u];
    double* CTCr1 = new double[len_u];
    double* CTCp = new double[nrows];
    double* CTCu = new double[nrows];
    double* CTCr = new double[nrows];

    ofstream plot_outf("Residual_plot.txt"); //Файл для отрисовки графика  |r^n|/|f|
    /*std::ofstream outf("CR_u, r, p.txt");*/

    double* Ap = new double[nrows];
    double* Ar = new double[nrows];
    double* p = new double[nrows];
    double* r = new double[nrows];

    int iter = 0;
    double alpha = 0.0;
    double beta = 0.0;
    double rr = 0;
    double rAr = 0;

    matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL };

    copy(f, f + nrows, r);
    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., A, descr, u, 1, r);

    cblas_daxpy(nrows, -gamma, CTCu, 1, r, 1);

    eps *= eps * cblas_ddot(nrows, f, 1, f, 1);
    copy(r, r + nrows, p);
    rr = cblas_ddot(nrows, r, 1, r, 1);

    mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, r, 0, Ar);
    rAr = cblas_ddot(nrows, r, 1, Ar, 1);

    /*outf << "Iter: " << iter << "\n" << "u: ";
    for (int i = 0; i < nrows; i++) outf << u[i] << "; ";
    outf << "\n" << "r: ";
    for (int i = 0; i < nrows; i++) outf << r[i] << "; ";
    outf << "\n" << "p: ";
    for (int i = 0; i < nrows; i++) outf << p[i] << "; ";*/

    while (rr > eps && iter < itermax)
    {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, p, 0, Ap);

        alpha = rAr / cblas_ddot(nrows, Ap, 1, Ap, 1);

        iter++;

        cblas_daxpy(nrows, alpha, p, 1, u, 1);
        cblas_daxpy(nrows, -alpha, Ap, 1, r, 1);

        rr = cblas_ddot(nrows, r, 1, r, 1);
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, r, 0, Ar);

        beta = rAr;
        rAr = cblas_ddot(nrows, r, 1, Ar, 1);
        beta = rAr / beta;

        cblas_daxpby(nrows, 1, r, 1, beta, p, 1);
        cblas_daxpby(nrows, 1, Ar, 1, beta, Ap, 1);

        /*outf << "\n\n" << "Iter: " << iter << "\n" << "u: ";
        for (int i = 0; i < nrows; i++) outf << u[i] << "; ";
        outf << "\n" << "r: ";
        for (int i = 0; i < nrows; i++) outf << r[i] << "; ";
        outf << "\n" << "p: ";
        for (int i = 0; i < nrows; i++) outf << p[i] << "; ";*/


        plot_outf << iter << "\t" << eucl_norm(r, nrows) / eucl_norm(f, nrows) << "\n";

        *number_of_operations = iter;
    }
    plot_outf.close();

    delete[] Ap;
    delete[] Ar;
    delete[] p;
    delete[] r;
    }


CRSArrays vertical_concatinate(CRSArrays A, CRSArrays B) { //A.m = B.m A-upper, B-lower
    if (A.m != B.m) { 
        cout << "error concatinate";
        return A; 
    }
    CRSArrays A_B;
    A_B.m = A.m;
    A_B.n = A.n + B.n;
    A_B.nnz = A.nnz + B.nnz;
    double* newa = new double[A_B.nnz];
    MKL_INT* newja = new MKL_INT[A_B.nnz];
    MKL_INT* newia = new MKL_INT[A_B.n + 1];
    for (int i = 0; i < A.nnz; i++) {
        newa[i] = A.a[i];
        newja[i] = A.ja[i];
    }
    for (int i = A.nnz; i < A_B.nnz; i++) {
        newa[i] = B.a[i-A.nnz];
        newja[i] = B.ja[i - A.nnz];
    }
    for (int i = 0; i < A.n + 1; i++) {
        newia[i] = A.ia[i];
    }
    for (int i = 1; i < B.n + 1; i++) {
        newia[A.n + i] = B.ia[i] + A.nnz;
    }
    A_B.a = newa;
    A_B.ia = newia;
    A_B.ja = newja;
    return A_B;
}
CRSArrays horizontal_concatinate(CRSArrays A, CRSArrays B) { //A.n = B.n A-left, B-right
    if (A.n != B.n) {
        cout << "error concatinate";
        return A;
    }
    CRSArrays A_B;
    A_B.m = A.m + B.m;
    A_B.n = A.n;
    A_B.nnz = A.nnz + B.nnz;
    double* newa = new double[A_B.nnz];
    MKL_INT* newja = new MKL_INT[A_B.nnz];
    MKL_INT* newia = new MKL_INT[A_B.n + 1];

    for (int i = 0; i < A_B.n + 1; i++) {
        newia[i] = A.ia[i] + B.ia[i];
    }

    for (int i = 0; i < A_B.n; i++) {
        for (int j = 0; j < A.ia[i + 1] - A.ia[i]; j++) {
            newa[newia[i]+j] = A.a[A.ia[i]+j];
            newja[newia[i] + j] = A.ja[A.ia[i] + j];
        }
        for (int j = A.ia[i + 1] - A.ia[i]; j < newia[i + 1] - newia[i]; j++) {
            newa[newia[i] + j] = B.a[B.ia[i] + j - (A.ia[i + 1] - A.ia[i])];
            newja[newia[i] + j] = B.ja[B.ia[i] + j - (A.ia[i + 1] - A.ia[i])] + A.m;
        }
    }
    ///////////////////////////
    //for (int i = 0; i < A.nnz; i++) {
    //    newa[i] = A.a[i];
    //}
    //for (int i = A.nnz; i < A_B.nnz; i++) {
    //    newa[i] = B.a[i - A.nnz];
    //}
    //for (int i = 0; i < A.nnz; i++) {
    //    newja[i] = A.ja[i];
    //}
    //for (int i = 0; i < B.nnz; i++) {
    //    newja[i + A.nnz] = B.ja[i] + A.m;
    //}
    /////////////////////////////
    A_B.a = newa;
    A_B.ia = newia;
    A_B.ja = newja;
    return A_B;

}

CRSSystem precondCCT(CRSSystem A,
    double gamma,
    MKL_INT* p_ind,
    MKL_INT* u_ind,
    MKL_INT len_p,
    MKL_INT len_u) {

    CRSArrays AZero_CRS = get_csr_submatrix(A.M, p_ind, p_ind, len_p, len_p);

    matrix_descr descr = {SPARSE_MATRIX_TYPE_GENERAL};
    CRSArrays C_CRS = get_csr_submatrix(A.M, u_ind, p_ind, len_u, len_p);
    sparse_matrix_t C;
    mkl_sparse_d_create_csr(&C, SPARSE_INDEX_BASE_ZERO, C_CRS.n, C_CRS.m, C_CRS.ia, (C_CRS.ia + 1), C_CRS.ja, C_CRS.a);
    CRSArrays CT_CRS = get_csr_submatrix(A.M, p_ind, u_ind, len_p, len_u);
    sparse_matrix_t CT;
    mkl_sparse_d_create_csr(&CT, SPARSE_INDEX_BASE_ZERO, CT_CRS.n, CT_CRS.m, CT_CRS.ia, (CT_CRS.ia + 1), CT_CRS.ja, CT_CRS.a);
    CRSArrays D_CRS = get_csr_submatrix(A.M, u_ind, u_ind, len_u, len_u);
    sparse_matrix_t D;
    mkl_sparse_d_create_csr(&D, SPARSE_INDEX_BASE_ZERO, D_CRS.n, D_CRS.m, D_CRS.ia, (D_CRS.ia + 1), D_CRS.ja, D_CRS.a);
    sparse_matrix_t CCT;
    mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, C, CT, &CCT);
    sparse_matrix_t D_;
    if (gamma != 0) mkl_sparse_d_add(SPARSE_OPERATION_NON_TRANSPOSE, CCT, gamma, D, &D_);
    else D_ = D;

    CRSArrays CRSD_;
    sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
    MKL_INT n;  //< the number of rows
    MKL_INT m;  //< the number of columns
    MKL_INT nnz;//< the number of nnz (== ia[n])
    double* a;  //< the values (of size NNZ)
    MKL_INT* ia;//< the usual rowptr (of size n+1)
    MKL_INT* ja;//< the colidx of each NNZ (of size nnz)
    MKL_INT* dummy;
    mkl_sparse_d_export_csr(D_, &indexing, &n, &m, &ia, &dummy, &ja, &a);
    CRSD_.a = a;
    CRSD_.ia = ia;
    CRSD_.ja = ja;
    CRSD_.m = m;
    CRSD_.n = n;
    CRSD_.nnz = ia[CRSD_.n];

    CRSArrays CRSD_C_CRS = horizontal_concatinate(CRSD_, C_CRS);
    CT_CRS.m += len_p;
    CRSArrays A_ = vertical_concatinate(CRSD_C_CRS, CT_CRS);

    double* f = new double[len_u+len_p];
    double* f1 = new double[len_u];
    double* f2 = new double[len_p];
    for (int i = 0; i < len_u; i++) {
        f1[i] = A.f[u_ind[i]];
    }
    for (int i = 0; i < len_p; i++) {
        f2[i] = A.f[p_ind[i]];
    }
    if (gamma != 0) mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, gamma, C, descr, f2, 1., f1);

    for (int i = 0; i < len_u; i++) f[i] = f1[i];
    for (int i = 0; i < len_p; i++) f[len_u + i] = f2[i];
    CRSSystem SystemPrecond;
    SystemPrecond.f = f;
    SystemPrecond.M = A_;
    return SystemPrecond;
}