#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <mkl.h>
#include <iomanip>
#include "Support_functions.h"

#define M_PI 3.14159265358979323846

using namespace std;

int box_check(
	int x, 
	int y, 
	int z, 
	int nx, 
	int ny, 
	int nz
)
{
	return
		(0 <= x) && (x < nx) &&
		(0 <= y) && (y < ny) &&
		(0 <= z) && (z < nz);
}

void diffconv3d_mx2(
	int nx, 
	int ny, 
	int nz, 
	double p, 
	double q, 
	double r, 
	int* n, 
	int** ia, 
	int** ja, 
	double** a
)
{
	int x, y, z, cp, cr, i, nnz, nrow;

	// output:
	int* irow, * icol;
	double* data;

	// offsets:
	int xof[7] = { 0, 0, -1, 0, 1, 0, 0 };
	int yof[7] = { 0, -1, 0, 0, 0, 1, 0 };
	int zof[7] = { -1, 0, 0, 0, 0, 0, 1 };
	int crof[7] = { -nx * ny, -nx, -1, 0, 1, nx, nx * ny };
	double c[7];

	// mx coeffs (exponential scheme):
	{
		c[0] = -exp(0.5 * r / ((double)nz + 1));
		c[1] = -exp(0.5 * q / ((double)ny + 1));
		c[2] = -exp(0.5 * p / ((double)nx + 1));
		c[4] = -exp(-0.5 * p / ((double)nx + 1));
		c[5] = -exp(-0.5 * q / ((double)ny + 1));
		c[6] = -exp(-0.5 * r / ((double)nz + 1));

		c[3] = -(c[0] + c[1] + c[2] + c[4] + c[5] + c[6]);
	}

	// alloc arrays:
	{
		nrow = nx * ny * nz;
		nnz = 7 * nx * ny * nz - 2 * (nx * ny + ny * nz + nz * nx);

		irow = new int[nrow + 1];
		icol = new int[nnz];
		data = new double[nnz];
	}

	// fill everything:
	cp = 0;

	for (z = 0; z < nz; z++)
		for (y = 0; y < ny; y++)
			for (x = 0; x < nx; x++)
			{
				cr = x + y * nx + z * nx * ny;

				for (i = 0; i < 7; i++)
					if (box_check(x + xof[i], y + yof[i], z + zof[i], nx, ny, nz))
					{
						icol[cp] = cr + crof[i];
						data[cp++] = c[i];
					}

				irow[cr + 1] = cp;
			}

	irow[0] = 0;

	// voila.
	if (n)  *n = nrow;
	if (ia) *ia = irow; else delete[] irow;
	if (ja) *ja = icol; else delete[] icol;
	if (a)  *a = data; else delete[] data;
}

double* exact(
	int nx, 
	int ny, 
	int nz
)
{
	int i, j, k;
	double* result = new double[nx * ny * nz];

	for (k = 0; k < nz; k++)
		for (j = 0; j < ny; j++)
			for (i = 0; i < nx; i++)
			{
				double
					x = (double)i / ((double)nx - 1),
					y = (double)j / ((double)ny - 1),
					z = (double)k / ((double)nz - 1);

				//result[i + j * nx + k * nx * ny] = x * x + y * y + z * z;  // for example
				//result[i + j * nx + k * nx * ny] = sin(M_PI * x) * sin(M_PI * y);
				result[i + j * nx + k * nx * ny] = 1;
			}

	return result;
}

void print(
	double* vec, 
	int size, 
	string vec_name
)
{
	std::cout << vec_name << ": ";
	for (int i = 0; i < size; i++)
	{
		std::cout << vec[i] << "; ";
	}

	std::cout << "\n";
}

void print(int* vec, 
	int size, 
	string vec_name
)
{
	std::cout << vec_name << ": ";
	for (int i = 0; i < size; i++)
	{
		std::cout << vec[i] << "; ";
	}

	std::cout << "\n";
}

//ПЕЧАТЬ Вектора CRS для Excel в виде столбца
void print_vect_to_Excel(double* vec, 
	int size,
	string vec_name
)
{
	ofstream f(vec_name + ".txt");
	for (int i = 0; i < size; i++)
	{
		f << vec[i] << "\n";
	}
	f.close();
}

//ПЕЧАТЬ ВЕКТОРА для Wolfram в плотном виде
void print_to_Wolfram(
	double* vec,
	int size, 
	string vec_name
)
{
	ofstream f(vec_name + ".txt");
	f << vec_name << " = ";
	f << "{";
	for (int i = 0; i < size; i++) {
		f << vec[i] << ", ";
	}
	f << "\b\b}";
}

double cheb_norm(
	double* vec, 
	int size
) {
	double res = 0;
	for (int i = 0; i < size; i++)
	{
		if (res < abs(vec[i]))
			res = abs(vec[i]);
	}
	return res;
}

double eucl_norm(
	double* vec,
	int size
) {
	double res = cblas_ddot(size, vec, 1, vec, 1);
	res = sqrt(res);
	return res;
}

double e_norm(
	double* vec, 
	int size
) {
	double res = 0;
	for (int i = 0; i < size; i++)
	{
		res += vec[i] * vec[i];
	}
	return sqrt(res);;
}

//Метод сопряжённых градиентов с использованием библиотеки MKL
void CG(
	int nrows,                  //число строк в матрице
	sparse_matrix_t& A,         //квадратная с.п.о. матрица в формате csr
	double* u,                  //вектор решения
	double* f,                  //правая часть
	double eps,                 //заданная погрешность
	int itermax,                //максимальное число итераций
	int* number_of_operations   //суммарное число выполненных итераций
)
{
	double* Ap = new double[nrows];
	double* p = new double[nrows];
	double* r = new double[nrows];

	int iter = 0;
	double alpha = 0.0;
	double beta = 0.0;
	double rr = 0;

	matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_UNIT };

	copy(f, f + nrows, r);
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., A, descr, u, 1, r);

	eps *= eps * cblas_ddot(nrows, f, 1, f, 1);
	copy(r, r + nrows, p);
	rr = cblas_ddot(nrows, r, 1, r, 1);
	while (rr > eps && iter < itermax)
	{
		mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, descr, p, 0, Ap);

		alpha = rr / cblas_ddot(nrows, Ap, 1, p, 1);

		cblas_daxpy(nrows, alpha, p, 1, u, 1);
		cblas_daxpy(nrows, -alpha, Ap, 1, r, 1);

		beta = rr;
		rr = cblas_ddot(nrows, r, 1, r, 1);
		beta = rr / beta;

		cblas_daxpby(nrows, 1, r, 1, beta, p, 1);

		iter++;
		*number_of_operations = iter;
	}
	delete[] Ap;
	delete[] p;
	delete[] r;
}

void CR(
	int nrows, 
	sparse_matrix_t& A, 
	double* u, 
	double* f, 
	double eps, 
	int itermax, 
	int* number_of_operations
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
	/*outf.close();*/

	delete[] Ap;
	delete[] Ar;
	delete[] p;
	delete[] r;
}

void CR_Eis(
	int nrows, 
	double* data, 
	int* irow, 
	int* icol, 
	double* u, 
	double* f, 
	double eps, 
	double omega, //релаксационный параметр
	int itermax, 
	int* number_of_operations
)
{
	double* Ap = new double[nrows];
	double* Ar = new double[nrows];
	double* p = new double[nrows];
	double* r = new double[nrows];
	double* tmp = new double[nrows];

	int iter = 0;
	double alpha = 0.0;
	double beta = 0.0;
	double rr = 0;
	double rAr = 0;

	ini_prec_Eis_pf(nrows, data, irow, icol, omega);  //Предобуславливание матрицы
	prec_u_pf(u);   //Предобуславливание вектора решений
	prec_f_pf(f);   //Предобуславливание правой части

	copy(f, f + nrows, r);
	pa_mul_x_pf(u, tmp);
	cblas_daxpby(nrows, -1, tmp, 1, 1, r, 1);

	eps *= eps * cblas_ddot(nrows, f, 1, f, 1);
	copy(r, r + nrows, p);
	rr = cblas_ddot(nrows, r, 1, r, 1);

	pa_mul_x_pf(r, Ar);
	rAr = cblas_ddot(nrows, r, 1, Ar, 1);

	while (rr > eps && iter < itermax)
	{
		pa_mul_x_pf(p, Ap);

		alpha = rAr / cblas_ddot(nrows, Ap, 1, Ap, 1);

		iter++;

		cblas_daxpy(nrows, alpha, p, 1, u, 1);
		cblas_daxpy(nrows, -alpha, Ap, 1, r, 1);

		rr = cblas_ddot(nrows, r, 1, r, 1);
		pa_mul_x_pf(r, Ar);

		beta = rAr;
		rAr = cblas_ddot(nrows, r, 1, Ar, 1);
		beta = rAr / beta;

		cblas_daxpby(nrows, 1, r, 1, beta, p, 1);
		cblas_daxpby(nrows, 1, Ar, 1, beta, Ap, 1);

		*number_of_operations = iter;
	}

	ret_u_pf(u);
	ret_f_pf(f);
	destruct_prec_Eis_pf();

	delete[] Ap;
	delete[] Ar;
	delete[] p;
	delete[] r;
}

void CR_AT(  //Метод сопряжённых невязок с предобуславливанием А^T
	int nrows,                //Размер матрицы
	sparse_matrix_t& A,       //Матрица в формате CSR
	double* u,                //Начальное приближение
	double* f,                //Правая часть
	double eps,               //Точность, критерий остановки
	int itermax,              //Максимальное число итераций, критерий остановки
	int* number_of_operations //Выполненное число итераций
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

	while (rr > eps && iter < itermax)	{
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

void CR_AT_Eis(
	int nrows, 
	double* data, 
	int* irow, 
	int* icol, 
	double* u, 
	double* f, 
	double eps, 
	double omega, 
	int itermax, 
	int* number_of_operations
)
{
	double* Ap = new double[nrows];
	double* Atr = new double[nrows]; // A^T*r^n
	double* p = new double[nrows];
	double* r = new double[nrows];
	double* tmp = new double[nrows];

	int iter = 0;
	double alpha = 0.0;
	double beta = 0.0;
	double rr = 0;       // (r^n, r^n)
	double Atr_p = 0;    // (A^T*r^n, p^n)
	double Atr_Atr = 0;  // (A^T*r^n, A^T*r^n)


	ini_prec_Eis_pf(nrows, data, irow, icol, omega);  //Предобуславливание матрицы
	prec_u_pf(u);   //Предобуславливание вектора решений
	prec_f_pf(f);   //Предобуславливание правой части

	eps *= eps * cblas_ddot(nrows, f, 1, f, 1);
	copy(f, f + nrows, r);
	pa_mul_x_pf(u, tmp);
	cblas_daxpby(nrows, -1, tmp, 1, 1, r, 1);
	rr = cblas_ddot(nrows, r, 1, r, 1);

	pat_mul_x_pf(r, Atr);
	copy(Atr, Atr + nrows, p);
	Atr_p = cblas_ddot(nrows, Atr, 1, p, 1);
	Atr_Atr = cblas_ddot(nrows, Atr, 1, Atr, 1);

	while (rr > eps && iter < itermax) {
		pa_mul_x_pf(p, Ap);
		alpha = Atr_p / cblas_ddot(nrows, Ap, 1, Ap, 1);

		iter++;

		cblas_daxpy(nrows, alpha, p, 1, u, 1);
		cblas_daxpy(nrows, -alpha, Ap, 1, r, 1);

		rr = cblas_ddot(nrows, r, 1, r, 1);
		pat_mul_x_pf(r, Atr);

		beta = Atr_Atr;
		Atr_Atr = cblas_ddot(nrows, Atr, 1, Atr, 1);
		beta = Atr_Atr / beta;

		cblas_daxpby(nrows, 1, Atr, 1, beta, p, 1);
		Atr_p = cblas_ddot(nrows, Atr, 1, p, 1);

		*number_of_operations = iter;
	}

	ret_u_pf(u);
	ret_f_pf(f);
	destruct_prec_Eis_pf();

	delete[] Ap;
	delete[] Atr;
	delete[] p;
	delete[] r;
	delete[] tmp;
}

void csr_cond(
	int nrows, 
	int nelems, 
	int* csr_pos, 
	int* csr_j_index, 
	double* csr_value, 
	double eps
) {
	int i;
	int pm[128];
	mkl_sparse_ee_init(pm);

	sparse_matrix_t A;

	int ret_k;
	double Dmax[1], Dmin[2];
	double* ret_X = new double[2 * nrows];
	double ret_res[2];
	int status;

	struct matrix_descr descs;
	descs.type = SPARSE_MATRIX_TYPE_GENERAL;

	mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, nrows, nrows, csr_pos, csr_pos + 1, csr_j_index, csr_value);

	cout << "singular values:\n";

	char ls_which = 'S';
	char ls_whichV = 'L';
	status = mkl_sparse_d_svd(&ls_which, &ls_whichV, pm, A, descs, 2, &ret_k, Dmin, ret_X, ret_X, ret_res);
	if (!status) {
		cout << "     smallest singular value = " << Dmin[0] << "\n";
		if (Dmin[0] <= eps) {
			cout << "A is degenerate matrix!!\n";
			cout << "     not zero smallest singular value = " << Dmin[1] << "\n";
		}
	}
	else
		cout << "error in mkl_sparse_d_ev, status = " << status << "\n";

	ls_which = 'L';
	ls_whichV = 'L';
	status = mkl_sparse_d_svd(&ls_which, &ls_whichV, pm, A, descs, 1, &ret_k, Dmax, ret_X, ret_X, ret_res);
	if (!status)
		cout << "     larges singular value  " << Dmax[0] << "\n";
	else
		cout << "error in mkl_sparse_d_ev, status = " << status << "\n";

	if (Dmin[0] == 0.0)
		cout << "     cond = " << Dmax[0] / Dmin[1] << "\n\n";
	else
		cout << "     cond = " << Dmax[0] / Dmin[0] << "\n\n";

	mkl_sparse_destroy(A);
	delete[] ret_X;
}

double csr_cond_ret(
	int nrows, 
	int nelems, 
	int* csr_pos, 
	int* csr_j_index, 
	double* csr_value, 
	double eps
) {
	int i;
	int pm[128];
	mkl_sparse_ee_init(pm);

	sparse_matrix_t A;

	int ret_k;
	double cond;
	double Dmax[1], Dmin[2];
	double* ret_X = new double[2 * nrows];
	double ret_res[2];

	struct matrix_descr descs;
	descs.type = SPARSE_MATRIX_TYPE_GENERAL;

	mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, nrows, nrows, csr_pos, csr_pos + 1, csr_j_index, csr_value);

	char ls_which = 'S';
	char ls_whichV = 'L';
	mkl_sparse_d_svd(&ls_which, &ls_whichV, pm, A, descs, 2, &ret_k, Dmin, ret_X, ret_X, ret_res);

	ls_which = 'L';
	ls_whichV = 'L';
	mkl_sparse_d_svd(&ls_which, &ls_whichV, pm, A, descs, 1, &ret_k, Dmax, ret_X, ret_X, ret_res);

	if (Dmin[0] == 0.0)
		cond = Dmax[0] / Dmin[1];
	else
		cond = Dmax[0] / Dmin[0];

	mkl_sparse_destroy(A);
	delete[] ret_X;
	return cond;
}

void csr_cond_ATA(
	int nrows, 
	int nelems, 
	int* csr_pos, 
	int* csr_j_index, 
	double* csr_value, 
	double eps
) {
	int pm[128];
	mkl_sparse_ee_init(pm);

	sparse_matrix_t A, C;

	int ret_k;
	double Dmax[1], Dmin[2];
	double* ret_X = new double[2 * nrows];
	double ret_res[2];
	int status;

	struct matrix_descr descs;
	descs.type = SPARSE_MATRIX_TYPE_GENERAL;

	mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, nrows, nrows, csr_pos, csr_pos + 1, csr_j_index, csr_value);
	mkl_sparse_d_create_csr(&C, SPARSE_INDEX_BASE_ZERO, nrows, nrows, csr_pos, csr_pos + 1, csr_j_index, csr_value);

	matrix_descr descr = { SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_UNIT };

	status = mkl_sparse_spmm(SPARSE_OPERATION_TRANSPOSE, A, A, &C);

	cout << "singular values:\n";

	char ls_which = 'S';
	char ls_whichV = 'L';
	status = mkl_sparse_d_svd(&ls_which, &ls_whichV, pm, C, descs, 2, &ret_k, Dmin, ret_X, ret_X, ret_res);
	if (!status) {
		cout << "     smallest singular value = " << Dmin[0] << "\n";
		if (Dmin[0] <= eps) {
			cout << "A is degenerate matrix!!\n";
			cout << "     not zero smallest singular value = " << Dmin[1] << "\n";
		}
	}
	else
		cout << "error in mkl_sparse_d_ev, status = " << status << "\n";

	ls_which = 'L';
	ls_whichV = 'L';
	status = mkl_sparse_d_svd(&ls_which, &ls_whichV, pm, C, descs, 1, &ret_k, Dmax, ret_X, ret_X, ret_res);
	if (!status)
		cout << "     larges singular value  " << Dmax[0] << "\n";
	else
		cout << "error in mkl_sparse_d_ev, status = " << status << "\n";

	if (Dmin[0] == 0.0)
		cout << "     cond = " << Dmax[0] / Dmin[1] << "\n\n";
	else
		cout << "     cond = " << Dmax[0] / Dmin[0] << "\n\n";

	mkl_sparse_destroy(A);
	mkl_sparse_destroy(C);
	delete[] ret_X;
}

int mult_Ax(
	void* Matr, 
	double* x, 
	double* Ax
) {
	int status;
	sparse_matrix_t& A = (sparse_matrix_t&)Matr;
	status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., A, { SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_UNIT }, x, 0, Ax);
	return status;
}

int multMatrVect(
	void* Matr, 
	double* v, 
	double* Mv
) {
	MatStruct* A = (MatStruct*)Matr;
	for (int i = 0; i <= A->n; i++) {
		Mv[i] = 0;
		for (int j = A->ia[i]; j < A->ia[i + 1]; j++)
			Mv[i] += A->a[j] * v[A->ja[j]];
	}

	return 0;
}

int Init_MatStruct(
	int nrows, 
	int* irow, 
	int* icol, 
	double* data, 
	MatStruct* Str
) {
	Str->n = nrows;
	Str->ia = irow;
	Str->ja = icol;
	Str->a = data;

	return 0;
}