#pragma once
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <mkl.h>
#include <iomanip>
#include <sstream>
#include<array>
using namespace std;

struct COOArrays {
	MKL_INT m;      //< the dimension of the matrix
	MKL_INT nnz;    //< the number of nnz inside the matrix
	double* val;    //< the values (size = nnz)
	MKL_INT* rowind;//< the row indexes (size = nnz)
	MKL_INT* colind;//< the col indexes (size = nnz)

	/** simply set ptr to null */
	COOArrays() {
		val = NULL;
		rowind = NULL;
		colind = NULL;
	}

	/** delete ptr */
	~COOArrays() {
		delete[] val;
		delete[] rowind;
		delete[] colind;
	}
};

struct CRSArrays {
	MKL_INT n;  //< the number of rows
	MKL_INT m;  //< the number of columns
	MKL_INT nnz;//< the number of nnz (== ia[n])
	double* a;  //< the values (of size NNZ)
	MKL_INT* ia;//< the usual rowptr (of size n+1)
	MKL_INT* ja;//< the colidx of each NNZ (of size nnz)

	CRSArrays() {
		a = NULL;
		ia = NULL;
		ja = NULL;
	}

	~CRSArrays() {
	}

	void print() const;
};

struct CRSSystem {
	CRSArrays M; //CRS matrix
	double* f;     //right part
	CRSSystem(){
		f = NULL;
	}
	~CRSSystem() {
	}

};


double eucl_norm(double* vec, MKL_INT size);
void ex_csrsort(MKL_INT n, double* a, MKL_INT* ja);

void ex_convert_COO_2_CSR(MKL_INT n, MKL_INT m, MKL_INT nnz, MKL_INT* coo_ia, MKL_INT* coo_ja, double* coo_a, MKL_INT** csr_ia, MKL_INT** csr_ja, double** csr_a);
double cheb_norm(double* vec, MKL_INT size);

void print(double* vec, MKL_INT size);
void CR(
	MKL_INT nrows,
	sparse_matrix_t& A,
	double* u,
	double* f,
	double eps,
	MKL_INT itermax,
	MKL_INT* number_of_operations
);

void CR_AT(  //Метод сопряжённых невязок с предобуславливанием А^T
	MKL_INT nrows,                //Размер матрицы
	sparse_matrix_t& A,       //Матрица в формате CSR
	double* u,                //Начальное приближение
	double* f,                //Правая часть
	double eps,               //Точность, критерий остановки
	MKL_INT itermax,              //Максимальное число итераций, критерий остановки
	MKL_INT* number_of_operations //Выполненное число итераций
);

CRSArrays get_csr_submatrix(const CRSArrays& A, MKL_INT* rows, MKL_INT* cols, MKL_INT N, MKL_INT M);
int len_file(string filename);


void CR_precond( //only symmetric matrix A
	CRSArrays& A_crs,
	double* u,
	double* f,
	double eps,
	int itermax,
	int* number_of_operations,
	double gamma,
	MKL_INT* p_ind,
	MKL_INT* u_ind,
	MKL_INT len_p,
	MKL_INT len_u

);

void CR_precond_2( //only symmetric matrix A
	CRSArrays& A_crs,
	double* u,
	double* f,
	double eps,
	int itermax,
	int* number_of_operations,
	double gamma,
	MKL_INT* p_ind,
	MKL_INT* u_ind,
	MKL_INT len_p,
	MKL_INT len_u

);

CRSArrays vertical_concatinate(CRSArrays A, CRSArrays B);
CRSArrays horizontal_concatinate(CRSArrays A, CRSArrays B);
CRSSystem precondCCT(CRSSystem A,
	double gamma,
	MKL_INT* p_ind,
	MKL_INT* u_ind,
	MKL_INT len_p,
	MKL_INT len_u);