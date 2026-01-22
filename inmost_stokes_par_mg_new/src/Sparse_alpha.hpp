#pragma once
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Dense>
// #include <eigen3/Eigen/Iterator>


using namespace std;
class COOMatrix;
class CSCMatrix;
class CRSMatrix;
class CRS_like_petsc;

void print_message(string message);


class COOMatrix {
private:

public:
	int n;
	int m;      //< the dimension of the matrix
	int nnz;    //< the number of nnz inside the matrix
	std::vector<double> a;    //< the values (size = nnz)
	std::vector<int> ia;//< the row indexes (size = nnz)
	std::vector<int> ja;//< the col indexes (size = nnz)
	void print() const;
	CRSMatrix coo_to_crs();
	CSCMatrix coo_to_csc();
	COOMatrix(int n, int m, double* a, const int* ia, const int* ja, int nnz);
	COOMatrix();
	CSCMatrix SPAI_precond(CSCMatrix pattern);
	CSCMatrix SPAI0();
	CSCMatrix SPAI1();
	CSCMatrix SPAI2();
	int _sort_type; // 0 - ia, 1 - ja, -1 - isn't sort
	void sort_ia();
	void sort_ja();
	Eigen::SparseMatrix<double> get_eigen_sparse();
	CRSMatrix compress_CRS(vector<int> garray);
	CRS_like_petsc generate_like_petsc(int* narray, int rank);
};

class CRSMatrix {
private:
public:
	int n;  //< the number of rows
	int m;  //< the number of columns
	int nnz;//< the number of nnz (== ia[n])
	std::vector<double> a;  //< the values (of size NNZ)
	std::vector<int> ia;//< the usual rowptr (of size n+1)
	std::vector<int> ja;//< the colidx of each NNZ (of size nnz)
	void print() const;
	CRSMatrix(int n, int m, double* a, const int* ia, const int* ja);
	CRSMatrix();
	COOMatrix crs_to_coo();
	CSCMatrix crs_to_csc();
	CSCMatrix SPAI_precond(CSCMatrix pattern);
	CSCMatrix SPAI0();
	CSCMatrix SPAI1();
	CSCMatrix SPAI2();
	CRSMatrix get_csr_submatrix(int* rows, int* cols, int N, int M);
	CRS_like_petsc generate_like_petsc(int* narray, int rank);
	CRSMatrix compress_CRS(vector<int> garray);

};
class CSCMatrix {
private:
public:
	int n;  //< the number of rows
	int m;  //< the number of columns
	int nnz;//< the number of nnz (== ia[n])
	std::vector<double> a;  //< the values (of size NNZ)
	std::vector<int> ia;//< the usual rowptr (of size nnz)
	std::vector<int> ja;//< the colidx of each NNZ (of size m+1)
	void print() const;
	COOMatrix csc_to_coo();
	CRSMatrix csc_to_crs();
	CSCMatrix(int n, int m, double* a, int* ia, int* ja);
	CSCMatrix SPAI_precond(CSCMatrix pattern);
	CSCMatrix SPAI0();
	CSCMatrix SPAI1();
	CSCMatrix SPAI2();

};


class CRS_like_petsc{
	private:
	public:
	CRSMatrix* Diag;
	CRSMatrix* Off_diag_Cpr;
	std::vector<int> garray;
	int first_row; 
	int n_global;
	int m_global;
	int rank;
	CRS_like_petsc();
	CRS_like_petsc(int n_loc, double* D_a, const int* D_ia, const int* D_ja, 
				   int m_off_loc, double* Off_a, const int* Off_ia, const int* Off_ja, 
				   const int* garr, int first_row, int n_global, int m_global, int rank);
	CRS_like_petsc(string test_matrix, int first_row, int n_global, int m_global, int rank);
	CRSMatrix spai_dbo();
	std::vector<int> get_narray();
	int get_min_n_loc(std::vector<int> narray);
	int get_max_n_loc(std::vector<int> narray);
	CRSMatrix get_A(std::vector<int> J, std::vector<int> narray);
	CRS_like_petsc SPAI_old(CRS_like_petsc pattern);
	CRS_like_petsc SPAI(CRS_like_petsc pattern);

};

int find_value(const std::vector<int>& data, int value);