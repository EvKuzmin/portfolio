#pragma once
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <Sparse>
#include <Eigen>
#include <Dense>
#include <iterator>


using namespace std;
class SystemCOO;
class SystemCSC;
class SystemCRS;
class CRSArrays;
class COOArrays;
class CSCArrays;

class COOArrays {
private:

public:
	int n;
	int m;      //< the dimension of the matrix
	int nnz;    //< the number of nnz inside the matrix
	double* val;    //< the values (size = nnz)
	int* rowind;//< the row indexes (size = nnz)
	int* colind;//< the col indexes (size = nnz)

	/** simply set ptr to null */
	COOArrays();
};

class CRSArrays {
private:
public:
	int n;  //< the number of rows
	int m;  //< the number of columns
	int nnz;//< the number of nnz (== ia[n])
	double* a;  //< the values (of size NNZ)
	int* ia;//< the usual rowptr (of size n+1)
	int* ja;//< the colidx of each NNZ (of size nnz)

	CRSArrays();
	CRSArrays get_csr_submatrix(int* rows, int* cols, int N, int M);

	void print_crs() const;
};
class CSCArrays {
private:
public:
	int n;  //< the number of rows
	int m;  //< the number of columns
	int nnz;//< the number of nnz (== ia[n])
	double* a;  //< the values (of size NNZ)
	int* ia;//< the usual rowptr (of size nnz)
	int* ja;//< the colidx of each NNZ (of size m+1)
	COOArrays csc_to_coo();
	CSCArrays();
};

class System {
private:
public:
	vector<double> rp;
	vector<double> load_rp(string filename);
	virtual CSCArrays SPAI_precond(CSCArrays pattern) = 0;

};


class SystemCRS :public System {
private:
public:
	CRSArrays crs_matrix;
	SystemCOO crs_to_coo();
	SystemCSC crs_to_csc();
	SystemCRS(int n, int m, int nnz, vector<double> rp, vector<double> a, vector<int> ia, vector<int> ja);
	CSCArrays SPAI_precond(CSCArrays pattern);
};
class SystemCSC :public System {
private:
public:
	SystemCSC(int n, int m, int nnz, vector<double> rp, vector<double> a, vector<int> ia, vector<int> ja);
	CSCArrays csc_matrix;
	SystemCOO csc_to_coo();
	CSCArrays SPAI_precond(CSCArrays pattern);
	SystemCRS csc_to_crs();
};

class SystemCOO :public System {
private:
public:
	SystemCOO(string matrix_name, string rp_name);
	SystemCOO(int n, int m, int nnz, vector<double> rprt,
		vector<double> a, vector<int> ia, vector<int> ja);
	int _sort_type; // 0 - ia, 1 - ja, -1 - isn't sort
	void sort_ia();
	void sort_ja();
	COOArrays coo_matrix;
	SystemCRS coo_to_crs();
	SystemCSC coo_to_csc();
	COOArrays load_coo_mtx(string filename);
	void read_system(string matrix, string right_part);
	CSCArrays SPAI_precond(CSCArrays pattern);
};



Eigen::SparseMatrix<double> get_eigen_sparse(vector<double> coo_a, vector<int> coo_ia, vector<int> coo_ja, int n, int m);