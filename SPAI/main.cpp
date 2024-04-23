#include <math.h>
#include <stdio.h>
#include "SPAI_func.h"
#include <filesystem>
#include <Sparse>
#include <Eigen>
#include <Dense>
#include "SystemSolver\Sparse_alpha.h"
#include <random>
#include <ctime>


using namespace std;

int test1() { //старый код main
	SystemCOO s_coo("test_matrix.mtx", "test_rp.txt");
	//хочу все переписать на vector...
	vector<double> s_coo_a(s_coo.coo_matrix.nnz);
	vector<int> s_coo_ia(s_coo.coo_matrix.nnz);
	vector<int> s_coo_ja(s_coo.coo_matrix.nnz);
	std::copy(s_coo.coo_matrix.val, s_coo.coo_matrix.val + s_coo.coo_matrix.nnz, s_coo_a.data());
	std::copy(s_coo.coo_matrix.rowind, s_coo.coo_matrix.rowind + s_coo.coo_matrix.nnz, s_coo_ia.data());
	std::copy(s_coo.coo_matrix.colind, s_coo.coo_matrix.colind + s_coo.coo_matrix.nnz, s_coo_ja.data());
	//
	CSCArrays pattern;
	pattern.m = s_coo.coo_matrix.m;
	pattern.n = s_coo.coo_matrix.n;
	pattern.nnz = s_coo.coo_matrix.m * s_coo.coo_matrix.n;
	double* a = new double[pattern.nnz];
	int* ia = new int[pattern.nnz];
	int* ja = new int[pattern.n + 1];
	for (int i = 0; i < pattern.nnz; i++) {
		a[i] = 1;
		ia[i] = i % 4;
	}
	for (int i = 0; i < pattern.n + 1; i++) {
		ja[i] = 4 * i;
	}
	pattern.a = a;
	pattern.ia = ia;
	pattern.ja = ja;
	CSCArrays s_approximatly_inverse = s_coo.SPAI_precond(pattern);
	COOArrays SAI_coo = s_approximatly_inverse.csc_to_coo();

	//хочу все переписать на vector...
	vector<double> SAI_coo_a(SAI_coo.nnz);
	vector<int> SAI_coo_ia(SAI_coo.nnz);
	vector<int> SAI_coo_ja(SAI_coo.nnz);
	std::copy(SAI_coo.val, SAI_coo.val + SAI_coo.nnz, SAI_coo_a.data());
	std::copy(SAI_coo.rowind, SAI_coo.rowind + SAI_coo.nnz, SAI_coo_ia.data());
	std::copy(SAI_coo.colind, SAI_coo.colind + SAI_coo.nnz, SAI_coo_ja.data());
	//

	Eigen::SparseMatrix<double> A = get_eigen_sparse(s_coo_a, s_coo_ia,
		s_coo_ja, s_coo.coo_matrix.n, s_coo.coo_matrix.m);
	Eigen::SparseMatrix<double> SAI = get_eigen_sparse(SAI_coo_a, SAI_coo_ia,
		SAI_coo_ja, SAI_coo.n, SAI_coo.m);
	Eigen::SparseMatrix<double> I(SAI_coo.n, SAI_coo.m);
	I.setIdentity();
	Eigen::SparseMatrix<double> res = I - A * SAI;
	return 0;
}
void test2() { //тест новых алгоритмов конвертации и конструкторов (done)
	int nnz = 4;
	int n = 6;
	int m = 2;
	double a[4]{ 1,1,2,5 };
	int ia[4]{ 1,2,4,4 };
	int ja[4]{ 1,0,0,1 };
	COOMatrix coo(n, m, a, ia, ja, nnz);
	std::cout << "COO: " << endl;
	coo.print();
	std::cout << endl;
	CRSMatrix crs = coo.coo_to_crs();
	std::cout << "CRS: " << endl;
	crs.print();
	std::cout << endl;
	COOMatrix coo2 = crs.crs_to_coo();
	std::cout << "COO2: " << endl;
	coo2.print();
	std::cout << endl;
	CSCMatrix csc = coo2.coo_to_csc();
	std::cout << "CSC: " << endl;
	csc.print();
	std::cout << endl;
	COOMatrix coo3 = csc.csc_to_coo();
	std::cout << "COO3: " << endl;
	coo3.print();
	std::cout << endl;
}
void test3() { //пробуем сконструировать через vector (done)
	int nnz = 4;
	int n = 5;
	int m = 2;
	vector<double> a{ 1,1,2,5 };
	vector<int> ia{ 0,0,1,2,4,4 };
	vector<int> ja{ 1,0,0,1 };
	CRSMatrix crs(n,m,&(a[0]), &(ia[0]), &(ja[0]));
	crs.print();
}
void test4() { //повторим старый код (done)
	int n = 4;
	int m = 4;
	int nnz = 7;
	vector<double> coo_a{ 1,7,1,1,5,1,4 };
	vector<int> coo_ia{ 2,3,3,0,0,1,2 };
	vector<int> coo_ja{ 2,1,3,0,3,1,0 };
	vector<double> rp{ 1,2,3,4 };
	COOMatrix coo(n, m, &(coo_a[0]), &(coo_ia[0]), &(coo_ja[0]), nnz);

	double* a = new double[n * m];
	int* ia = new int[n * m];
	int* ja = new int[n + 1];
	for (int i = 0; i < n * m; i++) {
		a[i] = 1;
		ia[i] = i % 4;
	}
	for (int i = 0; i < n + 1; i++) {
		ja[i] = 4 * i;
	}
	CSCMatrix pattern(n, m, &(a[0]), &(ia[0]), &(ja[0]));
	coo.print();
	pattern.print();
	CSCMatrix s_approximatly_inverse = coo.SPAI_precond(pattern);
	COOMatrix SAI_coo = s_approximatly_inverse.csc_to_coo();
	Eigen::SparseMatrix<double> A = coo.get_eigen_sparse();
	Eigen::SparseMatrix<double> SAI = SAI_coo.get_eigen_sparse();
	Eigen::SparseMatrix<double> I(SAI_coo.n, SAI_coo.m);
	I.setIdentity();
	Eigen::SparseMatrix<double> res = I - A * SAI;
	std::cout << res.norm() << endl;

}
auto dia_to_csr(double* dia_data, int* offset, int N_dia, int N, int M) {//адаптировать
	std::vector<double> csr_data;
	std::vector<int> csr_ia;
	std::vector<int> csr_ja;
	csr_ia.push_back(0);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N_dia; j++) {
			int o = offset[j];
			if (i + o >= 0 && i + o < M) {
				csr_data.push_back(dia_data[i + j * N]);
				csr_ja.push_back(i + o);
			}
		}
		csr_ia.push_back(csr_data.size());
	}
	double* data = new double[csr_data.size()];
	int* ia = new int[csr_ia.size()];
	int* ja = new int[csr_ja.size()];
	std::copy(csr_data.begin(), csr_data.end(), data);
	std::copy(csr_ia.begin(), csr_ia.end(), ia);
	std::copy(csr_ja.begin(), csr_ja.end(), ja);
	return std::make_tuple(ia, ja, data);
}
void test5() { //проверяем SPAI0 на случайных диагональных матрицах

	for (int N = 100; N <= 1e9; N *= 10) {
		std::mt19937_64 gen(12541254124);
		std::uniform_real_distribution<> dis_off(-2.0, -1.0);
		std::uniform_real_distribution<> dis(10.0, 30.0);
		int N_dia = 7;
		double* dia_data = new double[N * N_dia];
		int* dia_offset = new int[N_dia] {-900, -600, -300, 0, 300, 600, 900 };
		for (int j = 0; j < N_dia; j++) {
			for (int i = 0; i < N; i++) {
				dia_data[i + j * N] = dis_off(gen);
			}
		}
		for (int i = 0; i < N; i++) {
			dia_data[i + ((N_dia - 1) / 2) * N] = dis(gen);
		}
		auto [ia, ja, data] = dia_to_csr(dia_data, dia_offset, N_dia, N, N);
		CRSMatrix crs(N, N, data, ia, ja);
		COOMatrix coo = crs.crs_to_coo();

		unsigned int start_time = clock();
		CSCMatrix SAIMat = crs.SPAI_precond(crs.crs_to_csc());
		unsigned int end_time = clock();


		COOMatrix SAI_coo = SAIMat.csc_to_coo();
		Eigen::SparseMatrix<double> A = coo.get_eigen_sparse();
		Eigen::SparseMatrix<double> SAI = SAI_coo.get_eigen_sparse();
		Eigen::SparseMatrix<double> I(SAI_coo.n, SAI_coo.m);
		I.setIdentity();
		Eigen::SparseMatrix<double> res = I - A * SAI;
		std::cout << "N = " << N << ", res = " << res.norm()<< ", t = " << (end_time - start_time)/1000.0 << endl;


	}
}
void test6() {//проверяем SPAI1 на случайных диагональных матрицах
	for (int N = 1000; N <= 1e7; N *= 10) {
		std::mt19937_64 gen(12541254124);
		std::uniform_real_distribution<> dis_off(-2.0, -1.0);
		std::uniform_real_distribution<> dis(10.0, 30.0);
		int N_dia = 7;
		double* dia_data = new double[N * N_dia];
		int* dia_offset = new int[N_dia] {-900, -600, -300, 0, 300, 600, 900 };
		for (int j = 0; j < N_dia; j++) {
			for (int i = 0; i < N; i++) {
				dia_data[i + j * N] = dis_off(gen);
			}
		}
		for (int i = 0; i < N; i++) {
			dia_data[i + ((N_dia - 1) / 2) * N] = dis(gen);
		}
		auto [ia, ja, data] = dia_to_csr(dia_data, dia_offset, N_dia, N, N);
		CRSMatrix crs(N, N, data, ia, ja);
		COOMatrix coo = crs.crs_to_coo();

		unsigned int start_time = clock();
		CSCMatrix SAIMat = crs.SPAI1();
		unsigned int end_time = clock();


		COOMatrix SAI_coo = SAIMat.csc_to_coo();
		Eigen::SparseMatrix<double> A = coo.get_eigen_sparse();
		Eigen::SparseMatrix<double> SAI = SAI_coo.get_eigen_sparse();
		Eigen::SparseMatrix<double> I(SAI_coo.n, SAI_coo.m);
		I.setIdentity();
		Eigen::SparseMatrix<double> res = I - A * SAI;
		std::cout << "N = " << N << ", res = " << res.norm() << ", t = " << (end_time - start_time) / 1000.0 << endl;


	}
}
void test7() {//проверяем SPAI2 на случайных диагональных матрицах
	for (int N = 100; N <= 1e9; N *= 10) {
		std::mt19937_64 gen(12541254124);
		std::uniform_real_distribution<> dis_off(-2.0, -1.0);
		std::uniform_real_distribution<> dis(10.0, 30.0);
		int N_dia = 7;
		double* dia_data = new double[N * N_dia];
		int* dia_offset = new int[N_dia] {-900, -600, -300, 0, 300, 600, 900 };
		for (int j = 0; j < N_dia; j++) {
			for (int i = 0; i < N; i++) {
				dia_data[i + j * N] = dis_off(gen);
			}
		}
		for (int i = 0; i < N; i++) {
			dia_data[i + ((N_dia - 1) / 2) * N] = dis(gen);
		}
		auto [ia, ja, data] = dia_to_csr(dia_data, dia_offset, N_dia, N, N);
		CRSMatrix crs(N, N, data, ia, ja);
		COOMatrix coo = crs.crs_to_coo();

		unsigned int start_time = clock();
		CSCMatrix SAIMat = crs.SPAI2();
		unsigned int end_time = clock();


		COOMatrix SAI_coo = SAIMat.csc_to_coo();
		Eigen::SparseMatrix<double> A = coo.get_eigen_sparse();
		Eigen::SparseMatrix<double> SAI = SAI_coo.get_eigen_sparse();
		Eigen::SparseMatrix<double> I(SAI_coo.n, SAI_coo.m);
		I.setIdentity();
		Eigen::SparseMatrix<double> res = I - A * SAI;
		std::cout << "N = " << N << ", res = " << res.norm() << ", t = " << (end_time - start_time) / 1000.0 << endl;


	}
}
int main() {
	test6();
}