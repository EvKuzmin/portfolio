#include <mpi.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <mpi.h>
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/Dense>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <set>
#include "Sparse_alpha.hpp"

#include <thread>

using namespace std;
typedef Eigen::Triplet<double> T;
COOMatrix::COOMatrix(int n, int m, double* a, const int* ia, const int* ja, int nnz) {
	this->n = n;
	this->m = m;
	this->nnz = nnz;
	this->a.reserve(nnz);
	this->ia.reserve(nnz);
	this->ja.reserve(nnz);
	for (int i = 0; i < nnz; i++) {
		this->a.push_back(*(a + i));
		this->ia.push_back(*(ia + i));
		this->ja.push_back(*(ja + i));
	}
	_sort_type = -1;
}
CRSMatrix::CRSMatrix(int n, int m, double* a, const int* ia, const int* ja) {
	this->n = n;
	this->m = m;
	this->nnz = *(ia+n);
	this->a.reserve(this->nnz);
	this->ia.reserve(n + 1);
	this->ja.reserve(this->nnz);
	for (int i = 0; i < this->nnz; i++) {
		this->a.push_back(*(a + i));
		this->ja.push_back(*(ja + i));
	}
	for (int i = 0; i < this->n + 1; i++) {
		this->ia.push_back(*(ia + i));
	}
}
CSCMatrix::CSCMatrix(int n, int m, double* a, int* ia, int* ja) {
	this->n = n;
	this->m = m;
	this->nnz = *(ja + m);
	this->a.reserve(this->nnz);
	this->ia.reserve(this->nnz);
	this->ja.reserve(m + 1);
	for (int i = 0; i < this->nnz; i++) {
		this->a.push_back(*(a + i));
		this->ia.push_back(*(ia + i));
	}
	for (int i = 0; i < this->m + 1; i++) {
		this->ja.push_back(*(ja + i));
	}
}

void COOMatrix::print() const {
	std::cout << "a:  ";
	for (int i = 0; i < this->nnz; i++) {
		std::cout << this->a[i] << ", ";
	}
	std::cout << endl;
	std::cout << "ia:  ";
	for (int i = 0; i < this->nnz; i++) {
		std::cout << this->ia[i] << ", ";
	}
	std::cout << endl;
	std::cout << "ja:  ";
	for (int i = 0; i < this->nnz; i++) {
		std::cout << this->ja[i] << ", ";
	}
	std::cout << endl;
}
void CRSMatrix::print() const {
	std::cout << "a:  ";
	for (int i = 0; i < this->nnz; i++) {
		std::cout << this->a[i] << ", ";
	}
	std::cout << endl;
	std::cout << "ia:  ";
	for (int i = 0; i < this->n + 1; i++) {
		std::cout << this->ia[i] << ", ";
	}
	std::cout << endl;
	std::cout << "ja:  ";
	for (int i = 0; i < this->nnz; i++) {
		std::cout << this->ja[i] << ", ";
	}
	std::cout << endl;
}
void CSCMatrix::print() const {
	std::cout << "a:  ";
	for (int i = 0; i < this->nnz; i++) {
		std::cout << this->a[i] << ", ";
	}
	std::cout << endl;
	std::cout << "ia:  ";
	for (int i = 0; i < this->nnz; i++) {
		std::cout << this->ia[i] << ", ";
	}
	std::cout << endl;
	std::cout << "ja:  ";
	for (int i = 0; i < this->m + 1; i++) {
		std::cout << this->ja[i] << ", ";
	}
	std::cout << endl;
}

void COOMatrix::sort_ia() {
	std::vector<int> indexes(this->nnz);
	std::iota(indexes.begin(), indexes.end(), 0);
	auto compare = [this](int a, int b) {
		if (this->ia[a] < this->ia[b]) {
			return true;
		}
		else if (this->ia[a] == this->ia[b]) {
			if (this->ja[a] < this->ja[b]) return true;
			else return false;
		}
		else return false;
	};
	std::sort(indexes.begin(), indexes.end(), compare);
	std::vector<double> _a(this->nnz);
	std::vector<int> _ia(this->nnz);
	std::vector<int> _ja(this->nnz);

	for (int i = 0; i < this->nnz; i++) {
		_a[i] = this->a[indexes[i]];
		_ia[i] = this->ia[indexes[i]];
		_ja[i] = this->ja[indexes[i]];
	}
	std::copy(_a.begin(), _a.end(), this->a.begin());
	std::copy(_ia.begin(), _ia.end(), this->ia.begin());
	std::copy(_ja.begin(), _ja.end(), this->ja.begin());
	_sort_type = 0;

}
void COOMatrix::sort_ja() {
	std::vector<int> indexes(this->nnz);
	std::iota(indexes.begin(), indexes.end(), 0);
	auto compare = [this](int a, int b) {
		if (this->ja[a] < this->ja[b]) {
			return true;
		}
		else if (this->ja[a] == this->ja[b]) {
			if (this->ia[a] < this->ia[b]) return true;
			else return false;
		}
		else return false;
	};
	std::sort(indexes.begin(), indexes.end(), compare);
	std::vector<double> _a(this->nnz);
	std::vector<int> _ia(this->nnz);
	std::vector<int> _ja(this->nnz);

	for (int i = 0; i < this->nnz; i++) {
		_a[i] = this->a[indexes[i]];
		_ia[i] = this->ia[indexes[i]];
		_ja[i] = this->ja[indexes[i]];
	}
	std::copy(_a.begin(), _a.end(), this->a.begin());
	std::copy(_ia.begin(), _ia.end(), this->ia.begin());
	std::copy(_ja.begin(), _ja.end(), this->ja.begin());
	_sort_type = 1;
}

CRSMatrix COOMatrix::coo_to_crs() {
	sort_ia();
	vector<int> pre_ia;
	pre_ia.reserve(this->n + 1);
	vector<int> _ia(this->n + 1);
	int count = 0;
	int n_str = 0;
	pre_ia.push_back(0);
	while (n_str != this->ia[0]) {
		pre_ia.push_back(0);
		n_str++;
	}
	for (int i = 0; i < this->nnz; i++) {
		if (n_str == this->ia[i]) {
			count++;
		}
		else {
			pre_ia.push_back(count);
			n_str++;
			count = 0;
			while (n_str != this->ia[i]) {
				pre_ia.push_back(count);
				n_str++;
			}
			count = 1;
		}
	}
	pre_ia.push_back(count);
	n_str++;
	for (int i = n_str + 1; i < n + 1; i++) {
		pre_ia.push_back(0);
	}
	std::partial_sum(pre_ia.begin(), pre_ia.end(), _ia.begin(), plus<double>());
	CRSMatrix crs(this->n, this->m, &(this->a[0]), &(_ia[0]), &(this->ja[0]));
	return crs;
}
CSCMatrix COOMatrix::coo_to_csc() {
	sort_ja();
	vector<int> pre_ja;
	pre_ja.reserve(this->m + 1);
	vector<int> _ja(this->m + 1);
	int count = 0;
	int n_str = 0;
	pre_ja.push_back(0);
	while (n_str != this->ja[0]) {
		pre_ja.push_back(0);
		n_str++;
	}
	for (int i = 0; i < this->nnz; i++) {
		if (n_str == this->ja[i]) {
			count++;
		}
		else {
			pre_ja.push_back(count);
			n_str++;
			count = 0;
			while (n_str != this->ja[i]) {
				pre_ja.push_back(count);
				n_str++;
			}
			count = 1;
		}
	}
	pre_ja.push_back(count);
	n_str++;
	for (int i = n_str + 1; i < this->m + 1; i++) {
		pre_ja.push_back(0);
	}
	std::partial_sum(pre_ja.begin(), pre_ja.end(), _ja.begin(), plus<double>());
	CSCMatrix csc(this->n, this->m, &(this->a[0]), &(this->ia[0]), &(_ja[0]));
	return csc;
}
COOMatrix CRSMatrix::crs_to_coo() {
	std::vector<int> _ia;
	_ia.reserve(this->nnz);
	for (int i = 0; i < this->n; i++) {
		for (int j = this->ia[i]; j < this->ia[i + 1]; j++) {
			_ia.push_back(i);
		}
	}
	COOMatrix coo(this->n, this->m, &(this->a[0]), &(_ia[0]), &(this->ja[0]), this->nnz);
	coo._sort_type = 0;
	return coo;
}
COOMatrix CSCMatrix::csc_to_coo() {
	std::vector<int> _ja;
	_ja.reserve(this->nnz);

	for (int i = 0; i < this->m; i++) {
		for (int j = this->ja[i]; j < this->ja[i + 1]; j++) {
			_ja.push_back(i);
		}
	}
	COOMatrix coo(this->n, this->m, &(this->a[0]), &(this->ia[0]), &(_ja[0]), this->nnz);
	coo._sort_type = 1;
	return coo;
}
CSCMatrix CRSMatrix::crs_to_csc() {
	return this->crs_to_coo().coo_to_csc();
}
CRSMatrix CSCMatrix::csc_to_crs() {
	return this->csc_to_coo().coo_to_crs();
}

CRSMatrix CRSMatrix::get_csr_submatrix(int* rows, int* cols, int N, int M) {
	std::vector<int> ia_new(N + 1, 0);
	std::vector<int> ja_new;
	ja_new.reserve(N * M);
	std::vector<double> a_new;
	a_new.reserve(N * M);
	int n = 0;
	for (int i = 0; i < N; i++) {
		int row = rows[i];
		for (int j = this->ia[row]; j < this->ia[row + 1]; j++) {
			int col = this->ja[j];
			auto it = std::lower_bound(cols, cols + M, col);
			if (it != cols + M)
				if ((*it) == col) {
					n++;
					ja_new.push_back(it - cols);
					a_new.push_back(this->a[j]);
				}
		}
		ia_new[i + 1] = n;
	}
	CRSMatrix crs(N, M, &(a_new[0]), &(ia_new[0]), &(ja_new[0]));
	return crs;
}

CSCMatrix COOMatrix::SPAI_precond(CSCMatrix pattern) {
	return this->coo_to_csc().SPAI_precond(pattern);
}
CSCMatrix CRSMatrix::SPAI_precond(CSCMatrix pattern) {
	return this->crs_to_csc().SPAI_precond(pattern);
}
CSCMatrix CSCMatrix::SPAI_precond(CSCMatrix pattern) {
	std::vector<double> a;
	a.reserve(pattern.nnz);
	std::vector<int> ia;
	ia.reserve(pattern.nnz);
	std::vector<int> ja;
	ja.reserve(pattern.m + 1);
	int count = 0;
	ja.push_back(count);
	// get AT in CRS
	CRSMatrix AT(this->m, this->n, &(this->a[0]), &(this->ja[0]), &(this->ia[0]));
	for (int j = 1; j <= pattern.m; j++) {
		//J_j
		std::vector<int> J;
		for (int m = pattern.ja[j - 1]; m < pattern.ja[j]; m++) {
			J.push_back(pattern.ia[m]);
		}
		//I_j
		std::set<int> I_s;
		for (auto it = J.begin(); it != J.end(); it++) {
			for (int m = this->ja[*it]; m < this->ja[*it + 1]; m++) {
				I_s.insert(this->ia[m]);
			}
		}
		std::vector<int> I(I_s.begin(), I_s.end());
		// get AT_sub
		CRSMatrix AT_sub = AT.get_csr_submatrix(&(J[0]), &(I[0]), J.size(), I.size());
		Eigen::MatrixXd AT_sub_mat = Eigen::MatrixXd::Zero(J.size(), I.size());
		for (int i = 0; i < AT_sub.n; i++) {
			for (int j = AT_sub.ia[i]; j < AT_sub.ia[i + 1]; j++) {
				AT_sub_mat(i, AT_sub.ja[j]) = AT_sub.a[j];
			}
		}
		Eigen::MatrixXd D = 2 * AT_sub_mat * AT_sub_mat.transpose();
		Eigen::VectorXd e = Eigen::VectorXd::Zero(AT_sub.m);
		{																	//���������?
			int counter = 0;
			for (auto it = I.begin(); it != I.end(); it++) {
				if (*it == j - 1) {
					e(counter) = 1;
				}
				counter++;
			}
		}
		Eigen::VectorXd b = 2 * AT_sub_mat * e;
		Eigen::VectorXd m = D.householderQr().solve(b); //������� ������ ������?
		// Eigen::VectorXd m = D.ldlt().solve(b);
		for (int i = 0; i < m.size(); i++) {
			a.push_back(m(i));
			ia.push_back(J[i]);
		}
		count += m.size();
		ja.push_back(count);
	}
	CSCMatrix csc(this->n, this->m, &(a[0]), &(ia[0]), &(ja[0]));
	return csc;
}

CSCMatrix COOMatrix::SPAI0() {
	return this->SPAI_precond(this->coo_to_csc());
}
CSCMatrix CRSMatrix::SPAI0() {
	return this->SPAI_precond(this->crs_to_csc());
}
CSCMatrix CSCMatrix::SPAI0() {
	return this->SPAI_precond(*this);
}

CSCMatrix COOMatrix::SPAI1() {
	return this->coo_to_csc().SPAI1();
}
CSCMatrix CRSMatrix::SPAI1() {
	return this->crs_to_csc().SPAI1();
}
CSCMatrix CSCMatrix::SPAI1() {
	std::vector<double> a;
	a.reserve(this->nnz);
	std::vector<int> ia;
	ia.reserve(this->nnz);
	std::vector<int> ja;
	ja.reserve(this->m + 1);
	int count = 0;
	ja.push_back(count);
	// get AT in CRS
	CRSMatrix AT(this->m, this->n, &(this->a[0]), &(this->ja[0]), &(this->ia[0]));
	// get A in CRS
	CRSMatrix A = this->csc_to_crs();
	for (int j = 1; j <= this->m; j++) {
		////J_j
		//std::vector<int> Js;
		//std::vector<int> J;
		//for (int m = this->ja[j - 1]; m < this->ja[j]; m++) {
		//	Js.push_back(this->ia[m]);
		//}
		//for (int i = 0; i < this->n; i++) {
		//	for (int m = A.ia[i]; m < A.ia[i + 1]; m++) {
		//		if (std::find(Js.begin(), Js.end(), A.ja[m]) != Js.end()) { /*A.ja[m] ���� � Js*/
		//			J.push_back(A.ja[m]);
		//			break;
		//		}
		//	}
		//}
		//
		std::vector<int> Js;
		for (int m = this->ja[j - 1]; m < this->ja[j]; m++) {
			Js.push_back(this->ia[m]);
		}
		//J_j
		std::set<int> J_s;
		for (auto it = Js.begin(); it != Js.end(); it++) {
			for (int m = this->ja[*it]; m < this->ja[*it + 1]; m++) {
				J_s.insert(this->ia[m]);
			}
		}
		std::vector<int> J(J_s.begin(), J_s.end());
		//I_j
		std::set<int> I_s;
		for (auto it = J.begin(); it != J.end(); it++) {
			for (int m = this->ja[*it]; m < this->ja[*it + 1]; m++) {
				I_s.insert(this->ia[m]);
			}
		}
		std::vector<int> I(I_s.begin(), I_s.end());
		// get AT_sub
		CRSMatrix AT_sub = AT.get_csr_submatrix(&(J[0]), &(I[0]), J.size(), I.size());
		Eigen::MatrixXd AT_sub_mat = Eigen::MatrixXd::Zero(J.size(), I.size());
		for (int i = 0; i < AT_sub.n; i++) {
			for (int j = AT_sub.ia[i]; j < AT_sub.ia[i + 1]; j++) {
				AT_sub_mat(i, AT_sub.ja[j]) = AT_sub.a[j];
			}
		}
		Eigen::MatrixXd D = 2 * AT_sub_mat * AT_sub_mat.transpose();
		Eigen::VectorXd e = Eigen::VectorXd::Zero(AT_sub.m);
		{																	//���������?
			int counter = 0;
			for (auto it = I.begin(); it != I.end(); it++) {
				if (*it == j - 1) {
					e(counter) = 1;
				}
				counter++;
			}
		}
		Eigen::VectorXd b = 2 * AT_sub_mat * e;
		Eigen::VectorXd m = D.householderQr().solve(b); //������� ������ ������?
		// Eigen::VectorXd m = D.ldlt().solve(b);
		for (int i = 0; i < m.size(); i++) {
			a.push_back(m(i));
			ia.push_back(J[i]);
		}
		count += m.size();
		ja.push_back(count);
	}
	CSCMatrix csc(this->n, this->m, &(a[0]), &(ia[0]), &(ja[0]));
	return csc;
}

CSCMatrix COOMatrix::SPAI2() {
	return this->coo_to_csc().SPAI2();
}
CSCMatrix CRSMatrix::SPAI2() {
	return this->crs_to_csc().SPAI2();
}
CSCMatrix CSCMatrix::SPAI2() {
	std::vector<double> a;
	a.reserve(this->nnz);
	std::vector<int> ia;
	ia.reserve(this->nnz);
	std::vector<int> ja;
	ja.reserve(this->m + 1);
	int count = 0;
	ja.push_back(count);
	// get AT in CRS
	CRSMatrix AT(this->m, this->n, &(this->a[0]), &(this->ja[0]), &(this->ia[0]));
	for (int j = 1; j <= this->m; j++) {
		//J_j
		std::vector<int> Js;
		std::vector<int> J;
		for (int m = this->ja[j - 1]; m < this->ja[j]; m++) {
			Js.push_back(this->ia[m]);
		}
		for (int i = 0; i < this->n; i++) {
			for (int m = AT.ia[i]; m < AT.ia[i + 1]; m++) {
				if (std::find(Js.begin(), Js.end(), AT.ja[m]) != Js.end()) { /*A.ja[m] ���� � Js*/
					J.push_back(AT.ja[m]);
					break;
				}
			}
		}

		//I_j
		std::set<int> I_s;
		for (auto it = J.begin(); it != J.end(); it++) {
			for (int m = this->ja[*it]; m < this->ja[*it + 1]; m++) {
				I_s.insert(this->ia[m]);
			}
		}
		std::vector<int> I(I_s.begin(), I_s.end());
		// get AT_sub
		CRSMatrix AT_sub = AT.get_csr_submatrix(&(J[0]), &(I[0]), J.size(), I.size());
		Eigen::MatrixXd AT_sub_mat = Eigen::MatrixXd::Zero(J.size(), I.size());
		for (int i = 0; i < AT_sub.n; i++) {
			for (int j = AT_sub.ia[i]; j < AT_sub.ia[i + 1]; j++) {
				AT_sub_mat(i, AT_sub.ja[j]) = AT_sub.a[j];
			}
		}
		Eigen::MatrixXd D = 2 * AT_sub_mat * AT_sub_mat.transpose();
		Eigen::VectorXd e = Eigen::VectorXd::Zero(AT_sub.m);
		{																	//���������?
			int counter = 0;
			for (auto it = I.begin(); it != I.end(); it++) {
				if (*it == j - 1) {
					e(counter) = 1;
				}
				counter++;
			}
		}
		Eigen::VectorXd b = 2 * AT_sub_mat * e;
		Eigen::VectorXd m = D.householderQr().solve(b); //������� ������ ������?
		// Eigen::VectorXd m = D.ldlt().solve(b);
		for (int i = 0; i < m.size(); i++) {
			a.push_back(m(i));
			ia.push_back(J[i]);
		}
		count += m.size();
		ja.push_back(count);
	}
	CSCMatrix csc(this->n, this->m, &(a[0]), &(ia[0]), &(ja[0]));
	return csc;

}

Eigen::SparseMatrix<double> COOMatrix::get_eigen_sparse() {
	std::vector<T> triplets;
	triplets.reserve(this->ia.size());
	for (int i = 0; i < this->ia.size(); i++) {
		triplets.push_back(T(this->ia[i], this->ja[i], this->a[i]));
	}
	Eigen::SparseMatrix<double> M(this->n, this->m);
	M.setFromTriplets(triplets.begin(), triplets.end());
	return M;
}

CRS_like_petsc::CRS_like_petsc(){
	Diag = nullptr;
	Off_diag_Cpr = nullptr;
}
COOMatrix::COOMatrix(){
	n = 0;
	m = 0;      
	nnz = 0;
}
CRSMatrix::CRSMatrix(){
	n = 0;
	m = 0;      
	nnz = 0;
}
CRS_like_petsc::CRS_like_petsc(string test_matrix, int first_row, int n_global, int m_global, int rank) {
	this->rank = rank;
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	//Diag read begin
	vector<double> DiagA;
	vector<int> DiagIA;
	vector<int> DiagJA;
	{
		int nnz = 0;
		int n = -1;
		{
			string line;
			fstream g(test_matrix + "A_diag_data_debug_rank_" + std::to_string(rank) + ".txt");
			if (!g) {
				std::cerr << "нет файла diag_data_debug_" + std::to_string(rank) << endl;
				std::exit(1);
			}
			
			while (getline(g, line)) {
				nnz++;
			}
			g.close();
			DiagA.resize(nnz);
			fstream f(test_matrix + "A_diag_data_debug_rank_" + std::to_string(rank) + ".txt");
			{
				int i = 0;
				while (getline(f, line))
				{
					DiagA[i] = stod(line);
					i++;
				}
			}
			f.close();
		}
		{
			string line;
			fstream g(test_matrix + "A_diag_ia_debug_rank_" + std::to_string(rank) + ".txt");
			if (!g) {
				std::cerr << "нет файла diag_ia_debug" << endl;
				std::exit(1);
			}
			
			while (getline(g, line)) {
				n++;
			}
			g.close();
			DiagIA.resize(n+1);
			fstream f(test_matrix + "A_diag_ia_debug_rank_" + std::to_string(rank) + ".txt");
			{
				int i = 0;
				while (getline(f, line))
				{
					DiagIA[i] = stoi(line);
					i++;
				}
			}
			f.close();
		}
		{
			string line;
			DiagJA.resize(nnz);
			fstream f(test_matrix + "A_diag_ja_debug_rank_" + std::to_string(rank) + ".txt");
			{
				int i = 0;
				while (getline(f, line))
				{
					DiagJA[i] = stoi(line);
					i++;
				}
			}
			f.close();

		}
		this->Diag = new CRSMatrix(n, n, &(DiagA[0]), &(DiagIA[0]), &(DiagJA[0]));
	}
	//Diag read end
	//Off_diag read begin
	vector<double> Off_diag_CprA;
	vector<int> Off_diag_CprIA;
	vector<int> Off_diag_CprJA;
	int nnz = 0;
	int n = -1;
	{
		string line;
		fstream g(test_matrix + "A_off_diag_data_debug_rank_" + std::to_string(rank) + ".txt");
		if (!g) {
			std::cerr << "нет файла off_diag_data_debug" << endl;
			std::exit(1);
		}
		
		while (getline(g, line)) {
			nnz++;
		}
		g.close();
		Off_diag_CprA.resize(nnz);
		fstream f(test_matrix + "A_off_diag_data_debug_rank_" + std::to_string(rank) + ".txt");
		{
			int i = 0;
			while (getline(f, line))
			{
				Off_diag_CprA[i] = stod(line);
				i++;
			}
		}
		f.close();
	}
	{
		string line;
		fstream g(test_matrix + "A_off_diag_ia_debug_rank_" + std::to_string(rank) + ".txt");
		if (!g) {
			std::cerr << "нет файла off_diag_ia_debug" << endl;
			std::exit(1);
		}
		
		while (getline(g, line)) {
			n++;
		}
		g.close();
		Off_diag_CprIA.resize(n+1);
		fstream f(test_matrix + "A_off_diag_ia_debug_rank_" + std::to_string(rank) + ".txt");
		{
			int i = 0;
			while (getline(f, line))
			{
				Off_diag_CprIA[i] = stoi(line);
				i++;
			}
		}
		f.close();
	}
	{
		string line;
		Off_diag_CprJA.resize(nnz);
		fstream f(test_matrix + "A_off_diag_ja_debug_rank_" + std::to_string(rank) + ".txt");
		{
			int i = 0;
			while (getline(f, line))
			{
				Off_diag_CprJA[i] = stoi(line);
				i++;
			}
		}
		f.close();

	}
	//garray read
	int kl;
	{
		string line;
		fstream g(test_matrix + "A_off_diag_garray_debug_rank_" + std::to_string(rank) + ".txt");
		if (!g) {
			std::cerr << "нет файла off_diag_garray_debug" << endl;
			std::exit(1);
		}
		kl = 0;
		while (getline(g, line)) {
			kl++;
		}
		g.close();
		garray.resize(kl);
		fstream f(test_matrix + "A_off_diag_garray_debug_rank_" + std::to_string(rank) + ".txt");
		{
			int i = 0;
			while (getline(f, line))
			{

				garray[i] = stoi(line);
				i++;
			}
		}
		f.close();

	}
	this->Off_diag_Cpr = new CRSMatrix(n, kl, &(Off_diag_CprA[0]), &(Off_diag_CprIA[0]), &(Off_diag_CprJA[0]));
	//Off_diag read end
	this->first_row = first_row;
	this->n_global = n_global;
	this->m_global = m_global;
}

CRS_like_petsc::CRS_like_petsc(int n_loc, double* D_a, const int* D_ia, const int* D_ja, 
				   int m_off_loc, double* Off_a, const int* Off_ia, const int* Off_ja, 
				   const int* garr, int first_row, int n_global, int m_global, int rank) {
	this->Diag = new CRSMatrix(n_loc, n_loc, D_a, D_ia, D_ja);
	this->Off_diag_Cpr = new CRSMatrix(n_loc, m_off_loc, Off_a, Off_ia, Off_ja);
	this->garray.resize(m_off_loc);
	for (int i = 0; i < m_off_loc; i++) this->garray[i] = garr[i];
	this->first_row = first_row;
	this->n_global = n_global;
	this->m_global = m_global;	
	this->rank = rank;
}

void print_message(string message){
	std::cout << message << "\n";
}
int find_value(const std::vector<int>& data, int value){
    // auto result{ std::find(begin(data), end(data), value) };
    // if (result == end(data))
    //     return -1;
    // else
    //     return (result - begin(data));
	auto result{ std::lower_bound(begin(data), end(data), value) };
	if (result != end(data) && (*result) == value) {
		return (result - begin(data));
	} else {
		return -1;
	}
}

int find_value_old(const std::vector<int>& data, int value){
    auto result{ std::find(begin(data), end(data), value) };
    if (result == end(data))
        return -1;
    else
        return (result - begin(data));
}


CRSMatrix CRS_like_petsc::spai_dbo(){ //получаем на процессе Diag
	std::vector<double> a;
	a.reserve(Diag->nnz);
	std::vector<int> ia;
	ia.reserve(Diag->n + 1);
	std::vector<int> ja;
	ja.reserve(Diag->nnz);
	int count = 0;
	ia.push_back(count);

	for (int j = 0; j < Diag->n; j++) {
		std::vector<double> A_a;
		std::vector<int> A_ia;
		std::vector<int> A_ja;
		int A_nnz = 0;
		//J
		std::vector<int> J;
		for(int i = Diag->ia[j]; i<Diag->ia[j+1]; i++ ){
			J.push_back(Diag->ja[i]);
		}

		//I
		//I_Diag в локальной нумерации
		std::set<int> I_Diag_loc;																	
		for (auto it = J.begin(); it != J.end(); it++) {
			for (int m = Diag->ia[*it]; m < Diag->ia[*it + 1]; m++) {
				I_Diag_loc.insert(Diag->ja[m]);
				//сразу добавляем в A^ COO
				A_a.push_back(Diag->a[m]);
				A_ia.push_back(*it);
				A_ja.push_back(Diag->ja[m] + this->first_row);
				A_nnz++;

			}
		}


		//I_Diag в глобальной нумерации ????
		std::vector<int> I_Diag;
		I_Diag.reserve(I_Diag_loc.size());
		std::copy(I_Diag_loc.begin(), I_Diag_loc.end(), std::back_inserter(I_Diag));



		for(int i = 0; i < I_Diag_loc.size(); i++){
			I_Diag[i] = I_Diag[i] + this->first_row;
		}

		std::set<int> I_Off_diag_loc;
		std::vector<int> I_Off_diag;
		if (Off_diag_Cpr!=nullptr){
			//I_Off_diag в локальной нумерации																	
			for (auto it = J.begin(); it != J.end(); it++) {
				for (int m = Off_diag_Cpr->ia[*it]; m < Off_diag_Cpr->ia[*it + 1]; m++) {
					I_Off_diag_loc.insert(Off_diag_Cpr->ja[m]);
					//сразу добавляем в A^ COO
					A_a.push_back(Off_diag_Cpr->a[m]);
					A_ia.push_back(*it);
					A_ja.push_back(this->garray[Off_diag_Cpr->ja[m]]);
					A_nnz++;

				}
			}
			//I_Off_diag в глобальной нумерации ???
			I_Off_diag.reserve(I_Off_diag_loc.size());
			std::copy(I_Off_diag_loc.begin(), I_Off_diag_loc.end(), std::back_inserter(I_Off_diag));
			for(int i = 0; i < I_Off_diag.size(); i++){
				I_Off_diag[i] = this->garray[I_Off_diag[i]];
			}
		}

		//собираем I и сортируем ???
		std::vector<int> I;
		I.reserve(I_Off_diag.size() + I_Diag.size());
		for(int i = 0; i < I_Diag.size(); i++){
			I.push_back(I_Diag[i]);
		}
		for(int i = 0; i < I_Off_diag.size(); i++){
			I.push_back(I_Off_diag[i]);
		}
		sort(I.begin(), I.end());

		//формируем А^ в COO
		int A_n = J.size();
		int A_m = I_Diag_loc.size() + I_Off_diag_loc.size();

		//переходим в локальную нумерацию А^
		for (int i = 0; i<A_a.size();i++){
			A_ia[i] = find_value(J,A_ia[i]);
			A_ja[i] = find_value(I,A_ja[i]);
		}

		COOMatrix A_COO(A_m, A_n, &(A_a[0]), &(A_ja[0]), &(A_ia[0]), A_a.size());
		COOMatrix AT_COO(A_n, A_m, &(A_a[0]), &(A_ia[0]), &(A_ja[0]), A_a.size());


		Eigen::SparseMatrix<double> A = A_COO.get_eigen_sparse();
		Eigen::SparseMatrix<double> AT = AT_COO.get_eigen_sparse();
		Eigen::MatrixXd D = AT * A;
		Eigen::VectorXd e = Eigen::VectorXd::Zero(I.size()); //оптимизировать I.size()????
		{																	
			int counter = 0;
			for (auto it = I.begin(); it != I.end(); it++) {
				if (*it == this->first_row + j) {
					e(counter) = 1;
				}
				counter++;
			}
		}

		Eigen::VectorXd b = AT * e;
		// Eigen::VectorXd m = D.ldlt().solve(b);
		Eigen::VectorXd m = D.householderQr().solve(b);
		
		for (int i = 0; i < m.size(); i++) {
			a.push_back(m(i));
			ja.push_back(J[i]);
		}
		count += m.size();
		ia.push_back(count);

	}
	CRSMatrix M(Diag->n, Diag->m, &(a[0]), &(ia[0]), &(ja[0]));
	std::cout<<"\nrank = "<< rank <<", M = \n";
	M.print();
	std::cout<<"\n";
	return M;
}

CRSMatrix COOMatrix::compress_CRS(std::vector<int> garray){
		CRSMatrix crs = this->coo_to_crs();
		for (int i = 0; i < crs.n; i++){
			for (int k = crs.ia[i]; k < crs.ia[i + 1]; k++){
				crs.ja[k] = find_value(garray, crs.ja[k]);
			}
		}

		crs.m = garray.size();
		return crs;
}
CRSMatrix CRSMatrix::compress_CRS(std::vector<int> garray){
		CRSMatrix crs;
		crs.n = this->n;
		crs.m = this->m;
		crs.nnz = this->nnz;
		crs.a = this->a;
		crs.ia = this->ia;
		crs.ja = this->ja;
		for (int i = 0; i < crs.n; i++){
			for (int k = crs.ia[i]; k < crs.ia[i + 1]; k++){
				crs.ja[k] = find_value(garray, crs.ja[k]);
			}
		}

		crs.m = garray.size();
		return crs;
}

std::vector<int> CRS_like_petsc::get_narray(){
	std::vector<int> narray;
	int size;
	int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	narray.resize(size + 1);

	MPI_Allgather(&first_row, 1, MPI_INT, narray.data(), 1, MPI_INT, MPI_COMM_WORLD);
	narray[size] = n_global;
	return narray;
    //	MPI_Comm  mycomm;      								//. . . как прострелить себе колено
    //	std::vector<int> indexes;
	//	std::vector<int> edges;
	//	indexes.resize(size);
	//	edges.reserve((size-1)*size);
	//	indexes[0] = size - 1;
	//	for(int i = 1; i < size; i++){
	//		indexes[i] = indexes[i-1] + size - 1;
	//	}
	//	for(int i = 0; i < size; i++){
	//		for(int j = 0; j < size; j++){
	//			if (i != j){
	//				edges.push_back(j);
	//			}
	//		}
	//	}
    //	MPI_Graph_create(MPI_COMM_WORLD, size, indexes.data(), edges.data(), false, &mycomm);
	//	std::vector<int> sbuf;
    //	std::vector<int> rbuf; 
    //	std::vector<int> scounts; 
    //	std::vector<int> sdisp;
    //	std::vector<int> rcounts; 
    //	std::vector<int> rdisp;
	//	sbuf.push_back(first_row);
	//	scounts.reserve(size - 1);
	//	sdisp.reserve(size - 1);
	//	rcounts.reserve(size - 1);
	//	for (int i = 0; i < size - 1; i++){
	//		scounts.push_back(1);
	//		sdisp.push_back(0);
	//		rcounts.push_back(1);
	//	}
	//	rdisp.resize(size - 1);
	//	rdisp[0] = 0;
	//	for (int i = 1; i < size - 1; i++){
	//		rdisp[i] = rdisp[i-1] + rcounts[i-1];
	//	}
	//	rbuf.reserve(size);
	//	MPI_Neighbor_alltoallv(sbuf.data(), scounts.data(), sdisp.data(), MPI_INT, 
    //	                        rbuf.data(), rcounts.data(), rdisp.data(),MPI_INT, mycomm);

}

int CRS_like_petsc::get_min_n_loc(std::vector<int> narray){
	int min = this->n_global;
	int r;
	for (int i = 1; i < narray.size(); i++){
		r = narray[i] - narray[i-1];
		if (min > r) min = r;
	}
	return min;
}
int CRS_like_petsc::get_max_n_loc(std::vector<int> narray){
	int max = 0;
	int r;
	for (int i = 1; i < narray.size(); i++){
		r = narray[i] - narray[i-1];
		if (max < r) max = r;
	}
	return max;
}

CRSMatrix CRS_like_petsc::get_A(std::vector<int> J, std::vector<int> narray){
	int size;
	int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	std::vector<int> count_by_rank;
	count_by_rank.resize(size, 0);    //sbuf
	{
		int counter = 0;      			
		for (auto it = J.begin(); it != J.end(); it++){
			while (*it >= narray[counter]){
				counter++;
			}
			count_by_rank[counter - 1]++;
		}
	}

	std::vector<int> count_J_by_rank;
	count_J_by_rank.resize(size, 0); //rbuf
	MPI_Alltoall(count_by_rank.data(), 1, MPI_INT, count_J_by_rank.data(), 1, MPI_INT, MPI_COMM_WORLD); //пересылка 1
	// готовим MPI_Alltoallv
	std::vector<int> J_accepted;
	int sum_of_cjbr = std::accumulate(count_J_by_rank.begin(), count_J_by_rank.end(), 0);
	J_accepted.resize(sum_of_cjbr);
	{
		int sdist = count_by_rank[0];
		int rdist = count_J_by_rank[0];
		std::vector<int> sdisp;
		std::vector<int> rdisp;
		sdisp.resize(size, 0);
		rdisp.resize(size, 0);
		for (int i = 1; i < size; i++){
			if (count_by_rank[i] == 0){
				sdisp[i] = 0;
			}
			else{
				sdisp[i] = sdist;
				sdist += count_by_rank[i];
			}

			if (count_J_by_rank[i] == 0){
				rdisp[i] = 0;
			}
			else{
				rdisp[i] = rdist;
				rdist += count_J_by_rank[i];
			}
		}
		MPI_Alltoallv(J.data(), count_by_rank.data(), sdisp.data(), MPI_INT, 
					  J_accepted.data(), count_J_by_rank.data(), rdisp.data(), MPI_INT, MPI_COMM_WORLD); //пересылка 2

	}

	//буфера сюда
	std::vector<int> ind_A_rec;
	std::vector<double> data_A_rec;
	std::vector<int> len_for_J_rec;
	len_for_J_rec.resize(J.size());

	std::vector<int> len_for_J;
	len_for_J.resize(J_accepted.size());
	{	
		std::vector<int> ind_A;
		std::vector<double> data_A;
		std::cout << "J_accepted.size() = " << J_accepted.size() << "\n";
		std::cout << "m_global = " << m_global << "\n";
		//ind_A.reserve(J_accepted.size() * m_global);
		//data_A.reserve(J_accepted.size() * m_global);
		ind_A.reserve(m_global);
		data_A.reserve(m_global);
		std::vector<int> cjbr = count_J_by_rank;
		int counter;
		int counter_two = 0;
		std::vector<int> scount;
		scount.resize(size);
		for (auto v:J_accepted){
			counter = 0;
			while (cjbr[counter] == 0) counter++;
			cjbr[counter]--;
			if ( Diag->nnz != 0){
				for (int i = Diag->ia[v - first_row]; i < Diag->ia[v - first_row + 1]; i++){
					data_A.push_back(Diag->a[i]);
					ind_A.push_back(Diag->ja[i] + first_row);
					scount[counter]++;
					len_for_J[counter_two]++;
				}
			}
			if ( Off_diag_Cpr->nnz != 0){
				for (int i = Off_diag_Cpr->ia[v - first_row]; i < Off_diag_Cpr->ia[v - first_row + 1]; i++){
					data_A.push_back(Off_diag_Cpr->a[i]);
					ind_A.push_back(garray[Off_diag_Cpr->ja[i]]);
					scount[counter]++;
					len_for_J[counter_two]++;
				}
			}
			counter_two++;
		}
		std::vector<int> rcount;
		rcount.resize(size);
		MPI_Alltoall(scount.data(), 1, MPI_INT, rcount.data(), 1, MPI_INT, MPI_COMM_WORLD); //пересылка 3
		int sum_of_rcount = std::accumulate(rcount.begin(), rcount.end(), 0);
		ind_A_rec.resize(sum_of_rcount, 0);
		data_A_rec.resize(sum_of_rcount, 0);
		int sdist = scount[0];
		int rdist = rcount[0];
		std::vector<int> sdisp;
		std::vector<int> rdisp;
		sdisp.resize(size, 0);
		rdisp.resize(size, 0);
		for (int i = 1; i < size; i++){
			if (scount[i] == 0){
				sdisp[i] = 0;
			}
			else{
				sdisp[i] = sdist;
				sdist += scount[i];
			}

			if (rcount[i] == 0){
				rdisp[i] = 0;
			}
			else{
				rdisp[i] = rdist;
				rdist += rcount[i];
			}
		}
		MPI_Alltoallv(ind_A.data(), scount.data(), sdisp.data(), MPI_INT, 
					  ind_A_rec.data(), rcount.data(), rdisp.data(), MPI_INT, MPI_COMM_WORLD); //пересылка 4
		MPI_Alltoallv(data_A.data(), scount.data(), sdisp.data(), MPI_DOUBLE, 
					  data_A_rec.data(), rcount.data(), rdisp.data(), MPI_DOUBLE, MPI_COMM_WORLD); //пересылка 5
		
	}
	{
		int rdist = count_by_rank[0];
		int sdist = count_J_by_rank[0];
		std::vector<int> sdisp;
		std::vector<int> rdisp;
		sdisp.resize(size, 0);
		rdisp.resize(size, 0);
		for (int i = 1; i < size; i++){
			if (count_by_rank[i] == 0){
				rdisp[i] = 0;
			}
			else{
				rdisp[i] = rdist;
				rdist += count_by_rank[i];
			}

			if (count_J_by_rank[i] == 0){
				sdisp[i] = 0;
			}
			else{
				sdisp[i] = sdist;
				sdist += count_J_by_rank[i];
			}
		}
		MPI_Alltoallv(len_for_J.data(), count_J_by_rank.data(), sdisp.data(), MPI_INT, 
					  len_for_J_rec.data(), count_by_rank.data(), rdisp.data(), MPI_INT, MPI_COMM_WORLD); //пересылка 6
	}


	//	{	
	//		std::ofstream outfile("../debug/generate_like_petsc_"+std::to_string(rank)+".txt");
	//		outfile << "J = ";
	//		for (auto v:J) outfile << v << ", ";
	//		outfile << "\n";
	//	
	//		outfile << "cbr = ";
	//		for (auto v:count_by_rank) outfile << v << ", ";
	//		outfile << "\n";
	//	
	//		outfile << "cjbr = ";
	//		for (auto v:count_J_by_rank) outfile << v << ", ";
	//		outfile << "\n";
	//	
	//		outfile << "J_accepted = ";
	//		for (auto v:J_accepted) outfile << v << ", ";
	//		outfile << "\n";
	//	}
	std::vector<int> A_ia;
	A_ia.resize(J.size() + 1, 0);
	for (int i = 1; i < J.size() + 1; i++){
		A_ia[i] = A_ia[i - 1] + len_for_J_rec[i - 1];
	}

	//	{
	//		std::ofstream outfile("../debug/generate_like_petsc_"+std::to_string(rank)+".txt");
	//		outfile << "Data_A = ";
	//		for (auto v:data_A_rec) outfile << v << ", ";
	//		outfile << "\n";
	//	
	//		outfile << "Ind_A = ";
	//		for (auto v:ind_A_rec) outfile << v << ", ";
	//		outfile << "\n";
	//	
	//		outfile << "len_for_J_rec = ";
	//		for (auto v:len_for_J_rec) outfile << v << ", ";
	//		outfile << "\n";
	//	
	//		outfile << "A_ia = ";
	//		for (auto v:A_ia) outfile << v << ", ";
	//		outfile << "\n";
	//	
	//	}
	if (J.size() == 0) return CRSMatrix();
	CRSMatrix A(J.size(), m_global, &(data_A_rec[0]), &(A_ia[0]), &(ind_A_rec[0])); 



	return A;
}

CRS_like_petsc CRS_like_petsc::SPAI_old(CRS_like_petsc pattern){
	//матрица pattern должна быть распределена по процессам так же, как и *this
	//получим n_loc с каждого процесса
	std::vector<int> narray = get_narray(); //тут пересылка 1
	int min_n_loc = get_min_n_loc(narray);
	int max_n_loc = get_max_n_loc(narray);
	std::vector<double> a;
	a.reserve(Diag->nnz + Off_diag_Cpr->nnz);
	std::vector<int> ia;
	ia.reserve(Diag->nnz + Off_diag_Cpr->nnz);
	std::vector<int> ja;
	ja.reserve(Diag->nnz + Off_diag_Cpr->nnz);
	for (int k = 0; k < min_n_loc; k++){//первый цикл
		//J сразу в глобальной нумерации по неубыванию
		std::vector<int> J;
		J.reserve(this->m_global);
		if (pattern.Diag->nnz != 0){
			for(int i = pattern.Diag->ia[k]; i<pattern.Diag->ia[k+1]; i++ ){
				J.push_back(pattern.Diag->ja[i] + first_row);
			}
		}
		if (pattern.Off_diag_Cpr->nnz != 0){
			for(int i = pattern.Off_diag_Cpr->ia[k]; i<pattern.Off_diag_Cpr->ia[k+1]; i++ ){
				J.push_back(this->garray[pattern.Off_diag_Cpr->ja[i]]);
			}
		}
		auto compare = [this](int a, int b) {
			return a < b;
		};
		std::sort(J.begin(), J.end(), compare);
		CRSMatrix A_CRS = get_A(J, narray); //тут все остальные пересылки
		COOMatrix A_COO = A_CRS.crs_to_coo();

		std::set<int> I_set;
		std::vector<int> I;
		for (auto v:A_COO.ja){
			I_set.insert(v);
		}
		I.reserve(I_set.size());
		std::copy(I_set.begin(), I_set.end(), std::back_inserter(I));
		for (int i = 0; i < A_COO.nnz; i++){
			A_COO.ja[i] = find_value(I, A_COO.ja[i]);
		}
		A_COO.m = I.size();

		COOMatrix AT_COO(A_COO.m, A_COO.n, &(A_COO.a[0]), &(A_COO.ja[0]), &(A_COO.ia[0]), A_COO.nnz);
	
		Eigen::SparseMatrix<double> A = A_COO.get_eigen_sparse();
		Eigen::SparseMatrix<double> AT = AT_COO.get_eigen_sparse();
		Eigen::MatrixXd D = A * AT;
		Eigen::VectorXd e = Eigen::VectorXd::Zero(I.size()); //I ????
		{																	
			int counter = 0;
			for (auto it = I.begin(); it != I.end(); it++) {
				if (*it == this->first_row + k) {
					e(counter) = 1;
				}
				counter++;
			}
		}
		Eigen::VectorXd b = A * e;
		// Eigen::VectorXd m = D.ldlt().solve(b);
		Eigen::VectorXd m = D.householderQr().solve(b);
		for (int i = 0; i < m.size(); i++) {
			a.push_back(m(i));
			ia.push_back(k);
			ja.push_back(J[i]);
		}
	}


	for (int k = min_n_loc; k < max_n_loc; k++){//второй цикл
		std::vector<int> J;
		if (Diag->n > k){ //если остались строки на этом процессе
			//J сразу в глобальной нумерации по неубыванию
			J.reserve(this->m_global);
			for(int i = pattern.Diag->ia[k]; i<pattern.Diag->ia[k+1]; i++ ){
				J.push_back(pattern.Diag->ja[i] + first_row);
			}
			for(int i = pattern.Off_diag_Cpr->ia[k]; i<pattern.Off_diag_Cpr->ia[k+1]; i++ ){
				J.push_back(this->garray[pattern.Off_diag_Cpr->ja[i]]);
			}
			auto compare = [this](int a, int b) {
				return a < b;
			};
			std::sort(J.begin(), J.end(), compare);
		}

		CRSMatrix A_CRS = get_A(J, narray); //тут все остальные пересылки
		COOMatrix A_COO = A_CRS.crs_to_coo();

		if (Diag->n > k){
			std::set<int> I_set;
			std::vector<int> I;
			for (auto v:A_COO.ja){
				I_set.insert(v);
			}
			I.reserve(I_set.size());
			std::copy(I_set.begin(), I_set.end(), std::back_inserter(I));
			for (int i = 0; i < A_COO.nnz; i++){
				A_COO.ja[i] = find_value(I, A_COO.ja[i]);
			}
			A_COO.m = I.size();

			COOMatrix AT_COO(A_COO.m, A_COO.n, &(A_COO.a[0]), &(A_COO.ja[0]), &(A_COO.ia[0]), A_COO.a.size());
			Eigen::SparseMatrix<double> A = A_COO.get_eigen_sparse();
			Eigen::SparseMatrix<double> AT = AT_COO.get_eigen_sparse();
			Eigen::MatrixXd D = A * AT;
			Eigen::VectorXd e = Eigen::VectorXd::Zero(I.size()); //I ????
			{																	
				int counter = 0;
				for (auto it = I.begin(); it != I.end(); it++) {
					if (*it == this->first_row + k) {
						e(counter) = 1;
					}
					counter++;
				}
			}
			Eigen::VectorXd b = A * e;
			// Eigen::VectorXd m = D.ldlt().solve(b);
			Eigen::VectorXd m = D.householderQr().solve(b);
			for (int i = 0; i < m.size(); i++) {
				a.push_back(m(i));
				ia.push_back(k);
				ja.push_back(J[i]);
			}
		}

	}

	COOMatrix M_COO(pattern.Diag->n, pattern.m_global, &(a[0]), &(ia[0]), &(ja[0]), a.size());

	CRSMatrix M_CRS = M_COO.coo_to_crs();
	//CRS -> CRS_like_petsc
	
	CRS_like_petsc M;
	M.n_global = pattern.n_global;
	M.m_global = pattern.m_global;
	M.first_row = pattern.first_row;
	M.rank = pattern.rank;
	M.garray = pattern.garray;

	std::vector<double> D_a;
	D_a.reserve(pattern.Diag->nnz);
	std::vector<int> D_ja;
	D_ja.reserve(pattern.Diag->nnz);

	std::vector<double> Off_D_a;
	Off_D_a.reserve(pattern.Off_diag_Cpr->nnz);
	std::vector<int> Off_D_ja;
	Off_D_ja.reserve(pattern.Off_diag_Cpr->nnz);

	for (int i = 0; i < M_CRS.n; i++){
		for (int c = M_CRS.ia[i]; c < M_CRS.ia[i + 1]; c++){
			if ((M_CRS.ja[c] >= pattern.first_row) and (M_CRS.ja[c] < pattern.first_row + M_CRS.n)){
				D_a.push_back(M_CRS.a[c]);
				D_ja.push_back(M_CRS.ja[c] - pattern.first_row);
			}
			else{
				Off_D_a.push_back(M_CRS.a[c]);
				Off_D_ja.push_back(find_value(pattern.garray, M_CRS.ja[c]));
			}
		}
	}


	if (pattern.Diag->nnz != 0) M.Diag = new CRSMatrix(M_CRS.n, M_CRS.n, &(D_a[0]), &(pattern.Diag->ia[0]), &(D_ja[0]));
	else M.Diag = new CRSMatrix();
	if (pattern.Off_diag_Cpr->nnz != 0) M.Off_diag_Cpr = new CRSMatrix(M_CRS.n, pattern.garray.size(), &(Off_D_a[0]), &(pattern.Off_diag_Cpr->ia[0]), &(Off_D_ja[0]));
	else M.Off_diag_Cpr = new CRSMatrix();
	return M;

}

CRS_like_petsc COOMatrix::generate_like_petsc(int* narray, int rank){

	sort_ia();
	CRS_like_petsc A;
	A.n_global = this->n;
	A.m_global = this->m;
	A.first_row = narray[rank];

	std::vector<double> D_a_coo;
	int s2 = (narray[rank] - narray[rank + 1])*(narray[rank] - narray[rank + 1]);
	std::cout << s2 << "\n";
	size_t s = std::min(s2, this->nnz);
	std::cout << s << "\n";
	D_a_coo.reserve(s);
	std::vector<int> D_ia_coo;
	D_ia_coo.reserve(std::min((narray[rank] - narray[rank + 1])*(narray[rank] - narray[rank + 1]), this->nnz));
	std::vector<int> D_ja_coo;
	D_ja_coo.reserve(std::min((narray[rank] - narray[rank + 1])*(narray[rank] - narray[rank + 1]), this->nnz));

	std::vector<double> Off_D_a_coo;
	Off_D_a_coo.reserve(std::min((narray[rank] - narray[rank + 1])*(narray[rank] - narray[rank + 1]), this->nnz));
	std::vector<int> Off_D_ia_coo;
	Off_D_ia_coo.reserve(std::min((narray[rank] - narray[rank + 1])*(narray[rank] - narray[rank + 1]), this->nnz));
	std::vector<int> Off_D_ja_coo;
	Off_D_ja_coo.reserve(std::min((narray[rank] - narray[rank + 1])*(narray[rank] - narray[rank + 1]), this->nnz));

	std::set<int> garray_set;
	std::vector<int> garray;
	int n_loc = narray[rank + 1] - narray[rank];
	int nnz_D = 0;
	int nnz_Off_D = 0;
	{
		int n = 0;
		while((n!=this->nnz) && (this->ia[n] < narray[rank])){
			n++;
		}

		while((n!=this->nnz) && (this->ia[n] < narray[rank + 1])){
			if ((this->ja[n] >= narray[rank]) && (this->ja[n] < narray[rank + 1])){
				D_a_coo.push_back(this->a[n]);
				D_ia_coo.push_back(this->ia[n] - narray[rank]);
				D_ja_coo.push_back(this->ja[n] - narray[rank]);
				nnz_D++;
			}
			else{
				nnz_Off_D++;
				Off_D_a_coo.push_back(this->a[n]);
				Off_D_ia_coo.push_back(this->ia[n] - narray[rank]);
				if(this->ja[n]<narray[rank]){
					Off_D_ja_coo.push_back(this->ja[n]);
				}
				else{
					Off_D_ja_coo.push_back(this->ja[n]);
				}
				garray_set.insert(this->ja[n]);

			}
			n++;
		}

	}

	//garray copy
	garray.reserve(garray_set.size());
	std::copy(garray_set.begin(), garray_set.end(), std::back_inserter(garray));
	if (nnz_D != 0){
		COOMatrix D_coo(n_loc, n_loc, &(D_a_coo[0]), &(D_ia_coo[0]), &(D_ja_coo[0]), nnz_D);
		CRSMatrix D_crs = D_coo.coo_to_crs();
		A.Diag = new CRSMatrix(D_crs.n, D_crs.m, &(D_crs.a[0]), &(D_crs.ia[0]), &(D_crs.ja[0]));
	}
	else A.Diag = new CRSMatrix();
	if (nnz_Off_D != 0){
		COOMatrix Off_D_coo(n_loc, this->n - n_loc, &(Off_D_a_coo[0]), &(Off_D_ia_coo[0]), &(Off_D_ja_coo[0]), nnz_Off_D);

		CRSMatrix Off_D_cpr = Off_D_coo.compress_CRS(garray);
		A.Off_diag_Cpr = new CRSMatrix(Off_D_cpr.n, Off_D_cpr.m, &(Off_D_cpr.a[0]), &(Off_D_cpr.ia[0]), &(Off_D_cpr.ja[0]));
	}
	else A.Off_diag_Cpr = new CRSMatrix();
	A.garray = garray;
	A.rank = rank;



	return A;
}

CRS_like_petsc CRSMatrix::generate_like_petsc(int* narray, int rank){//Не используем coo
	CRS_like_petsc A;
	A.n_global = this->n;
	A.m_global = this->m;
	A.first_row = narray[rank];
	int n_loc = narray[rank + 1] - narray[rank];
	A.rank = rank;

	std::vector<double> D_a;
	std::vector<int> D_ia;
	D_ia.reserve(n_loc + 1);
	D_ia.push_back(0);
	std::vector<int> D_ja;

	std::vector<double> Off_a;
	std::vector<int> Off_ia;
	Off_ia.reserve(n_loc + 1);
	Off_ia.push_back(0);
	std::vector<int> Off_ja;

	std::set<int> garray_set;
	int nnz_D = 0;
	int nnz_Off = 0;
	for (int i = 0; i < this->n; i++){
		if ((i >= narray[rank]) and (i < narray[rank] + n_loc)){
			for (int k = this->ia[i]; k < this->ia[i + 1]; k++){
				if ((this->ja[k]>= narray[rank]) and (this->ja[k] < narray[rank] + n_loc)){
					nnz_D++;
					D_a.push_back(this->a[k]);
					D_ja.push_back(this->ja[k] - narray[rank]);

				}
				else{
					nnz_Off++;
					Off_a.push_back(this->a[k]);
					Off_ja.push_back(this->ja[k]);
					garray_set.insert(this->ja[k]);
				}
			}
			D_ia.push_back(nnz_D);
			Off_ia.push_back(nnz_Off);
		}
	}
	std::vector<int> garray;
	garray.reserve(garray_set.size());
	std::copy(garray_set.begin(), garray_set.end(), std::back_inserter(garray));
	A.garray = garray;
	if (nnz_D != 0){
		A.Diag = new CRSMatrix(n_loc, n_loc, &(D_a[0]), &(D_ia[0]), &(D_ja[0]));

	}
	else A.Diag = new CRSMatrix();

	if (nnz_Off != 0){
		for (int i = 0; i < n_loc; i++){
			for (int k = Off_ia[i]; k < Off_ia[i + 1]; k++){
				Off_ja[k] = find_value(garray, Off_ja[k]);
			}
		}
		A.Off_diag_Cpr = new CRSMatrix(n_loc, garray_set.size(), &(Off_a[0]), &(Off_ia[0]), &(Off_ja[0]));

	}
	else A.Off_diag_Cpr = new CRSMatrix();
	return A;

}


CRS_like_petsc CRS_like_petsc::SPAI(CRS_like_petsc pattern){
	//матрица pattern должна быть распределена по процессам так же, как и *this
	//получим n_loc с каждого процесса
	
	std::vector<int> narray = get_narray(); //тут пересылка 1
	int n_loc = narray[rank + 1] - narray[rank];
	int ind_str;
	int A_nnz;

	std::vector<double> a;
	a.reserve(Diag->nnz + Off_diag_Cpr->nnz);
	std::vector<int> ia;
	ia.reserve(Diag->nnz + Off_diag_Cpr->nnz);
	std::vector<int> ja;
	ja.reserve(Diag->nnz + Off_diag_Cpr->nnz);

	std::set<int> All_col_set;

	for (auto v:pattern.Diag->ja){
		All_col_set.insert(v + first_row);
	}
	for (auto v:pattern.garray){
		All_col_set.insert(v);
	}
	std::vector<int> All_col;
	All_col.reserve(All_col_set.size());
	std::copy(All_col_set.begin(), All_col_set.end(), std::back_inserter(All_col));
	CRSMatrix Af_CRS = get_A(All_col, narray);


	for (int k = 0; k < n_loc; k++){
		//J сразу в глобальной нумерации по неубыванию
		std::vector<int> J;
		J.reserve(this->m_global);
		if (pattern.Diag->nnz != 0){
			for(int i = pattern.Diag->ia[k]; i<pattern.Diag->ia[k+1]; i++ ){
				J.push_back(pattern.Diag->ja[i] + first_row);
			}
		}
		if (pattern.Off_diag_Cpr->nnz != 0){
			for(int i = pattern.Off_diag_Cpr->ia[k]; i<pattern.Off_diag_Cpr->ia[k+1]; i++ ){
				// J.push_back(this->garray[pattern.Off_diag_Cpr->ja[i]]);
				J.push_back(pattern.garray[pattern.Off_diag_Cpr->ja[i]]);
			}
		}
		auto compare = [this](int a, int b) {
			return a < b;
		};
		std::sort(J.begin(), J.end(), compare);

		A_nnz = 0;
		std::vector<int> A_ia;
		A_ia.reserve(J.size() + 1);
		A_ia.push_back(0);

		for (auto v:J){
			ind_str = find_value(All_col, v);
			for(int i = Af_CRS.ia[ind_str]; i<Af_CRS.ia[ind_str + 1]; i++ ){
				// auto it = std::lower_bound(J.begin(), J.end(), Af_CRS.ja[i]);
				// if( it != J.end() ) {
				// 	if( (*it) == Af_CRS.ja[i] ) {
						A_nnz++;
				// 	}
				// }
			}
			A_ia.push_back(A_nnz);
		}

		std::vector<double> A_a;
		A_a.reserve(A_nnz);
		std::vector<int> A_ja;
		A_ja.reserve(A_nnz);



		for (auto v:J){
			ind_str = find_value(All_col, v);
			// int center = -1;
			// double acc = 0.0;
			for(int i = Af_CRS.ia[ind_str]; i<Af_CRS.ia[ind_str + 1]; i++ ){
				// auto it = std::lower_bound(J.begin(), J.end(), Af_CRS.ja[i]);
				// if( it != J.end() ) {
				// 	if( (*it) == Af_CRS.ja[i] ) {
						// std::cout << "Af_CRS.a[i] = " << Af_CRS.a[i] << "\n";
						// std::cout << "Af_CRS.ja[i] = " << Af_CRS.ja[i] << "\n";
						// std::cout << "i = " << i << "\n";
						A_a.push_back(Af_CRS.a[i]);
						A_ja.push_back(Af_CRS.ja[i]);
				// 		if(Af_CRS.ja[i]==v) {
				// 			center = A_a.size() - 1;
				// 		}
				// 	} else {
				// 		acc += Af_CRS.a[i];
				//     }
				// }  else {
				// 	acc += Af_CRS.a[i];
				// }
				

			}			
			// std::cout << "center = " << center << "\n";
			// if(center!=-1) A_a[center]+=acc;
		}


		// std::this_thread::sleep_for(std::chrono::milliseconds(10000));

		// std::exit(0);

		CRSMatrix A_CRS(J.size(), this->m_global, &(A_a[0]), &(A_ia[0]), &(A_ja[0]));
		std::set<int> I_set;
		std::vector<int> I;
		for (auto v:A_CRS.ja){
			I_set.insert(v);
		}
		I.reserve(I_set.size());
		std::copy(I_set.begin(), I_set.end(), std::back_inserter(I));
		for (int i = 0; i < A_CRS.nnz; i++){
			A_CRS.ja[i] = find_value(I, A_CRS.ja[i]);
		}
		A_CRS.m = I.size();
		COOMatrix A_COO = A_CRS.crs_to_coo();

		Eigen::SparseMatrix<double> A = A_COO.get_eigen_sparse();
		Eigen::MatrixXd D = A * A.transpose();
		Eigen::VectorXd e = Eigen::VectorXd::Zero(I.size());
		{																	
			int counter = 0;
			for (auto it = I.begin(); it != I.end(); it++) {
				if (*it == this->first_row + k) {
					e(counter) = 1;
				}
				counter++;
			}
		}
		Eigen::VectorXd b = A * e;
		// Eigen::VectorXd m = D.ldlt().solve(b);
		Eigen::VectorXd m = D.householderQr().solve(b);
		// Eigen::VectorXd m = D.colPivHouseholderQr().solve(b);
		// Eigen::VectorXd m = D.fullPivHouseholderQr().solve(b);

		for (int i = 0; i < m.size(); i++) {
			
			// if(std::isnan(m(i))) {
			// 	std::cout << Eigen::MatrixXd(D) << std::endl;
			// 	Eigen::JacobiSVD<Eigen::MatrixXd, Eigen::HouseholderQRPreconditioner> svd{Eigen::MatrixXd(D), Eigen::ComputeThinU | Eigen::ComputeThinV};
			// 	std::cout << "Its singular values are:" << std::endl << svd.singularValues() << std::endl;
			// 	std::cout << "Its left singular vectors are the columns of the thin U matrix:" << std::endl << svd.matrixU() << std::endl;
			// 	std::cout << "Its right singular vectors are the columns of the thin V matrix:" << std::endl << svd.matrixV() << std::endl;
			// 	std::cout << "main row = " << this->first_row + k << "\n";
			// 	for(int v = 0 ; v < I.size() ; v++) {
			// 		std::cout << I[v] << "\n";
			// 	}
			// 	std::cout << "\n";
			// 	for(int v = 0 ; v < A_ia.size()-1 ; v++) {
			// 		for( int z = A_ia[v] ; z < A_ia[v+1] ; z++ ) {
			// 			std::cout << A_a[z] << " ";
			// 		}
			// 		std::cout << "\n";
			// 	}
			// 	std::cout << "\n";
			// 	for(int v = 0 ; v < A_ia.size()-1 ; v++) {
			// 		for( int z = A_ia[v] ; z < A_ia[v+1] ; z++ ) {
			// 			std::cout << A_ja[z] << " ";
			// 		}
			// 		std::cout << "\n";
			// 	}
			// 	std::cout << "\n";
			// 	std::cout << "ERROR!\n";
			// 	std::exit(0);
			// }
			a.push_back(m(i));
			ia.push_back(k);
			ja.push_back(J[i]);
		}

	}
	COOMatrix M_COO(pattern.Diag->n, pattern.m_global, &(a[0]), &(ia[0]), &(ja[0]), a.size());

	CRSMatrix M_CRS = M_COO.coo_to_crs();
	//CRS -> CRS_like_petsc
	
	CRS_like_petsc M;
	M.n_global = pattern.n_global;
	M.m_global = pattern.m_global;
	M.first_row = pattern.first_row;
	M.rank = pattern.rank;
	M.garray = pattern.garray;

	std::vector<double> D_a;
	D_a.reserve(pattern.Diag->nnz);
	std::vector<int> D_ja;
	D_ja.reserve(pattern.Diag->nnz);

	std::vector<double> Off_D_a;
	Off_D_a.reserve(pattern.Off_diag_Cpr->nnz);
	std::vector<int> Off_D_ja;
	Off_D_ja.reserve(pattern.Off_diag_Cpr->nnz);

	for (int i = 0; i < M_CRS.n; i++){
		for (int c = M_CRS.ia[i]; c < M_CRS.ia[i + 1]; c++){
			if ((M_CRS.ja[c] >= pattern.first_row) and (M_CRS.ja[c] < pattern.first_row + M_CRS.n)){
				D_a.push_back(M_CRS.a[c]);
				D_ja.push_back(M_CRS.ja[c] - pattern.first_row);
			}
			else{
				Off_D_a.push_back(M_CRS.a[c]);
				Off_D_ja.push_back(find_value(pattern.garray, M_CRS.ja[c]));
			}
		}
	}

	if (pattern.Diag->nnz != 0) M.Diag = new CRSMatrix(M_CRS.n, M_CRS.n, &(D_a[0]), &(pattern.Diag->ia[0]), &(D_ja[0]));
	else M.Diag = new CRSMatrix();
	if (pattern.Off_diag_Cpr->nnz != 0) M.Off_diag_Cpr = new CRSMatrix(M_CRS.n, pattern.garray.size(), &(Off_D_a[0]), &(pattern.Off_diag_Cpr->ia[0]), &(Off_D_ja[0]));
	else M.Off_diag_Cpr = new CRSMatrix();
	return M;
}