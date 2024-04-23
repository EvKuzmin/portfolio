#include <fstream>
#include <iomanip>
#include <sstream>
#include <Sparse>
#include <Eigen>
#include <Dense>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <set>
#include "Sparse_alpha.h"

using namespace std;
typedef Eigen::Triplet<double> T;
COOMatrix::COOMatrix(int n, int m, double* a, int* ia, int* ja, int nnz) {
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
CRSMatrix::CRSMatrix(int n, int m, double* a, int* ia, int* ja) {
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
		{																	//окружение?
			int counter = 0;
			for (auto it = I.begin(); it != I.end(); it++) {
				if (*it == j - 1) {
					e(counter) = 1;
				}
				counter++;
			}
		}
		Eigen::VectorXd b = 2 * AT_sub_mat * e;
		//Eigen::VectorXd m = D.householderQr().solve(b); //пробуем другие методы?
		Eigen::VectorXd m = D.ldlt().solve(b);
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
		//		if (std::find(Js.begin(), Js.end(), A.ja[m]) != Js.end()) { /*A.ja[m] есть в Js*/
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
		{																	//окружение?
			int counter = 0;
			for (auto it = I.begin(); it != I.end(); it++) {
				if (*it == j - 1) {
					e(counter) = 1;
				}
				counter++;
			}
		}
		Eigen::VectorXd b = 2 * AT_sub_mat * e;
		//Eigen::VectorXd m = D.householderQr().solve(b); //пробуем другие методы?
		Eigen::VectorXd m = D.ldlt().solve(b);
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
				if (std::find(Js.begin(), Js.end(), AT.ja[m]) != Js.end()) { /*A.ja[m] есть в Js*/
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
		{																	//окружение?
			int counter = 0;
			for (auto it = I.begin(); it != I.end(); it++) {
				if (*it == j - 1) {
					e(counter) = 1;
				}
				counter++;
			}
		}
		Eigen::VectorXd b = 2 * AT_sub_mat * e;
		//Eigen::VectorXd m = D.householderQr().solve(b); //пробуем другие методы?
		Eigen::VectorXd m = D.ldlt().solve(b);
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