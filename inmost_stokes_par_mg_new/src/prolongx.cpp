#include "prolongx.hpp"
#include "Sparse_alpha.hpp"
#include <mpi.h>
#include <vector>
#include <set>
#include <iostream>
#include <numeric>
#include <string>
#include <fstream>
#include <sstream>
std::tuple<std::vector<int>, std::vector<int>> get_coarse_A(std::vector<int> coarse_local_index_wo, std::vector<int> J, std::vector<int> narray){
    int size;
	int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int first_row = narray[rank];

    int n_coarse = coarse_local_index_wo.size();
    int n_coarse_ex_sum = 0;
    MPI_Exscan(&n_coarse,
                   &n_coarse_ex_sum,
                   1,
               MPI_INT,
               MPI_SUM,
               MPI_COMM_WORLD);
    

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
	MPI_Alltoall(count_by_rank.data(), 1, MPI_INT, count_J_by_rank.data(), 1, MPI_INT, MPI_COMM_WORLD);
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



    std::vector<bool> coarse_elements_wo( narray[rank + 1] - narray[rank], 0 );
    std::vector<int> coarse_elements_wo_id( narray[rank + 1] - narray[rank], 0 );
    for( int i = 0 ; i < coarse_local_index_wo.size() ; i++ ){
        coarse_elements_wo[coarse_local_index_wo[i]] = 1;
        coarse_elements_wo_id[coarse_local_index_wo[i]] = i;
    }


    std::vector<int> count_by_rank_s(size, 0);
    std::vector<int> count_by_rank_r(size, 0);
    std::vector<int> sbuf;
    std::vector<int> sbuf2;
    int k = 0;
    for (int i = 0; i < size; i++){
        for (int j = 0; j < count_J_by_rank[i]; j++){
            if(coarse_elements_wo[J_accepted[k] - first_row]){
                count_by_rank_s[i]++;
                sbuf.push_back(J_accepted[k]);
                sbuf2.push_back(n_coarse_ex_sum + coarse_elements_wo_id[J_accepted[k] - first_row]);
            }
            k++;
        }
    }

    MPI_Alltoall(count_by_rank_s.data(), 1, MPI_INT, count_by_rank_r.data(), 1, MPI_INT, MPI_COMM_WORLD); //пересылка 3

    int sum_of_cbrr = std::accumulate(count_by_rank_r.begin(), count_by_rank_r.end(), 0);
    std::vector<int> rbuf(sum_of_cbrr, 0);
    std::vector<int> rbuf2(sum_of_cbrr, 0);

    {
		int sdist = count_by_rank_s[0];
		int rdist = count_by_rank_r[0];
		std::vector<int> sdisp;
		std::vector<int> rdisp;
		sdisp.resize(size, 0);
		rdisp.resize(size, 0);
		for (int i = 1; i < size; i++){
			if (count_by_rank_s[i] == 0){
				sdisp[i] = 0;
			}
			else{
				sdisp[i] = sdist;
				sdist += count_by_rank_s[i];
			}

			if (count_by_rank_r[i] == 0){
				rdisp[i] = 0;
			}
			else{
				rdisp[i] = rdist;
				rdist += count_by_rank_r[i];
			}
		}
		MPI_Alltoallv(sbuf.data(), count_by_rank_s.data(), sdisp.data(), MPI_INT, 
					 rbuf.data(), count_by_rank_r.data(), rdisp.data(), MPI_INT, MPI_COMM_WORLD); //пересылка 4
        
        MPI_Alltoallv(sbuf2.data(), count_by_rank_s.data(), sdisp.data(), MPI_INT, 
					 rbuf2.data(), count_by_rank_r.data(), rdisp.data(), MPI_INT, MPI_COMM_WORLD); //пересылка 5

	}

    return std::make_tuple(rbuf, rbuf2);

}


std::tuple<Mat, Mat, int, int, int, int> prolongx( Mat A_wo_main_diag, int face_unknowns_local_wo, std::vector<int> &coarse_local_index_wo, int overflow = 1)
    {
    int size;
	int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int m, n, M, N;
    MatGetSize(A_wo_main_diag, &M, &N);
    MatGetLocalSize(A_wo_main_diag, &m, &n);

    int n_coarse = coarse_local_index_wo.size();
    int n_coarse_ex_sum = 0;
    int n_ex_sum = 0;

    MPI_Exscan(&n_coarse,
                   &n_coarse_ex_sum,
                   1,
               MPI_INT,
               MPI_SUM,
               MPI_COMM_WORLD);
    
    MPI_Exscan(&n,
                   &n_ex_sum,
                   1,
               MPI_INT,
               MPI_SUM,
               MPI_COMM_WORLD);

    int N_coarse = 0;

    MPI_Allreduce(&n_coarse, &N_coarse, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    Mat Prolong;

    MatCreate(PETSC_COMM_WORLD, &Prolong);
    MatSetSizes(Prolong, m, n_coarse, M, N_coarse);

    MatMPIAIJSetPreallocation(Prolong, 8, NULL, 3, NULL);

    Mat A_diag, A_off_diag;
    const int *garray;

    MatMPIAIJGetSeqAIJ(A_wo_main_diag, &A_diag, &A_off_diag, &garray);

    // PetscInt N_A_diag;
    // const PetscInt *ia_diag;
    // const PetscInt *ja_diag;
    // PetscScalar *data_diag;
    // PetscBool success_diag;
    // 
    // MatGetRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_diag, &ia_diag, &ja_diag, &success_diag);
    // MatSeqAIJGetArray(A_diag, &data_diag);
    // 
    // PetscInt N_A_off_diag;
    // const PetscInt *ia_off_diag;
    // const PetscInt *ja_off_diag;
    // PetscScalar *data_off_diag;
    // PetscBool success_off_diag;
    // 
    // MatGetRowIJ(A_off_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_off_diag, &ia_off_diag, &ja_off_diag, &success_off_diag);
    // MatSeqAIJGetArray(A_off_diag, &data_off_diag);

    int N_A_diag;
    const int *ia_diag;
    const int *ja_diag;
    double *data_diag;
    PetscBool success_diag;
    
    MatGetRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_diag, &ia_diag, &ja_diag, &success_diag);
    MatSeqAIJGetArray(A_diag, &data_diag);
    
    int N_A_off_diag;
    const int *ia_off_diag;
    const int *ja_off_diag;
    double *data_off_diag;
    PetscBool success_off_diag;
    
    MatGetRowIJ(A_off_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_off_diag, &ia_off_diag, &ja_off_diag, &success_off_diag);
    MatSeqAIJGetArray(A_off_diag, &data_off_diag);

    int m_off_loc;
    int n_off_loc;
    MatGetLocalSize(A_off_diag, &n_off_loc, &m_off_loc);

    ///////////////////////////////////////////////////////////////////////////////////////////
    CRS_like_petsc A_com(N_A_diag, data_diag, ia_diag, ja_diag, 
                        m_off_loc, data_off_diag, ia_off_diag, ja_off_diag, garray,
                        n_ex_sum, N, M, rank);                       

    

    std::vector<int> narray = A_com.get_narray();
    int first_row = narray[rank];

    std::set<int> All_col_set;
	for (auto v:A_com.Diag->ja){
		All_col_set.insert(v + first_row);
	}
	for (auto v:A_com.garray){
		All_col_set.insert(v);
	}
	std::vector<int> All_col;
	All_col.reserve(All_col_set.size());
	std::copy(All_col_set.begin(), All_col_set.end(), std::back_inserter(All_col));




	CRSMatrix Af_CRS = A_com.get_A(All_col, narray);


    for (int l = 1; l < overflow; l++){
        for (auto v:Af_CRS.ja){
	        All_col_set.insert(v);
	    }
        All_col.clear();
        All_col.reserve(All_col_set.size());
	    std::copy(All_col_set.begin(), All_col_set.end(), std::back_inserter(All_col));
        Af_CRS = A_com.get_A(All_col, narray);
    }
    auto[GI, GCI] = get_coarse_A(coarse_local_index_wo, All_col, narray);
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //std::vector<bool> coarse_elements_wo( All_col.size(), 0 );
    //std::vector<int> coarse_elements_wo_id( All_col.size(), 0 );
    //
    //
    //for( int i = 0 ; i < GI.size() ; i++ ){                         //попробовать без fv
    //    coarse_elements_wo[find_value(All_col, GI[i])] = 1;
    //    coarse_elements_wo_id[find_value(All_col, GI[i])] = i;
    //}


    std::vector<bool> coarse_elements_wo( N_A_diag, 0 );
    std::vector<int> coarse_elements_wo_id( N_A_diag, 0 );
    for( int i = 0 ; i < coarse_local_index_wo.size() ; i++ ){
        coarse_elements_wo[coarse_local_index_wo[i]] = 1;
        coarse_elements_wo_id[coarse_local_index_wo[i]] = i;
    }

    std::vector<bool> GI_coarse(All_col.size(), 0);
    std::vector<int> GI_coarse_id(All_col.size(), 0);
    for( int i = 0 ; i < GI.size() ; i++ ){                        
        GI_coarse[find_value(All_col, GI[i])] = 1;
        GI_coarse_id[find_value(All_col, GI[i])] = GCI[i];
    }


    int* rows = new int[All_col.size()];
    for (int i = 0; i < All_col.size(); i++) rows[i] = i;
    int* cols = new int[All_col.size()];
    for (int i = 0; i < All_col.size(); i++) cols[i] = All_col[i];

    CRSMatrix Af_diag = Af_CRS.get_csr_submatrix(rows, cols, All_col.size(), All_col.size());
    {
        std::vector<int> queue;
        std::vector<int> new_queue;
        std::vector<char> added_cells(Af_diag.n, 0);
        for( int i = 0 ; i < N_A_diag ; i++ ) {
            if( coarse_elements_wo[i] != 1 ) {
                {
                    auto add_neighbors_vel = [&Af_diag]( std::vector<int>& queue, std::vector<char>& added_cells, std::vector<int>& new_queue ){
                        for( auto v:queue ) {
                            for( int j = Af_diag.ia[v] ; j < Af_diag.ia[v+1] ; j++ ) {
                                if(Af_diag.ja[j]!=v && !added_cells[Af_diag.ja[j]]) {
                                    added_cells[Af_diag.ja[j]] = 1;
                                    new_queue.push_back(Af_diag.ja[j]);
                                }
                            }
                        }
                    };
                    auto process_neighbors = [i, &GI_coarse](std::vector<int>& queue, std::vector<int>& fine_coarse_neighbors_wo, int &N_found_coarse) {
                        for( auto v:queue ) {
                            if( GI_coarse[v] == 1 ) {
                                fine_coarse_neighbors_wo.push_back(v);
                                N_found_coarse++;
                            }
                        }
                    };
                    
                    
                    std::vector<int> fine_coarse_neighbors_wo;
                    
                    int N_found_coarse = 0;
                    queue.push_back(find_value(All_col, i + first_row));
                    while( N_found_coarse < 2 && queue.size() > 0 ) {
                        process_neighbors(queue, fine_coarse_neighbors_wo, N_found_coarse);
                        add_neighbors_vel(queue, added_cells, new_queue);
                        std::swap(new_queue, queue);
                        new_queue.clear();
                    }
                    queue.clear();
                    std::fill(added_cells.begin(), added_cells.end(), 0);


                    if(fine_coarse_neighbors_wo.size()!=0) {
                        for(int k = 0 ; k < fine_coarse_neighbors_wo.size() ; k++) {
                            int row = i + n_ex_sum;
                            int col = GI_coarse_id[fine_coarse_neighbors_wo[k]];
                            double value = 1.0/fine_coarse_neighbors_wo.size();
                            //std::cout << rank << " " << row << " " << col << " " << value << "\n";
                            //set_value(Prolong_coo, value, row, col);
                            MatSetValues(Prolong,1,&row,1,&col,&value,ADD_VALUES);  
                        }
                    }
                }
            } else {
                int row = i + n_ex_sum;
                int col = coarse_elements_wo_id[i] + n_coarse_ex_sum;
                double value = 1.0;
                //std::cout << rank << " " << row << " " << col << " " << value << "\n";
                //set_value(Prolong_coo, value, row, col);
                MatSetValues(Prolong,1,&row,1,&col,&value,ADD_VALUES);
            }
        }
    }

    MatRestoreRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_diag, &ia_diag, &ja_diag, &success_diag);

    MatAssemblyBegin(Prolong, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Prolong, MAT_FINAL_ASSEMBLY);

    Mat Restrict;

    MatTranspose(Prolong, MAT_INITIAL_MATRIX, &Restrict);

    return std::make_tuple( Prolong, Restrict, n_coarse, N_coarse, n_coarse_ex_sum, n_ex_sum );
}

void set_value(COOMatrix& A, double val, int i, int j){
    if ((i >= A.n)||(j >= A.m)) throw std::runtime_error("(i >= n)||(j >= m)\n");
    A.nnz++;
    A.a.push_back(val);
    A.ia.push_back(i);
    A.ja.push_back(j);

}
CRSMatrix prolongx_change(CRS_like_petsc A_com, std::vector<int> &coarse_local_index_wo, int overflow = 1){

    int size;
	int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = A_com.Diag->n;
    int N_A_diag = A_com.Diag->n;
    int n_coarse = coarse_local_index_wo.size();
    int n_coarse_ex_sum = 0;
    int n_ex_sum = 0;

    MPI_Exscan(&n_coarse,
                   &n_coarse_ex_sum,
                   1,
               MPI_INT,
               MPI_SUM,
               MPI_COMM_WORLD);
    
    MPI_Exscan(&n,
                   &n_ex_sum,
                   1,
               MPI_INT,
               MPI_SUM,
               MPI_COMM_WORLD);

    int N_coarse = 0;

    MPI_Allreduce(&n_coarse, &N_coarse, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    double* an;
    int* ian;
    int* jan;
    COOMatrix Prolong_coo(A_com.n_global, N_coarse, an, ian, jan, 0);
    std::vector<int> narray = A_com.get_narray();
    int first_row = narray[rank];

    std::set<int> All_col_set;
	for (auto v:A_com.Diag->ja){
		All_col_set.insert(v + first_row);
	}
	for (auto v:A_com.garray){
		All_col_set.insert(v);
	}
	std::vector<int> All_col;
	All_col.reserve(All_col_set.size());
	std::copy(All_col_set.begin(), All_col_set.end(), std::back_inserter(All_col));
	CRSMatrix Af_CRS = A_com.get_A(All_col, narray);

    for (int l = 1; l < overflow; l++){
        for (auto v:Af_CRS.ja){
	        All_col_set.insert(v);
	    }
        All_col.clear();
        All_col.reserve(All_col_set.size());
	    std::copy(All_col_set.begin(), All_col_set.end(), std::back_inserter(All_col));
        Af_CRS = A_com.get_A(All_col, narray);
    }
    auto[GI, GCI] = get_coarse_A(coarse_local_index_wo, All_col, narray);

    std::vector<bool> coarse_elements_wo( N_A_diag, 0 );
    std::vector<int> coarse_elements_wo_id( N_A_diag, 0 );
    for( int i = 0 ; i < coarse_local_index_wo.size() ; i++ ){
        coarse_elements_wo[coarse_local_index_wo[i]] = 1;
        coarse_elements_wo_id[coarse_local_index_wo[i]] = i;
    }

    std::vector<bool> GI_coarse(All_col.size(), 0);
    std::vector<int> GI_coarse_id(All_col.size(), 0);
    for( int i = 0 ; i < GI.size() ; i++ ){                        
        GI_coarse[find_value(All_col, GI[i])] = 1;
        GI_coarse_id[find_value(All_col, GI[i])] = GCI[i];
    }


    int* rows = new int[All_col.size()];
    for (int i = 0; i < All_col.size(); i++) rows[i] = i;
    int* cols = new int[All_col.size()];
    for (int i = 0; i < All_col.size(); i++) cols[i] = All_col[i];

    CRSMatrix Af_diag = Af_CRS.get_csr_submatrix(rows, cols, All_col.size(), All_col.size());
    {
        std::vector<int> queue;
        std::vector<int> new_queue;
        std::vector<char> added_cells(Af_diag.n, 0);
        for( int i = 0 ; i < N_A_diag ; i++ ) {
            if( coarse_elements_wo[i] != 1 ) {
                {
                    auto add_neighbors_vel = [&Af_diag]( std::vector<int>& queue, std::vector<char>& added_cells, std::vector<int>& new_queue ){
                        for( auto v:queue ) {
                            for( int j = Af_diag.ia[v] ; j < Af_diag.ia[v+1] ; j++ ) {
                                if(Af_diag.ja[j]!=v && !added_cells[Af_diag.ja[j]]) {
                                    added_cells[Af_diag.ja[j]] = 1;
                                    new_queue.push_back(Af_diag.ja[j]);
                                }
                            }
                        }
                    };
                    auto process_neighbors = [i, &GI_coarse](std::vector<int>& queue, std::vector<int>& fine_coarse_neighbors_wo, int &N_found_coarse) {
                        for( auto v:queue ) {
                            if( GI_coarse[v] == 1 ) {
                                fine_coarse_neighbors_wo.push_back(v);
                                N_found_coarse++;
                            }
                        }
                    };
                    
                    
                    std::vector<int> fine_coarse_neighbors_wo;
                    
                    int N_found_coarse = 0;
                    queue.push_back(find_value(All_col, i + first_row));
                    while( N_found_coarse < 2 && queue.size() > 0 ) {
                        process_neighbors(queue, fine_coarse_neighbors_wo, N_found_coarse);
                        add_neighbors_vel(queue, added_cells, new_queue);
                        std::swap(new_queue, queue);
                        new_queue.clear();
                    }
                    queue.clear();
                    std::fill(added_cells.begin(), added_cells.end(), 0);


                    if(fine_coarse_neighbors_wo.size()!=0) {
                        for(int k = 0 ; k < fine_coarse_neighbors_wo.size() ; k++) {
                            int row = i + n_ex_sum;
                            int col = GI_coarse_id[fine_coarse_neighbors_wo[k]];
                            double value = 1.0/fine_coarse_neighbors_wo.size();
                            std::cout << rank << " " << row << " " << col << " " << value << "\n";
                            set_value(Prolong_coo, value, row, col);
                            //MatSetValues(Prolong,1,&row,1,&col,&value,ADD_VALUES);  
                        }
                    }
                }
            } else {
                int row = i + n_ex_sum;
                int col = coarse_elements_wo_id[i] + n_coarse_ex_sum;
                double value = 1.0;
                std::cout << rank << " " << row << " " << col << " " << value << "\n";
                set_value(Prolong_coo, value, row, col);
                //MatSetValues(Prolong,1,&row,1,&col,&value,ADD_VALUES);
            }
        }
    }

    CRSMatrix Prolong = Prolong_coo.coo_to_crs();
    return Prolong;
}

void print_prolong(CRSMatrix Prolong, int rank){
    COOMatrix Prolong_coo = Prolong.crs_to_coo();
    std::ofstream outfile("../prolong_rank_"+std::to_string(rank) + ".txt");
    for (int i = 0; i < Prolong.n; i++){
        int count = 0;
        for (int j = Prolong.ia[i]; j < Prolong.ia[i+1]; j++){
            for (int k = count; k < Prolong.ja[j]; k++) outfile << "0" << " ";
            outfile << "+" << " ";
            count = Prolong.ja[j] + 1;
        }
        
        for (int k = count; k < Prolong.m; k++) outfile << "0" << " ";
        outfile << "\n";
    }
    outfile.close();
}


void print_prolong_petsc(Mat A_wo_main_diag){
    int size;
	int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int m, n, M, N;
    MatGetSize(A_wo_main_diag, &M, &N);
    MatGetLocalSize(A_wo_main_diag, &m, &n);

    int n_ex_sum = 0;

    
    MPI_Exscan(&n,
                   &n_ex_sum,
                   1,
               MPI_INT,
               MPI_SUM,
               MPI_COMM_WORLD);


    MPI_Barrier(MPI_COMM_WORLD);


    Mat A_diag, A_off_diag;
    const int *garray;

    MatMPIAIJGetSeqAIJ(A_wo_main_diag, &A_diag, &A_off_diag, &garray);


    int N_A_diag;
    const int *ia_diag;
    const int *ja_diag;
    double *data_diag;
    PetscBool success_diag;
    
    MatGetRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_diag, &ia_diag, &ja_diag, &success_diag);
    MatSeqAIJGetArray(A_diag, &data_diag);
    
    int N_A_off_diag;
    const int *ia_off_diag;
    const int *ja_off_diag;
    double *data_off_diag;
    PetscBool success_off_diag;
    
    MatGetRowIJ(A_off_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_off_diag, &ia_off_diag, &ja_off_diag, &success_off_diag);
    MatSeqAIJGetArray(A_off_diag, &data_off_diag);

    int m_off_loc;
    int n_off_loc;
    MatGetLocalSize(A_off_diag, &n_off_loc, &m_off_loc);

    ///////////////////////////////////////////////////////////////////////////////////////////
    CRS_like_petsc A_com(N_A_diag, data_diag, ia_diag, ja_diag, 
                        m_off_loc, data_off_diag, ia_off_diag, ja_off_diag, garray,
                        n_ex_sum, N, M, rank);
    double* an;
    int* ian;
    int* jan;
    COOMatrix Prolong_coo(N_A_diag, N, an, ian, jan, 0);

    //Diag
    for (int i = 0; i < A_com.Diag->n; i++){
        for (int j = A_com.Diag->ia[i]; j < A_com.Diag->ia[i+1]; j++){
            int row = i;
            int col = A_com.Diag->ja[j] + A_com.first_row;
            double value = A_com.Diag->a[j];
            set_value(Prolong_coo, value, row, col);
        }
    }
    //Off_diag
    for (int i = 0; i < A_com.Off_diag_Cpr->n; i++){
        for (int j = A_com.Off_diag_Cpr->ia[i]; j < A_com.Off_diag_Cpr->ia[i+1]; j++){
            int row = i;    
            int col = A_com.garray[A_com.Off_diag_Cpr->ja[j]];
            double value = A_com.Off_diag_Cpr->a[j];
            set_value(Prolong_coo, value, row, col);
        }
    }
    CRSMatrix Prolong = Prolong_coo.coo_to_crs();
    print_prolong(Prolong, rank);
}