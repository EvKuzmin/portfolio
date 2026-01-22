#include"simple_interp_prolong_improved.hpp"

#include<vector>
#include<iostream>

std::tuple<Mat, Mat, int, int, int, int> simple_interp_prolong_improved( Mat A_wo_main_diag, int face_unknowns_local_wo, std::vector<int> &coarse_local_index_wo )
{
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

    double average = 0.0;
    MPI_Allreduce(&n_coarse, &N_coarse, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
 
    Mat Prolong;

    MatCreate(PETSC_COMM_WORLD, &Prolong);
    MatSetSizes(Prolong, m, n_coarse, M, N_coarse);

    MatMPIAIJSetPreallocation(Prolong, 8, NULL, 3, NULL);

    Mat A_diag, A_off_diag;
    const int *garray;

    MatMPIAIJGetSeqAIJ(A_wo_main_diag, &A_diag, &A_off_diag, &garray);

    PetscInt N_A_diag;
    const PetscInt *ia_diag;
    const PetscInt *ja_diag;
    PetscScalar *data_diag;
    PetscBool success_diag;

    MatGetRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_diag, &ia_diag, &ja_diag, &success_diag);
    MatSeqAIJGetArray(A_diag, &data_diag);

    std::vector<char> coarse_elements_wo( N_A_diag, 0 );
    std::vector<int> coarse_elements_wo_id( N_A_diag, 0 );
    // {

    //     std::sort(coarse_local_index_wo.begin(), coarse_local_index_wo.end());
    //     bool containsDuplicates = (std::unique(coarse_local_index_wo.begin(), coarse_local_index_wo.end()) != coarse_local_index_wo.end());
    //     std::cout << containsDuplicates << "\n";
    //     std::exit(0);
    // }
    for( int i = 0 ; i < coarse_local_index_wo.size() ; i++ ){
        coarse_elements_wo[coarse_local_index_wo[i]] = 1;
        coarse_elements_wo_id[coarse_local_index_wo[i]] = i;
    }

    {
        std::vector<int> queue;
        std::vector<int> new_queue;
        std::vector<char> added_cells(N_A_diag, 0);
        // std::vector<int> fine_coarse_neighbors_wo_N;
        // fine_coarse_neighbors_wo_N.push_back(0);
        for( int i = 0 ; i < N_A_diag ; i++ ) {
            if( coarse_elements_wo[i] != 1 ) {
                {
                    auto add_neighbors_vel = [&ia_diag, &ja_diag]( std::vector<int>& queue, std::vector<char>& added_cells, std::vector<int>& new_queue ){
                        for( auto v:queue ) {
                            for( int j = ia_diag[v] ; j < ia_diag[v+1] ; j++ ) {
                                if(ja_diag[j]!=v && !added_cells[ja_diag[j]]) {
                                    added_cells[ja_diag[j]] = 1;
                                    new_queue.push_back(ja_diag[j]);
                                }
                            }
                        }
                    };
                    auto process_neighbors = [i, &coarse_elements_wo](std::vector<int>& queue, std::vector<int>& fine_coarse_neighbors_wo, int &N_found_coarse) {
                        for( auto v:queue ) {
                            if( coarse_elements_wo[v] == 1 ) {
                                fine_coarse_neighbors_wo.push_back(v);
                                N_found_coarse++;
                            }
                        }
                    };
                    
                    
                    std::vector<int> fine_coarse_neighbors_wo;
                    
                    int N_found_coarse = 0;
                    queue.push_back(i);
                    while( N_found_coarse < 2 && queue.size() > 0 ) {
                        process_neighbors(queue, fine_coarse_neighbors_wo, N_found_coarse);
                        add_neighbors_vel(queue, added_cells, new_queue);
                        std::swap(new_queue, queue);
                        new_queue.clear();
                    }
                    queue.clear();
                    std::fill(added_cells.begin(), added_cells.end(), 0);
                    // fine_coarse_neighbors_wo_N.push_back(fine_coarse_neighbors_wo.size());

                    if(fine_coarse_neighbors_wo.size()!=0) {
                        for(int k = 0 ; k < fine_coarse_neighbors_wo.size() ; k++) {
                            int row = i + n_ex_sum;
                            int col = coarse_elements_wo_id[fine_coarse_neighbors_wo[k]] + n_coarse_ex_sum;
                            double value = 1.0/fine_coarse_neighbors_wo.size();
                            MatSetValues(Prolong,1,&row,1,&col,&value,ADD_VALUES);
                        }
                    }
                }
            } else {
                int row = i + n_ex_sum;
                int col = coarse_elements_wo_id[i] + n_coarse_ex_sum;
                double value = 1.0;
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