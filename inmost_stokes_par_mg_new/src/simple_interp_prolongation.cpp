
#include"simple_interp_prolongation.hpp"

#include<vector>
#include<iostream>

std::tuple<Mat, Mat> simple_interp_prolongation( std::map< std::string, IS > &index_sequences, std::map< std::string, Mat > &submatrices, std::vector<int> &coarse_local_index_wo, SubmatrixManager &submatrix_map )
{
    PetscInt N_wo;
    ISGetLocalSize(index_sequences.at("without_dirichlet"), &N_wo);

    std::cout << "N_wo = " << N_wo << "\n";

    PetscInt N_wo_U, N_wo_P;
    ISGetLocalSize(index_sequences.at("Au"), &N_wo_U);
    ISGetLocalSize(index_sequences.at("Ap"), &N_wo_P);

    std::vector<char> coarse_elements_wo( N_wo, 0 );

    for( int i = 0 ; i < coarse_local_index_wo.size() ; i++ ){
        coarse_elements_wo[coarse_local_index_wo[i]] = 1;
    }

    std::vector<int> fine_coarse_neighbors_wo;
    std::vector<int> fine_coarse_neighbors_wo_N;

    {
        std::vector<char> added_cells(N_wo, 0);

        Mat A_diag, A_off_diag;
        const int *garray;

        MatMPIAIJGetSeqAIJ(submatrices.at("without_dirichlet"), &A_diag, &A_off_diag, &garray);

        PetscInt N_A_diag;
        const PetscInt *ia_diag;
        const PetscInt *ja_diag;
        PetscScalar *data_diag;
        PetscBool success_diag;

        MatGetRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_diag, &ia_diag, &ja_diag, &success_diag);
        MatSeqAIJGetArray(A_diag, &data_diag);

        PetscInt N_A_off_diag;
        const PetscInt *ia_off_diag;
        const PetscInt *ja_off_diag;
        PetscScalar *data_off_diag;
        PetscBool success_off_diag;

        MatGetRowIJ(A_off_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_off_diag, &ia_off_diag, &ja_off_diag, &success_off_diag);
        MatSeqAIJGetArray(A_off_diag, &data_off_diag);

        std::vector<int> queue;
        std::vector<int> new_queue;
        fine_coarse_neighbors_wo_N.push_back(0);

        for( int i = 0 ; i < N_wo_U ; i++ )
        {
            // std::cout << "i = " << i << " of " << N_wo_U << "\n";
            auto add_neighbors_vel = [&ia_diag, &ja_diag, N_wo_U]( std::vector<int>& queue, std::vector<char>& added_cells, std::vector<int>& new_queue ){
                for( auto v:queue ) {
                    for( int j = ia_diag[v] ; j < ia_diag[v+1] ; j++ ) {
                        if(ja_diag[j]!=v && !added_cells[ja_diag[j]]) {
                            added_cells[ja_diag[j]] = 1;
                            if( ja_diag[j] >= 0 && ja_diag[j] < N_wo_U ) {
                                new_queue.push_back(ja_diag[j]);
                            }
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
            if(coarse_elements_wo[i] != 1) {
                int N_found_coarse = 0;
                queue.push_back(i);
                while( N_found_coarse < 2 && queue.size() > 0 ) {
                    process_neighbors(queue, fine_coarse_neighbors_wo, N_found_coarse);
                    add_neighbors_vel(queue, added_cells, new_queue);
                    std::swap(new_queue, queue);
                    new_queue.clear();
                }
                queue.clear();
            }
            std::fill(added_cells.begin(), added_cells.end(), 0);
            fine_coarse_neighbors_wo_N.push_back(fine_coarse_neighbors_wo.size());
            
        }

        for( int i = N_wo_U ; i < N_wo_U + N_wo_P ; i++ )
        {
            // std::cout << "i = " << i << " of " << N_wo_U + N_wo_P << "\n";
            auto add_neighbors_vel = [&ia_diag, &ja_diag, N_wo_U, N_wo_P]( std::vector<int>& queue, std::vector<char>& added_cells, std::vector<int>& new_queue ){
                for( auto v:queue ) {
                    for( int j = ia_diag[v] ; j < ia_diag[v+1] ; j++ ) {
                        if(ja_diag[j]!=v && !added_cells[ja_diag[j]]) {
                            added_cells[ja_diag[j]] = 1;
                            if( ja_diag[j] >= N_wo_U && ja_diag[j] < N_wo_U + N_wo_P ) {
                                new_queue.push_back(ja_diag[j]);
                            }
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
            if(coarse_elements_wo[i] != 1) {
                int N_found_coarse = 0;
                queue.push_back(i);
                while( N_found_coarse < 2 && queue.size() > 0 ) {
                    process_neighbors(queue, fine_coarse_neighbors_wo, N_found_coarse);
                    add_neighbors_vel(queue, added_cells, new_queue);
                    std::swap(new_queue, queue);
                    new_queue.clear();
                }
                queue.clear();
            }
            std::fill(added_cells.begin(), added_cells.end(), 0);
            fine_coarse_neighbors_wo_N.push_back(fine_coarse_neighbors_wo.size());
            
        }

        MatRestoreRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_diag, &ia_diag, &ja_diag, &success_diag);
        MatRestoreRowIJ(A_off_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_off_diag, &ia_off_diag, &ja_off_diag, &success_off_diag);
    }

    // for(int i = 0 ; i < fine_coarse_neighbors_wo_N.size()-1 ; i++) {
    //     int N = fine_coarse_neighbors_wo_N[i+1] - fine_coarse_neighbors_wo_N[i];
    //     std::cout << "N = " << N << "\n";

    //     // if(N == 0) {
    //     //     if( coarse_elements_wo[i] != 1 ) {
    //     //         std::cerr << "zero element is not coarse\n";
    //     //         std::exit(0);
    //     //     }
    //     // }
    //     for(int j = fine_coarse_neighbors_wo_N[i] ; j < fine_coarse_neighbors_wo_N[i+1] ; j++) {
    //         std::cout << "   " << fine_coarse_neighbors_wo[j] << " ";
    //     }
    //     std::cout << "\n";
    // }

    Mat Afc_P, Acf_C;

    PetscInt N_wo_fine_local, N_wo_coarse_local, N_wo_fine_global, N_wo_coarse_global;
    ISGetLocalSize(index_sequences.at("without_dirichlet_fine"), &N_wo_fine_local);
    ISGetLocalSize(index_sequences.at("without_dirichlet_coarse"), &N_wo_coarse_local);
    ISGetSize(index_sequences.at("without_dirichlet_fine"), &N_wo_fine_global);
    ISGetSize(index_sequences.at("without_dirichlet_coarse"), &N_wo_coarse_global);

    MatCreate(PETSC_COMM_WORLD, &Afc_P);
    MatSetSizes(Afc_P, N_wo_fine_local, N_wo_coarse_local, N_wo_fine_global, N_wo_coarse_global);

    MatMPIAIJSetPreallocation(Afc_P, 9, NULL, 9, NULL);

    const auto& without_dirichlet_fine_map = submatrix_map.find_nodes("without_dirichlet_fine")[0]->map_to_parent;
    const auto& without_dirichlet_coarse_map = submatrix_map.find_nodes("without_dirichlet_coarse")[0]->map_from_parent;

    for(int i = 0 ; i < N_wo_fine_local ; i++) {
        // std::cout << "i = " << i << " of " << N_wo_fine_local << "\n";
        for( int j = fine_coarse_neighbors_wo_N[without_dirichlet_fine_map[i]] ; j < fine_coarse_neighbors_wo_N[without_dirichlet_fine_map[i]+1] ; j++) {
            int N_coarse = fine_coarse_neighbors_wo_N[without_dirichlet_fine_map[i]+1] - fine_coarse_neighbors_wo_N[without_dirichlet_fine_map[i]];
            int coarse_id_local = without_dirichlet_coarse_map[ fine_coarse_neighbors_wo[j] ];
            if( coarse_id_local == -1 ) {
                std::cerr << "wrong coarse id\n";
                std::exit(0);
            }
            int coarse_id = coarse_id_local + submatrix_map.find_nodes("without_dirichlet_coarse")[0]->size_ex_sum;
            int fine_id = i + submatrix_map.find_nodes("without_dirichlet_fine")[0]->size_ex_sum;
            double value = 1.0 / N_coarse;
            MatSetValues(Afc_P, 1, &fine_id, 1, &coarse_id, &value, INSERT_VALUES);
        }
    }

    MatAssemblyBegin(Afc_P, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Afc_P, MAT_FINAL_ASSEMBLY);

    MatTranspose(Afc_P, MAT_INITIAL_MATRIX, &Acf_C);

    return std::make_tuple( Afc_P, Acf_C );
}