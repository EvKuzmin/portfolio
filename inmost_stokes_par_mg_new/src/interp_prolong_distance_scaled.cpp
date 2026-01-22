#include "interp_prolong_distance_scaled.hpp"
#include <iostream>

std::tuple<Mat, Mat, int, int, int, int> interp_prolong_distance_scaled( Mat A_wo_main_diag, std::vector<int> &coarse_local_index_wo, std::vector<int> &unknowns_local, std::vector< std::tuple< double, double, double > >& all_center_coordinates, std::vector< std::tuple< double, double, double > >& coarse_all_center_coordinates )
{
    int m, n, M, N;
    MatGetSize(A_wo_main_diag, &M, &N);
    MatGetLocalSize(A_wo_main_diag, &m, &n);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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

    auto dist = [](std::tuple< double, double, double > coord1, std::tuple< double, double, double > coord2){
        auto [x1,y1,z1] = coord1;
        auto [x2,y2,z2] = coord2;
        return std::sqrt( (x2-x1) * (x2-x1) + (y2-y1) * (y2-y1) + (z2-z1) * (z2-z1) );
    };

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
        for( int u = 0 ; u < unknowns_local.size()-1 ; u++ ) {
            for( int i = unknowns_local[u] ; i < unknowns_local[u+1] ; i++ ) {
                if( coarse_elements_wo[i] != 1 ) {
                    {
                        auto add_neighbors_vel = [&ia_diag, &ja_diag, &unknowns_local]( std::vector<int>& queue, std::vector<char>& added_cells, std::vector<int>& new_queue, int u ){
                            for( auto v:queue ) {
                                for( int j = ia_diag[v] ; j < ia_diag[v+1] ; j++ ) {
                                    if(ja_diag[j] >=unknowns_local[u] && ja_diag[j] < unknowns_local[u+1]) {
                                    if(ja_diag[j]!=v && !added_cells[ja_diag[j]]) {
                                        added_cells[ja_diag[j]] = 1;
                                        new_queue.push_back(ja_diag[j]);
                                    }
                                    }
                                }
                            }
                        };
                        auto process_neighbors = [i, &coarse_elements_wo, &coarse_elements_wo_id, &all_center_coordinates, &coarse_all_center_coordinates, dist](std::vector<int>& queue, std::vector<int>& fine_coarse_neighbors_wo, int &N_found_coarse, std::vector<double> &distance_neighbors_wo) {
                            for( auto v:queue ) {
                                if( coarse_elements_wo[v] == 1 ) {
                                    fine_coarse_neighbors_wo.push_back(v);
                                    // auto [x,y,z] = coarse_all_center_coordinates[coarse_elements_wo_id[v]];
                                    // std::cout << "x = " << x << " y = " << y << " z = " << z << "\n";
                                    // auto [x2,y2,z2] = all_center_coordinates[v];
                                    // std::cout << "x2 = " << x2 << " y2 = " << y2 << " z2 = " << z2 << "\n";
                                    distance_neighbors_wo.push_back( dist(coarse_all_center_coordinates[coarse_elements_wo_id[v]], all_center_coordinates[i]) );
                                    N_found_coarse++;
                                }
                            }
                        };

                        auto add_neighbors_vel_aggressive = [&ia_diag, &ja_diag, &unknowns_local]( std::vector<int>& queue, std::vector<char>& added_cells, std::vector<int>& new_queue, int u ){
                            for( auto v:queue ) {
                                for( int j = ia_diag[v] ; j < ia_diag[v+1] ; j++ ) {
                                    if(ja_diag[j]!=v && !added_cells[ja_diag[j]]) {
                                        added_cells[ja_diag[j]] = 1;
                                        new_queue.push_back(ja_diag[j]);
                                    }
                                }
                            }
                        };
                        auto process_neighbors_aggressive = [i, &coarse_elements_wo, &coarse_elements_wo_id, &all_center_coordinates, &coarse_all_center_coordinates, dist, &unknowns_local](std::vector<int>& queue, std::vector<int>& fine_coarse_neighbors_wo, int &N_found_coarse, std::vector<double> &distance_neighbors_wo, int u) {
                            for( auto v:queue ) {
                                if(v >=unknowns_local[u] && v < unknowns_local[u+1]) {
                                    if( coarse_elements_wo[v] == 1 ) {
                                        fine_coarse_neighbors_wo.push_back(v);
                                        distance_neighbors_wo.push_back( dist(coarse_all_center_coordinates[coarse_elements_wo_id[v]], all_center_coordinates[i]) );
                                        N_found_coarse++;
                                    }
                                }
                            }
                        };
                        
                        
                        std::vector<int> fine_coarse_neighbors_wo;
                        std::vector<double> distance_neighbors_wo;
                        
                        int N_found_coarse = 0;
                        queue.push_back(i);
                        while( N_found_coarse < 2 && queue.size() > 0 ) {
                            process_neighbors(queue, fine_coarse_neighbors_wo, N_found_coarse, distance_neighbors_wo);
                            add_neighbors_vel(queue, added_cells, new_queue, u);
                            std::swap(new_queue, queue);
                            new_queue.clear();
                        }
                        queue.clear();
                        std::fill(added_cells.begin(), added_cells.end(), 0);
                        // fine_coarse_neighbors_wo_N.push_back(fine_coarse_neighbors_wo.size());

                        N_found_coarse = 0;
                        if(fine_coarse_neighbors_wo.size()==0) {
                            queue.push_back(i);
                            while( N_found_coarse < 2 && queue.size() > 0 ) {
                                process_neighbors_aggressive(queue, fine_coarse_neighbors_wo, N_found_coarse, distance_neighbors_wo, u);
                                add_neighbors_vel_aggressive(queue, added_cells, new_queue, u);
                                std::swap(new_queue, queue);
                                new_queue.clear();
                            }
                            queue.clear();
                            std::fill(added_cells.begin(), added_cells.end(), 0);
                        }

                        if(fine_coarse_neighbors_wo.size()!=0) {
                            double sum = 0.0;
                            for( auto a : distance_neighbors_wo ) {
                                sum += 1.0 / a;
                            }
                            for(int k = 0 ; k < fine_coarse_neighbors_wo.size() ; k++) {
                                int row = i + n_ex_sum;
                                int col = coarse_elements_wo_id[fine_coarse_neighbors_wo[k]] + n_coarse_ex_sum;
                                double value = 1.0/distance_neighbors_wo[k] / sum;
                                MatSetValues(Prolong,1,&row,1,&col,&value,ADD_VALUES);
                            }
                        } else {
                            std::cerr << "Not found coarse element" << std::endl;
                            std::exit(0);
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
    }

    MatRestoreRowIJ(A_diag, 0, PETSC_FALSE, PETSC_FALSE, &N_A_diag, &ia_diag, &ja_diag, &success_diag);

    MatAssemblyBegin(Prolong, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(Prolong, MAT_FINAL_ASSEMBLY);

    Mat Restrict;

    MatTranspose(Prolong, MAT_INITIAL_MATRIX, &Restrict);

    return std::make_tuple( Prolong, Restrict, n_coarse, N_coarse, n_coarse_ex_sum, n_ex_sum );
}

std::tuple<Mat, Mat, int, int, int, int> interp_prolong_distance_scaled_cached( std::string name, Mat A_wo_main_diag, std::vector<int> &coarse_local_index_wo, std::vector<int> &unknowns_local, std::map< std::string, std::tuple<Mat, Mat, int, int, int, int> > &cache, std::vector< std::tuple< double, double, double > >& all_center_coordinates, std::vector< std::tuple< double, double, double > >& coarse_all_center_coordinates )
{
    if(cache.count(name) == 0) {
        cache[name] = interp_prolong_distance_scaled( A_wo_main_diag, coarse_local_index_wo, unknowns_local, all_center_coordinates, coarse_all_center_coordinates );
    }
    return cache.at(name);
}