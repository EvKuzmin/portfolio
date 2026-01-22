
#include"mesh_coarsen.hpp"
#include<iostream>

// #include "aij.h"
#include "../src/ksp/pc/impls/gamg/gamg.h"



std::tuple< std::vector<int>, std::vector<int> > mesh_coarsen(Mesh *m, Mat A, std::map< std::string, Mat > &submatrices, SubmatrixManager &submatrix_map, std::vector<unsigned int> &cell_id_local_to_handle, int face_unknowns_local, int lapl_matrix_size_sum)
{
    TagInteger coarse_level = TagInteger( m->GetTag( "coarse_level" ) );
    TagInteger face_id_local = TagInteger( m->GetTag( "face_id_local" ) );
    TagInteger cell_id_local = TagInteger( m->GetTag( "cell_id_local" ) );

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Mat A_lapl;

    MatMatMult( submatrices.at("Apu"), submatrices.at("Aup"), MAT_INITIAL_MATRIX, PETSC_DEFAULT, &A_lapl );

    int A_lapl_N1, A_lapl_N2, A_full_N1, A_full_N2;

    MatGetOwnershipRange(A_lapl, &A_lapl_N1, &A_lapl_N2);
    MatGetOwnershipRange(A, &A_full_N1, &A_full_N2);

    std::cout << "rank = " << rank << " A_lapl_N1 = " << A_lapl_N1 << " A_lapl_N2 = " << A_lapl_N2 << "\n";


    MatCoarsen A_coarsen;
    Mat a_Gmat;

    MatCoarsenCreate(MPI_COMM_WORLD, &A_coarsen);
    MatCoarsenSetType(A_coarsen, MATCOARSENMISK);
    MatCoarsenSetMaximumIterations(A_coarsen, 100);
    // MatCoarsenSetThreshold(A_coarsen, vfilter);   
    MatCoarsenMISKSetDistance(A_coarsen, 2);
    MatCoarsenSetStrictAggs(A_coarsen, PETSC_TRUE);
    

    MatCreateGraph(A_lapl, PETSC_FALSE, PETSC_TRUE, -1, 0, nullptr, &a_Gmat);
    MatCoarsenSetAdjacency(A_coarsen, a_Gmat);

    MatCoarsenApply(A_coarsen);
    PetscCoarsenData *agg_lists;
    MatCoarsenGetData(A_coarsen, &agg_lists);
    MatCoarsenDestroy(&A_coarsen);


    int agg_lists_N = agg_lists->size;


    std::vector<int> coarse_local_index;
    std::vector<int> fine_local_index;

    std::vector<char> all_index(A_full_N2 - A_full_N1, 0);

    std::vector<int> gid_list;
    // SubmatrixManager::Node *wo_node = submatrix_map.find_nodes("without_dirichlet")[0];
    // int full_count = 0;
    for( int i = 0 ; i < agg_lists_N ; i++ ) {
        // full_count++;
        PetscCDIntNd *item = agg_lists->array[i];
        double x_avg = 0.0;
        double y_avg = 0.0;
        double z_avg = 0.0;
        int count = 0;
        bool near_interprocess = false;
        if(item) {
            while( item != NULL ) {
                Storage::real cnt1[3];
                if(item->gid < A_lapl_N2 & item->gid >= A_lapl_N1) {
                    Cell c = Cell( m, cell_id_local_to_handle[ item->gid - A_lapl_N1] );
                    // c.Integer(coarse_level) = full_count;
                    c.Barycenter(cnt1);
                    x_avg += cnt1[0];
                    y_avg += cnt1[1];
                    z_avg += cnt1[2];
                    count++;
                } else {
                    near_interprocess = true;
                }
                item = item->next;
            }
            x_avg /= count;
            y_avg /= count;
            z_avg /= count;
        }
        item = agg_lists->array[i];
        double dist = 1e100;
        int index = -1;
        if(item) { // && (count > 2 || near_interprocess)) {
            while( item != NULL ) {
                Storage::real cnt1[3];
                if(item->gid < A_lapl_N2 & item->gid >= A_lapl_N1) {
                    Cell c = Cell( m, cell_id_local_to_handle[ item->gid - A_lapl_N1] );
                    c.Barycenter(cnt1);
                    double dist_new = std::sqrt( std::pow(x_avg - cnt1[0], 2.0) + std::pow(y_avg - cnt1[1], 2.0) + std::pow(z_avg - cnt1[2], 2.0) );
                    if( dist_new < dist ) {
                        dist = dist_new;
                        index = item->gid;
                    }
                }
                item = item->next;
            }
            if(index != -1) {
                gid_list.push_back(index);
            }
        }
    }

    double h;

    {
        Mesh::iteratorCell cell = m->BeginCell();
        ElementArray<Face> faces = cell->getFaces();
        Face back = faces[0].getAsFace();
        Face front = faces[1].getAsFace();
        Storage::real cnt1[3];
        Storage::real cnt2[3];
        back.Barycenter(cnt1);
        front.Barycenter(cnt2);
        h = cnt2[0] - cnt1[0];
    }

    gid_list.clear();
    for( Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell ) {
        if( cell->GetStatus() != Element::Ghost ) {
            Storage::real cnt[3];
            cell->Barycenter(cnt);
            int i = cnt[0] / h;
            int j = cnt[1] / h;
            int k = cnt[2] / h;
            // if((i+j+k)%2 ==0 ){
            //     gid_list.push_back(cell->Integer(cell_id_local)+lapl_matrix_size_sum);
            // }
            if(i%2 == 0 && j%2 == 0 && k%2 == 0 ){
                gid_list.push_back(cell->Integer(cell_id_local)+lapl_matrix_size_sum);
            }
        }
    }

    for( int i = 0 ; i < gid_list.size() ; i++ ) {
        int gid = gid_list[i];
        if(gid >= A_lapl_N2 || gid < A_lapl_N1) {
            std::cerr << "rank = " << rank << " gid = " << gid << "\n";
            std::abort();
        }
        Cell c = Cell( m, cell_id_local_to_handle[ gid - A_lapl_N1] );
        c.Integer(coarse_level) = 1;
        ElementArray<Face> faces = c.getFaces();
        Face back = faces[0].getAsFace();
        Face front = faces[1].getAsFace();
        Face left = faces[2].getAsFace();
        Face right = faces[3].getAsFace();
        Face bottom = faces[4].getAsFace();
        Face top = faces[5].getAsFace();
        if( back.GetStatus() != Element::Ghost ) {
            coarse_local_index.push_back( back.Integer( face_id_local ) );
            all_index[ back.Integer( face_id_local ) ] = 1;
        } else if (front.GetStatus() != Element::Ghost) {
            coarse_local_index.push_back( front.Integer( face_id_local ) );
            all_index[ front.Integer( face_id_local ) ] = 1;
        }
        if( left.GetStatus() != Element::Ghost ) {
            coarse_local_index.push_back( left.Integer( face_id_local ) );
            all_index[ left.Integer( face_id_local ) ] = 1;
        } else if (right.GetStatus() != Element::Ghost) {
            coarse_local_index.push_back( right.Integer( face_id_local ) );
            all_index[ right.Integer( face_id_local ) ] = 1;
        }
        if( bottom.GetStatus() != Element::Ghost ) {
            coarse_local_index.push_back( bottom.Integer( face_id_local ) );
            all_index[ bottom.Integer( face_id_local ) ] = 1;
        } else if (top.GetStatus() != Element::Ghost) {
            coarse_local_index.push_back( top.Integer( face_id_local ) );
            all_index[ top.Integer( face_id_local ) ] = 1;
        }

        // if( back.GetStatus() != Element::Ghost ) {
        //     coarse_local_index.push_back( back.Integer( face_id_local ) );
        //     all_index[ back.Integer( face_id_local ) ] = 1;
        // } 
        // if( front.GetStatus() != Element::Ghost ) {
        //     coarse_local_index.push_back( front.Integer( face_id_local ) );
        //     all_index[ front.Integer( face_id_local ) ] = 1;
        // } 
        // if( left.GetStatus() != Element::Ghost ) {
        //     coarse_local_index.push_back( left.Integer( face_id_local ) );
        //     all_index[ left.Integer( face_id_local ) ] = 1;
        // }
        // if( right.GetStatus() != Element::Ghost ) {
        //     coarse_local_index.push_back( right.Integer( face_id_local ) );
        //     all_index[ right.Integer( face_id_local ) ] = 1;
        // }
        // if( bottom.GetStatus() != Element::Ghost ) {
        //     coarse_local_index.push_back( bottom.Integer( face_id_local ) );
        //     all_index[ bottom.Integer( face_id_local ) ] = 1;
        // } 
        // if( top.GetStatus() != Element::Ghost ) {
        //     coarse_local_index.push_back( top.Integer( face_id_local ) );
        //     all_index[ top.Integer( face_id_local ) ] = 1;
        // } 
    }

    

    int coarse_matrix_local_size_U = coarse_local_index.size();

    for( int i = 0 ; i < gid_list.size() ; i++ ) {
        int gid = gid_list[i];
        
        if(gid >= A_lapl_N2 || gid < A_lapl_N1) {
            std::cerr << "rank = " << rank << " gid = " << gid << "\n";
            std::abort();
        }
        Cell c = Cell( m, cell_id_local_to_handle[ gid - A_lapl_N1] );
        coarse_local_index.push_back( c.Integer( cell_id_local ) + face_unknowns_local );
        all_index[ c.Integer( cell_id_local ) + face_unknowns_local ] = 1;
    }

    std::sort(coarse_local_index.begin(), coarse_local_index.end());

    int coarse_matrix_local_size_P = coarse_local_index.size() - coarse_matrix_local_size_U;

    for( int i = 0 ; i < all_index.size() ; i++ ) {
        if(!all_index[i]) {
            fine_local_index.push_back( i );
        }
    }

    return std::make_tuple( coarse_local_index, fine_local_index );
}