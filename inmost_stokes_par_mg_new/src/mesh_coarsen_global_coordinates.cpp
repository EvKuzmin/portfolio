#include"mesh_coarsen_global_coordinates.hpp"

#include<iostream>

// #include "aij.h"
#include "../src/ksp/pc/impls/gamg/gamg.h"
#include <unordered_map>
#include <tuple>

struct GraphProperties {
    int indegree;
    int outdegree; 
    int weighted;
    std::vector<int> sources;
    std::vector<int> destinations;
};

struct RecvArrayProperties {
    std::vector<int> sources;
    std::vector<int> recvcounts;
    std::vector<int> rdispls;
};

// Petsc coarsening gets aggregates which are partially on remote processes. This function gets one node in aggregate closest to geometrical center of aggregate.
std::tuple< std::vector<int>, 
            std::vector<int>, 
            std::vector< std::tuple< double, double, double > >, 
            std::vector< std::tuple< double, double, double > >, 
            std::vector< std::tuple< int, int, int, int, int, int > >,
            int,
            int, 
            std::vector<int> >
            mesh_coarsen_global_coordinates(Mat A_full, 
                           Mat A_lapl, 
                           int face_unknowns_local, 
                           int cell_unknowns_local, 
                           std::vector< std::tuple< double, double, double > > &cell_center_coordinates, 
                           std::vector< std::tuple< int, int, int, int, int, int > > &cell_to_face_local,
                           std::vector<VariableOnMesh> &variables, 
                           std::vector<int> unknowns_local,
                           int MISKDistance)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int A_lapl_N1, A_lapl_N2, A_full_N1, A_full_N2;

    MatGetOwnershipRange(A_lapl, &A_lapl_N1, &A_lapl_N2);
    MatGetOwnershipRange(A_full, &A_full_N1, &A_full_N2);

    std::vector< int > A_lapl_owner_range;

    { // Gather ownership ranges
        std::vector< std::tuple<int, int> > A_lapl_owner_range_vec(size);
        std::tuple< int, int > A_lapl_local_ownership_range{A_lapl_N1, A_lapl_N2};

        MPI_Allgather(
        &A_lapl_local_ownership_range,
        2,
        MPI_INTEGER,
        A_lapl_owner_range_vec.data(),
        2,
        MPI_INTEGER,
        MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);

        for( auto v : A_lapl_owner_range_vec ) {
            A_lapl_owner_range.push_back(std::get<0>(v));
        }
        A_lapl_owner_range.push_back( std::get<1>( A_lapl_owner_range_vec.back() ) );
    }

    PetscCoarsenData *agg_lists;
    {
        MatCoarsen A_coarsen;
        Mat a_Gmat;

        MatCoarsenCreate(MPI_COMM_WORLD, &A_coarsen);
        MatCoarsenSetType(A_coarsen, MATCOARSENMISK);
        MatCoarsenSetMaximumIterations(A_coarsen, 100);
        // MatCoarsenSetThreshold(A_coarsen, vfilter);   
        MatCoarsenMISKSetDistance(A_coarsen, MISKDistance);
        MatCoarsenSetStrictAggs(A_coarsen, PETSC_TRUE);
        

        MatCreateGraph(A_lapl, PETSC_FALSE, PETSC_TRUE, -1, 0, nullptr, &a_Gmat);
        MatCoarsenSetAdjacency(A_coarsen, a_Gmat);

        MatCoarsenApply(A_coarsen);
        
        MatCoarsenGetData(A_coarsen, &agg_lists);
        MatCoarsenDestroy(&A_coarsen);
        MatDestroy(&a_Gmat);
    }


    int agg_lists_N = agg_lists->size;

    // SubmatrixManager::Node *wo_node = submatrix_map.find_nodes("without_dirichlet")[0];
    // int full_count = 0;
    std::set< int > neighbor_gids; // off process gids
    for( int i = 0 ; i < agg_lists_N ; i++ ) {
        PetscCDIntNd *item = agg_lists->array[i];
        if(item) {
            while( item != NULL ) {
                if(item->gid < A_lapl_N2 && item->gid >= A_lapl_N1) {
                } else {
                    neighbor_gids.insert(item->gid);
                }
                item = item->next;
            }
        }
    }

    std::unordered_map< int, std::vector< int > > gid_to_get_from_rank; // map rank to array of gids to get coordinates

    for( auto v:neighbor_gids ) {
        auto it = std::upper_bound(A_lapl_owner_range.begin(), A_lapl_owner_range.end(), v);
        if (it != A_lapl_owner_range.end()) {
            gid_to_get_from_rank[it - A_lapl_owner_range.begin()-1].push_back(v);
        }
    }

    MPI_Comm send_size_comm; // Graph for sending size of gid arrays

    GraphProperties send_size_comm_properties;

    { // Prepare size transfer graph
        std::vector< int > sources;
        // for( auto v:gid_to_get_from_rank ) {
        //     sources.push_back(v.first);
        // }
        int degrees = gid_to_get_from_rank.size();

        std::vector<int> destinations;
        for( auto v:gid_to_get_from_rank ) {
            destinations.push_back(v.first);
            // std::cout << "rank = " << rank << " dest = " << v.first << " number = " << v.second.size() << "\n";
        }
        // std::cout << "rank = " << rank << " degrees = " << degrees << " destinations.data() = " << destinations.data() << " &send_size_comm = " << &send_size_comm << "\n";  
        MPI_Barrier(MPI_COMM_WORLD);
        int n;
        if(gid_to_get_from_rank.size() != 0) {
            n = 1;
        } else {
            n = 0;
        }
        int error = MPI_Dist_graph_create(MPI_COMM_WORLD, n, &rank, &degrees, destinations.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, 0, &send_size_comm);
        if(error != MPI_SUCCESS) {
            std::cerr << "MPI_Dist_graph_create error = " << error << "\n";
            MPI_Barrier(MPI_COMM_WORLD);
            std::exit(0);
        }
        

        error = MPI_Dist_graph_neighbors_count(send_size_comm, &send_size_comm_properties.indegree, &send_size_comm_properties.outdegree, &send_size_comm_properties.weighted);

        if(error != MPI_SUCCESS) {
            std::cerr << "MPI_Dist_graph_neighbors_count error = " << error << "\n";
            MPI_Barrier(MPI_COMM_WORLD);
            std::exit(0);
        }
        send_size_comm_properties.sources.resize(send_size_comm_properties.indegree);
        send_size_comm_properties.destinations.resize(send_size_comm_properties.outdegree);

        MPI_Dist_graph_neighbors(send_size_comm, send_size_comm_properties.indegree, send_size_comm_properties.sources.data(), MPI_UNWEIGHTED, send_size_comm_properties.outdegree, send_size_comm_properties.destinations.data(), MPI_UNWEIGHTED);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    std::vector<int> gid_sizes_to_receive; // array sizes this rank need to receive

    { // Send gid sizes 

        std::vector<int> gid_sizes_to_send;

        for( int i = 0 ; i < send_size_comm_properties.outdegree ; i++ ) {
            gid_sizes_to_send.push_back(gid_to_get_from_rank.at(send_size_comm_properties.destinations[i]).size());
            // std::cout << "rank = " << rank << " gid_sizes_to_send = " << gid_sizes_to_send.back() << "\n";
        }
        gid_sizes_to_receive.resize(send_size_comm_properties.indegree);

        // error = MPI_Neighbor_allgather(gid_sizes_to_send.data(), 1, MPI_INTEGER, gid_sizes_to_receive.data(), 1, MPI_INTEGER, send_size_comm);

        std::vector<int> sendcounts(send_size_comm_properties.outdegree, 1);

        std::vector<int> sdispls;
        sdispls.push_back(0);
        for( int i = 0 ; i < sendcounts.size() ; i++ ) {
            sdispls.push_back(sdispls.back() + sendcounts[i]);
        }

        std::vector<int> recvcounts(send_size_comm_properties.indegree, 1);

        std::vector<int> rdispls;
        rdispls.push_back(0);
        for( int i = 0 ; i < recvcounts.size() ; i++ ) {
            rdispls.push_back(rdispls.back() + recvcounts[i]);
        }

        int error = MPI_Neighbor_alltoallv(gid_sizes_to_send.data(), sendcounts.data(), sdispls.data(), MPI_INTEGER, gid_sizes_to_receive.data(), recvcounts.data(), rdispls.data(), MPI_INTEGER, send_size_comm);
        if(error != MPI_SUCCESS) {
            std::cerr << "MPI_Neighbor_alltoallv error = " << error << "\n";
            MPI_Barrier(MPI_COMM_WORLD);
            std::exit(0);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    std::vector<int> gid_list_to_send; // array of gid this rank need to send
    RecvArrayProperties gid_list_to_send_props;

    { // Send gid list to send back
      // Same graph used
        std::vector<int> gid_to_send;
        std::vector<int> sendcounts;

        for( int i = 0 ; i < send_size_comm_properties.outdegree ; i++ ) {
            sendcounts.push_back( gid_to_get_from_rank.at( send_size_comm_properties.destinations[i] ).size() );
            for( auto v2:gid_to_get_from_rank.at( send_size_comm_properties.destinations[i] ) ){
                gid_to_send.push_back(v2);
            }
        }

        // for( auto v:gid_to_get_from_rank ) {
        //     sendcounts.push_back( v.second.size() );
        //     for( auto v2:v.second ){
        //         gid_to_send.push_back(v2);
        //     }
        //     std::cout << "rank = " << rank << " dest = " << v.first << " number = " << v.second.size() << "\n";
        // }

        std::vector<int> sdispls;
        sdispls.push_back(0);
        for( int i = 0 ; i < sendcounts.size() ; i++ ) {
            sdispls.push_back(sdispls.back() + sendcounts[i]);
        }

        std::vector<int> rdispls;
        rdispls.push_back(0);
        int size_to_receive = 0;
        for( int i = 0 ; i < gid_sizes_to_receive.size() ; i++ ) {
            rdispls.push_back(rdispls.back() + gid_sizes_to_receive[i]);
            size_to_receive += gid_sizes_to_receive[i];
        }

        gid_list_to_send.resize(size_to_receive);
        

        int error = MPI_Neighbor_alltoallv(gid_to_send.data(), sendcounts.data(), sdispls.data(), MPI_INTEGER, gid_list_to_send.data(), gid_sizes_to_receive.data(), rdispls.data(), MPI_INTEGER, send_size_comm);
        if(error != MPI_SUCCESS) {
            std::cerr << "MPI_Neighbor_alltoallv Send gid list to send back error = " << error << "\n";
            MPI_Barrier(MPI_COMM_WORLD);
            std::exit(0);
        }

        gid_list_to_send_props.rdispls = rdispls;
        gid_list_to_send_props.recvcounts = gid_sizes_to_receive;
        gid_list_to_send_props.sources = send_size_comm_properties.sources;
    }

    MPI_Comm send_coordinate_comm; // Graph to send back coordinates

    GraphProperties send_coordinate_comm_properties; 

    { // Prepare coordinate transfer graph
        std::vector< int > sources;
        std::vector< int > degrees;
        std::vector< int > destinations;
        for( auto v:gid_to_get_from_rank ) {
            sources.push_back(v.first);
            degrees.push_back(1);
            destinations.push_back(rank);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        int error = MPI_Dist_graph_create(MPI_COMM_WORLD, sources.size(), sources.data(), degrees.data(), destinations.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, 0, &send_coordinate_comm);
        if(error != MPI_SUCCESS) {
            std::cerr << "MPI_Dist_graph_create error = " << error << "\n";
            MPI_Barrier(MPI_COMM_WORLD);
            std::exit(0);
        }

        error = MPI_Dist_graph_neighbors_count(send_coordinate_comm, &send_coordinate_comm_properties.indegree, &send_coordinate_comm_properties.outdegree, &send_coordinate_comm_properties.weighted);

        if(error != MPI_SUCCESS) {
            std::cerr << "MPI_Dist_graph_neighbors_count error = " << error << "\n";
            MPI_Barrier(MPI_COMM_WORLD);
            std::exit(0);
        }
        send_coordinate_comm_properties.sources.resize(send_coordinate_comm_properties.indegree);
        send_coordinate_comm_properties.destinations.resize(send_coordinate_comm_properties.outdegree);

        MPI_Dist_graph_neighbors(send_coordinate_comm, send_coordinate_comm_properties.indegree, send_coordinate_comm_properties.sources.data(), MPI_UNWEIGHTED, send_coordinate_comm_properties.outdegree, send_coordinate_comm_properties.destinations.data(), MPI_UNWEIGHTED);
    }

    std::vector< std::tuple< double, double, double > > coord_to_receive; // Coordinates of neighbour cell centers
    RecvArrayProperties coord_to_receive_props;
    
    { // Coordinate transfer
        std::vector< std::tuple< double, double, double > > coord_to_send;
        std::vector< int > sendcounts, sendcounts_double;

        for( int i = 0 ; i < send_coordinate_comm_properties.outdegree ; i++ ) {
            auto it = std::find( gid_list_to_send_props.sources.begin(), gid_list_to_send_props.sources.end(), send_coordinate_comm_properties.destinations[i] );
            int i_gid_list = it - gid_list_to_send_props.sources.begin();
            sendcounts.push_back( gid_list_to_send_props.recvcounts[i_gid_list] );

            sendcounts_double.push_back( gid_list_to_send_props.recvcounts[i_gid_list] * sizeof( std::tuple< double, double, double > ) / 8 );

            for( int j = gid_list_to_send_props.rdispls[i_gid_list] ; j < gid_list_to_send_props.rdispls[i_gid_list] + sendcounts.back() ; j++ ) {
                coord_to_send.push_back(  cell_center_coordinates[ gid_list_to_send[j] - A_lapl_N1 ] );
            }
        }

        std::vector<int> sdispls;
        std::vector<int> sdispls_double;
        sdispls.push_back(0);
        sdispls_double.push_back(0);
        for( int i = 0 ; i < sendcounts.size() ; i++ ) {
            sdispls.push_back(sdispls.back() + sendcounts[i]);
            sdispls_double.push_back(sdispls_double.back() + sendcounts_double[i]);
        }

        std::vector<int> recvcounts(send_coordinate_comm_properties.indegree);
        std::vector<int> recvcounts_double(send_coordinate_comm_properties.indegree);

        for( int i = 0 ; i < send_coordinate_comm_properties.indegree ; i++ ) {
            recvcounts[i] = gid_to_get_from_rank.at(send_coordinate_comm_properties.sources[i]).size();
            recvcounts_double[i] = gid_to_get_from_rank.at(send_coordinate_comm_properties.sources[i]).size() * sizeof( std::tuple< double, double, double > ) / 8;
        }

        std::vector<int> rdispls, rdispls_double;
        rdispls.push_back(0);
        rdispls_double.push_back(0);
        int size_to_receive = 0;
        for( int i = 0 ; i < recvcounts.size() ; i++ ) {
            rdispls.push_back(rdispls.back() + recvcounts[i]);
            rdispls_double.push_back(rdispls_double.back() + recvcounts_double[i]);
            size_to_receive += recvcounts[i];
        }

        coord_to_receive.resize(size_to_receive);

        int error = MPI_Neighbor_alltoallv(coord_to_send.data(), sendcounts_double.data(), sdispls_double.data(), MPI_DOUBLE, coord_to_receive.data(), recvcounts_double.data(), rdispls_double.data(), MPI_DOUBLE, send_coordinate_comm);
        if(error != MPI_SUCCESS) {
            std::cerr << "MPI_Neighbor_alltoallv send_coordinate_comm error = " << error << "\n";
            MPI_Barrier(MPI_COMM_WORLD);
            std::exit(0);
        }

        coord_to_receive_props.rdispls = rdispls;
        coord_to_receive_props.recvcounts = recvcounts;
        coord_to_receive_props.sources = send_coordinate_comm_properties.sources;

    }

    std::unordered_map< int, std::tuple< double, double, double > > gid_to_neighb_coord;

    for( int i = 0 ; i < coord_to_receive_props.sources.size() ; i++ ) {
        auto it = std::find( gid_list_to_send_props.sources.begin(), gid_list_to_send_props.sources.end(), send_coordinate_comm_properties.destinations[i] );
        int i_gid_list = it - gid_list_to_send_props.sources.begin();
        std::vector<int> &gids = gid_to_get_from_rank.at(coord_to_receive_props.sources[i]);
        for(int j = 0 ; j < gids.size() ; j++) {
            gid_to_neighb_coord[gids[j]] = coord_to_receive[ coord_to_receive_props.rdispls[i]+j ];
            // std::cout << "rank = " << rank 
            //           << " gid = " << gids[j] 
            //           << " received coords = " << std::get<0>(coord_to_receive[ coord_to_receive_props.rdispls[i]+j ])
            //           << " " << std::get<1>(coord_to_receive[ coord_to_receive_props.rdispls[i]+j ])
            //           << " " << std::get<2>(coord_to_receive[ coord_to_receive_props.rdispls[i]+j ]) << "\n";
        }
    }









    std::vector<int> gid_list;

    std::unordered_map< int, std::vector<int> > gid_to_send_to_rank;
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
                if(item->gid < A_lapl_N2 && item->gid >= A_lapl_N1) {
                    auto [x_val, y_val, z_val] = cell_center_coordinates[ item->gid - A_lapl_N1 ];
                    x_avg += x_val;
                    y_avg += y_val;
                    z_avg += z_val;
                    count++;
                } else {
                    auto [x_val, y_val, z_val] = gid_to_neighb_coord.at(item->gid);
                    x_avg += x_val;
                    y_avg += y_val;
                    z_avg += z_val;
                    count++;
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
                if(item->gid < A_lapl_N2 && item->gid >= A_lapl_N1) {
                    auto [x_val, y_val, z_val] = cell_center_coordinates[ item->gid - A_lapl_N1 ];
                    double dist_new = std::sqrt( std::pow(x_avg - x_val, 2.0) + std::pow(y_avg - y_val, 2.0) + std::pow(z_avg - z_val, 2.0) );
                    if( dist_new < dist ) {
                        dist = dist_new;
                        index = item->gid;
                    }
                } else {
                    auto [x_val, y_val, z_val] = gid_to_neighb_coord.at(item->gid);
                    double dist_new = std::sqrt( std::pow(x_avg - x_val, 2.0) + std::pow(y_avg - y_val, 2.0) + std::pow(z_avg - z_val, 2.0) );
                    if( dist_new < dist ) {
                        dist = dist_new;
                        index = item->gid;
                    }
                }
                item = item->next;
            }
            if(index != -1) {
                if(index < A_lapl_N2 && index >= A_lapl_N1) {
                    gid_list.push_back(index);
                } else {
                    auto it = std::upper_bound(A_lapl_owner_range.begin(), A_lapl_owner_range.end(), index);
                    if (it != A_lapl_owner_range.end()) {
                        gid_to_send_to_rank[it - A_lapl_owner_range.begin()-1].push_back(index);
                    }
                    // std::cout << "Coarse node in neighbor process\n";
                }
            }
        }
    }







    MPI_Comm send_coarse_gid_comm;

    GraphProperties send_coarse_gid_comm_properties;

    { // Prepare coarse gid transfer graph
        std::vector< int > sources;
        std::vector< int > degrees;
        std::vector< int > destinations;
        if( gid_to_send_to_rank.size() != 0 ) {
            sources.push_back(rank);
            degrees.push_back(gid_to_send_to_rank.size());
        }

        for( auto v:gid_to_send_to_rank ) {
            destinations.push_back(v.first);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        int error = MPI_Dist_graph_create(MPI_COMM_WORLD, sources.size(), sources.data(), degrees.data(), destinations.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, 0, &send_coarse_gid_comm);
        if(error != MPI_SUCCESS) {
            std::cerr << "MPI_Dist_graph_create error = " << error << "\n";
            MPI_Barrier(MPI_COMM_WORLD);
            std::exit(0);
        }

        error = MPI_Dist_graph_neighbors_count(send_coarse_gid_comm, &send_coarse_gid_comm_properties.indegree, &send_coarse_gid_comm_properties.outdegree, &send_coarse_gid_comm_properties.weighted);

        if(error != MPI_SUCCESS) {
            std::cerr << "MPI_Dist_graph_neighbors_count error = " << error << "\n";
            MPI_Barrier(MPI_COMM_WORLD);
            std::exit(0);
        }
        send_coarse_gid_comm_properties.sources.resize(send_coarse_gid_comm_properties.indegree);
        send_coarse_gid_comm_properties.destinations.resize(send_coarse_gid_comm_properties.outdegree);

        MPI_Dist_graph_neighbors(send_coarse_gid_comm, send_coarse_gid_comm_properties.indegree, send_coarse_gid_comm_properties.sources.data(), MPI_UNWEIGHTED, send_coarse_gid_comm_properties.outdegree, send_coarse_gid_comm_properties.destinations.data(), MPI_UNWEIGHTED);
    }

    std::vector<int> coarse_gid_sizes_to_receive;

    { // Send coarse gid sizes 

        std::vector<int> coarse_gid_sizes_to_send;

        for( int i = 0 ; i < send_coarse_gid_comm_properties.outdegree ; i++ ) {
            coarse_gid_sizes_to_send.push_back(gid_to_send_to_rank.at(send_coarse_gid_comm_properties.destinations[i]).size());
        }
        coarse_gid_sizes_to_receive.resize(send_coarse_gid_comm_properties.indegree);

        // error = MPI_Neighbor_allgather(gid_sizes_to_send.data(), 1, MPI_INTEGER, gid_sizes_to_receive.data(), 1, MPI_INTEGER, send_coarse_gid_comm);

        std::vector<int> sendcounts(send_coarse_gid_comm_properties.outdegree, 1);

        std::vector<int> sdispls;
        sdispls.push_back(0);
        for( int i = 0 ; i < sendcounts.size() ; i++ ) {
            sdispls.push_back(sdispls.back() + sendcounts[i]);
        }

        std::vector<int> recvcounts(send_coarse_gid_comm_properties.indegree, 1);

        std::vector<int> rdispls;
        rdispls.push_back(0);
        for( int i = 0 ; i < recvcounts.size() ; i++ ) {
            rdispls.push_back(rdispls.back() + recvcounts[i]);
        }

        int error = MPI_Neighbor_alltoallv(coarse_gid_sizes_to_send.data(), sendcounts.data(), sdispls.data(), MPI_INTEGER, coarse_gid_sizes_to_receive.data(), recvcounts.data(), rdispls.data(), MPI_INTEGER, send_coarse_gid_comm);
        if(error != MPI_SUCCESS) {
            std::cerr << "MPI_Neighbor_alltoallv Send coarse gid sizes error = " << error << "\n";
            MPI_Barrier(MPI_COMM_WORLD);
            std::exit(0);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }


    std::vector<int> coarse_gid_list_received;
    RecvArrayProperties coarse_gid_list_received_props;

    { // Send coarse gid list
        std::vector<int> gid_to_send;
        std::vector<int> sendcounts;

        for( int i = 0 ; i < send_coarse_gid_comm_properties.outdegree ; i++ ) {
            sendcounts.push_back( gid_to_send_to_rank.at( send_coarse_gid_comm_properties.destinations[i] ).size() );
            for( auto v2:gid_to_send_to_rank.at( send_coarse_gid_comm_properties.destinations[i] ) ){
                gid_to_send.push_back(v2);
            }
        }

        // for( auto v:gid_to_get_from_rank ) {
        //     sendcounts.push_back( v.second.size() );
        //     for( auto v2:v.second ){
        //         gid_to_send.push_back(v2);
        //     }
        //     std::cout << "rank = " << rank << " dest = " << v.first << " number = " << v.second.size() << "\n";
        // }

        std::vector<int> sdispls;
        sdispls.push_back(0);
        for( int i = 0 ; i < sendcounts.size() ; i++ ) {
            sdispls.push_back(sdispls.back() + sendcounts[i]);
        }

        std::vector<int> rdispls;
        rdispls.push_back(0);
        int size_to_receive = 0;
        for( int i = 0 ; i < coarse_gid_sizes_to_receive.size() ; i++ ) {
            rdispls.push_back(rdispls.back() + coarse_gid_sizes_to_receive[i]);
            size_to_receive += coarse_gid_sizes_to_receive[i];
        }

        coarse_gid_list_received.resize(size_to_receive);
        

        int error = MPI_Neighbor_alltoallv(gid_to_send.data(), sendcounts.data(), sdispls.data(), MPI_INTEGER, coarse_gid_list_received.data(), coarse_gid_sizes_to_receive.data(), rdispls.data(), MPI_INTEGER, send_coarse_gid_comm);
        if(error != MPI_SUCCESS) {
            std::cerr << "MPI_Neighbor_alltoallv Send coarse gid list error = " << error << "\n";
            MPI_Barrier(MPI_COMM_WORLD);
            std::exit(0);
        }

        coarse_gid_list_received_props.rdispls = rdispls;
        coarse_gid_list_received_props.recvcounts = gid_sizes_to_receive;
        coarse_gid_list_received_props.sources = send_coarse_gid_comm_properties.sources;
    }

    for(auto v : coarse_gid_list_received) {
        if(v < A_lapl_N2 && v >= A_lapl_N1) {
            // std::cout << "rank = " << rank << " node received = " << v << "\n";
        } else {
            std::cerr << "coarse_gid received not local\n";
            std::exit(0);
        }
        gid_list.push_back(v);
    }
    std::sort(gid_list.begin(), gid_list.end());







    std::vector<int> coarse_local_index;
    std::vector<int> fine_local_index;

    std::vector<int> face_local_id_list;

    std::vector<char> all_index(A_full_N2 - A_full_N1, 0);

    int coarse_face_unknowns_local = 0;

    std::vector< std::tuple< double, double, double > > coarse_cell_center_coordinates;

    std::vector< std::tuple< double, double, double > > coarse_face_center_coordinates;
    
    std::vector<int> coarse_unknowns_local;

    coarse_unknowns_local.push_back(0);

    std::vector< std::tuple< int, int, int, int, int, int > > coarse_cell_to_face_local;

    for( int i = 0 ; i < gid_list.size() ; i++ ) {
        int gid = gid_list[i];
        if(gid >= A_lapl_N2 || gid < A_lapl_N1) {
            std::cerr << "rank = " << rank << " gid = " << gid << "\n";
            std::abort();
        }
        int back = std::get<0>(cell_to_face_local[gid - A_lapl_N1]);
        int front = std::get<1>(cell_to_face_local[gid - A_lapl_N1]);
        int left = std::get<2>(cell_to_face_local[gid - A_lapl_N1]);
        int right = std::get<3>(cell_to_face_local[gid - A_lapl_N1]);
        int bottom = std::get<4>(cell_to_face_local[gid - A_lapl_N1]);
        int top = std::get<5>(cell_to_face_local[gid - A_lapl_N1]);
        int back_face_id = -1;
        int front_face_id = -1;
        int left_face_id = -1;
        int right_face_id = -1;
        int bottom_face_id = -1;
        int top_face_id = -1;
        if( back != -1 ) {
            face_local_id_list.push_back( back );
            back_face_id = coarse_face_unknowns_local;
            coarse_face_unknowns_local++;
            coarse_face_center_coordinates.push_back( cell_center_coordinates[gid - A_lapl_N1] );
        }
        //  else if (front != -1) {
        //     face_local_id_list.push_back( front );
        //     front_face_id = coarse_face_unknowns_local;
        //     coarse_face_unknowns_local++;
        // }
        if( left != -1 ) {
            face_local_id_list.push_back( left );
            left_face_id = coarse_face_unknowns_local;
            coarse_face_unknowns_local++;
            coarse_face_center_coordinates.push_back( cell_center_coordinates[gid - A_lapl_N1] );
        } 
        // else if (right != -1) {
        //     face_local_id_list.push_back( right );
        //     right_face_id = coarse_face_unknowns_local;
        //     coarse_face_unknowns_local++;
        // }
        if( bottom != -1 ) {
            face_local_id_list.push_back( bottom );
            bottom_face_id = coarse_face_unknowns_local;
            coarse_face_unknowns_local++;
            coarse_face_center_coordinates.push_back( cell_center_coordinates[gid - A_lapl_N1] );
        } 
        // else if (top != -1) {
        //     face_local_id_list.push_back( top );
        //     top_face_id = coarse_face_unknowns_local;
        //     coarse_face_unknowns_local++;
        // }

        coarse_cell_to_face_local.push_back( std::make_tuple( back_face_id, front_face_id, left_face_id, right_face_id, bottom_face_id, top_face_id ) );

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

    for( int i = 0 ; i < gid_list.size() ; i++ ) {
        int gid = gid_list[i];
        
        if(gid >= A_lapl_N2 || gid < A_lapl_N1) {
            std::cerr << "rank = " << rank << " gid = " << gid << "\n";
            std::abort();
        }

        coarse_cell_center_coordinates.push_back( cell_center_coordinates[gid - A_lapl_N1] );
    }

    for( int i = 0 ; i < variables.size() ; i++ ) {
        if( variables[i].type == "all_cells" ) {
            // variables[i].properties["local_range"] = std::pair<int, int>( unknowns_local.back(), unknowns_local.back() + cell_unknowns_local );
            coarse_unknowns_local.push_back( coarse_unknowns_local.back() + gid_list.size() );
        } else if( variables[i].type == "all_faces" ) {
            // variables[i].properties["local_range"] = std::pair<int, int>( unknowns_local.back(), unknowns_local.back() + cell_unknowns_local );
            coarse_unknowns_local.push_back( coarse_unknowns_local.back() + coarse_face_unknowns_local );
        } else {
            std::cerr << "Wrong variable type" << std::endl;
            std::exit(1);
        }
    }

    std::vector< std::tuple< double, double, double > > coarse_all_center_coordinates;

    for( int i = 0 ; i < variables.size() ; i++ ) {
        if( variables[i].type == "all_cells" ) {
            for( int j = 0 ; j < gid_list.size() ; j++ ) {
                int gid = gid_list[j];
                
                if(gid >= A_lapl_N2 || gid < A_lapl_N1) {
                    std::cerr << "rank = " << rank << " gid = " << gid << "\n";
                    std::abort();
                }

                coarse_local_index.push_back( gid - A_lapl_N1 + unknowns_local[i] );
                all_index[ gid - A_lapl_N1 + unknowns_local[i] ] = 1;
                coarse_all_center_coordinates.push_back( coarse_cell_center_coordinates[j] );
            }
        } else if( variables[i].type == "all_faces" ) {
            for( int j = 0 ; j < face_local_id_list.size() ; j++ ) {
                int local_face_id = face_local_id_list[j];

                coarse_local_index.push_back( local_face_id + unknowns_local[i] );
                all_index[ local_face_id + unknowns_local[i] ] = 1;

                coarse_all_center_coordinates.push_back( coarse_face_center_coordinates[j] );
            }
        } else {
            std::cerr << "Wrong variable type" << std::endl;
            std::exit(1);
        }
    }


    

    std::sort(coarse_local_index.begin(), coarse_local_index.end());

    for( int i = 0 ; i < all_index.size() ; i++ ) {
        if(!all_index[i]) {
            fine_local_index.push_back( i );
        }
    }

    PetscCDDestroy(agg_lists);

    MPI_Comm_free(&send_size_comm);
    MPI_Comm_free(&send_coordinate_comm);
    MPI_Comm_free(&send_coarse_gid_comm);

    // std::cout << "mesh coarsen improved end\n";

    return std::make_tuple( coarse_local_index, fine_local_index, coarse_cell_center_coordinates, coarse_all_center_coordinates, coarse_cell_to_face_local, coarse_face_unknowns_local, gid_list.size(), coarse_unknowns_local );
}