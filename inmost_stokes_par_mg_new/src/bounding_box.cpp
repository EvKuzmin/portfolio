
#include "bounding_box.hpp"

std::array<double, 6> prepare_bounding_box(INMOST::Mesh *m) {
    int rank = m->GetProcessorRank();
    int size = m->GetProcessorsNumber();

    std::array<double, 6> bounding_box{1e100,-1e100, 1e100,-1e100, 1e100,-1e100}; // x0, x1, y0, y1, z0, z1

    for( INMOST::Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
    {
        INMOST::Storage::real cnt[3];
        face->Barycenter(cnt);
        if( bounding_box[0] > cnt[0] ) {
            bounding_box[0] = cnt[0];
        }
        if( bounding_box[1] < cnt[0] ) {
            bounding_box[1] = cnt[0];
        }
        if( bounding_box[2] > cnt[1] ) {
            bounding_box[2] = cnt[1];
        }
        if( bounding_box[3] < cnt[1] ) {
            bounding_box[3] = cnt[1];
        }
        if( bounding_box[4] > cnt[2] ) {
            bounding_box[4] = cnt[2];
        }
        if( bounding_box[5] < cnt[2] ) {
            bounding_box[5] = cnt[2];
        }
    }

    std::vector<double> all_bounding_box(size*6);

    MPI_Allgather(
    bounding_box.data(),
    6,
    MPI_DOUBLE,
    all_bounding_box.data(),
    6,
    MPI_DOUBLE,
    INMOST_MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    for( int i = 0 ; i < size ; i++ )
    {
        if( bounding_box[0] > all_bounding_box[0 + i*6] ) {
            bounding_box[0] = all_bounding_box[0 + i*6];
        }
        if( bounding_box[1] < all_bounding_box[1 + i*6] ) {
            bounding_box[1] = all_bounding_box[1 + i*6];
        }
        if( bounding_box[2] > all_bounding_box[2 + i*6] ) {
            bounding_box[2] = all_bounding_box[2 + i*6];
        }
        if( bounding_box[3] < all_bounding_box[3 + i*6] ) {
            bounding_box[3] = all_bounding_box[3 + i*6];
        }
        if( bounding_box[4] > all_bounding_box[4 + i*6] ) {
            bounding_box[4] = all_bounding_box[4 + i*6];
        }
        if( bounding_box[5] < all_bounding_box[5 + i*6] ) {
            bounding_box[5] = all_bounding_box[5 + i*6];
        }
    }
    return bounding_box;
}