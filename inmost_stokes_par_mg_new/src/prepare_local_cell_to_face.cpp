
#include"prepare_local_cell_to_face.hpp"

std::vector< std::tuple< int, int, int, int, int, int > > prepare_local_cell_to_face( INMOST::Mesh *m )
{
    std::vector< std::tuple< int, int, int, int, int, int > > cell_to_face_local; // x0, x1, y0, y1, z0, z1
    INMOST::TagInteger face_id_local = INMOST::TagInteger( m->GetTag( "face_id_local" ) );
    for( INMOST::Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
    {
        if( cell->GetStatus() != INMOST::Element::Ghost ) {
            INMOST::ElementArray<INMOST::Face> faces = cell->getFaces();
            INMOST::Face back = faces[0].getAsFace();
            INMOST::Face front = faces[1].getAsFace();
            INMOST::Face left = faces[2].getAsFace();
            INMOST::Face right = faces[3].getAsFace();
            INMOST::Face bottom = faces[4].getAsFace();
            INMOST::Face top = faces[5].getAsFace();

            int back_id = -1;
            int front_id = -1;
            int left_id = -1;
            int right_id = -1;
            int bottom_id = -1;
            int top_id = -1;
            if( back->GetStatus() != INMOST::Element::Ghost ) {
                back_id = back.Integer(face_id_local);
            }
            if( front->GetStatus() != INMOST::Element::Ghost ) {
                front_id = front.Integer(face_id_local);
            }
            if( left->GetStatus() != INMOST::Element::Ghost ) {
                left_id = left.Integer(face_id_local);
            }
            if( right->GetStatus() != INMOST::Element::Ghost ) {
                right_id = right.Integer(face_id_local);
            }
            if( bottom->GetStatus() != INMOST::Element::Ghost ) {
                bottom_id = bottom.Integer(face_id_local);
            }
            if( top->GetStatus() != INMOST::Element::Ghost ) {
                top_id = top.Integer(face_id_local);
            }

            cell_to_face_local.push_back(std::make_tuple(back_id, 
                                                         front_id, 
                                                         left_id, 
                                                         right_id, 
                                                         bottom_id, 
                                                         top_id));
        }
    }
    return cell_to_face_local;
}