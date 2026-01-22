#include"make_adjacent.hpp"

void make_face_adj_cells( Mesh *m, 
                          MarkerType u_mrk, 
                          MarkerType v_mrk, 
                          MarkerType w_mrk, 
                          MarkerType inner_face_mrk ) {
    TagReference face_adj_cells = TagReference( m->GetTag( "face_adj_cells" ) );

    for( Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
    {
        if( face->GetStatus() != Element::Ghost ) {
            ElementArray<Cell> cells = face->getCells();
            if(cells.size() > 1) {
                if(cells[0].isValid() && cells[1].isValid()) {
                    face->SetMarker(inner_face_mrk);
                    if( face->GetMarker(u_mrk) ) {
                        Storage::real cnt1[3];
                        Storage::real cnt2[3];
                        cells[0].Barycenter(cnt1);
                        cells[1].Barycenter(cnt2);
                        if(cnt1[0] < cnt2[0]) {
                            face->ReferenceArray(face_adj_cells).push_back(cells[0]);
                            face->ReferenceArray(face_adj_cells).push_back(cells[1]);
                        } else {
                            face->ReferenceArray(face_adj_cells).push_back(cells[1]);
                            face->ReferenceArray(face_adj_cells).push_back(cells[0]);
                        }
                    }
                    if( face->GetMarker(v_mrk) ) {
                        Storage::real cnt1[3];
                        Storage::real cnt2[3];
                        cells[0].Barycenter(cnt1);
                        cells[1].Barycenter(cnt2);
                        if(cnt1[1] < cnt2[1]) {
                            face->ReferenceArray(face_adj_cells).push_back(cells[0]);
                            face->ReferenceArray(face_adj_cells).push_back(cells[1]);
                        } else {
                            face->ReferenceArray(face_adj_cells).push_back(cells[1]);
                            face->ReferenceArray(face_adj_cells).push_back(cells[0]);
                        }
                    }
                    if( face->GetMarker(w_mrk) ) {
                        Storage::real cnt1[3];
                        Storage::real cnt2[3];
                        cells[0].Barycenter(cnt1);
                        cells[1].Barycenter(cnt2);
                        if(cnt1[2] < cnt2[2]) {
                            face->ReferenceArray(face_adj_cells).push_back(cells[0]);
                            face->ReferenceArray(face_adj_cells).push_back(cells[1]);
                        } else {
                            face->ReferenceArray(face_adj_cells).push_back(cells[1]);
                            face->ReferenceArray(face_adj_cells).push_back(cells[0]);
                        }
                    }
                }
            }
        }
    }
}

void make_face_lapl_adj( Mesh *m, 
                         MarkerType u_mrk, 
                         MarkerType v_mrk, 
                         MarkerType w_mrk, 
                         MarkerType inner_face_mrk) {
    TagReference face_lapl_adj = TagReference( m->GetTag( "face_lapl_adj" ) );
    TagReference face_adj_cells = TagReference( m->GetTag( "face_adj_cells" ) );
    for( Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
    {
        if( face->GetStatus() != Element::Ghost ) {
            if( face->GetMarker(inner_face_mrk) ) {
                Cell cell1 = Cell(m, face->ReferenceArray(face_adj_cells)[0].GetHandle());
                Cell cell2 = Cell(m, face->ReferenceArray(face_adj_cells)[1].GetHandle());

                ElementArray<Face> faces1 = cell1.getFaces();
                ElementArray<Face> faces2 = cell2.getFaces();
                if( face->GetMarker(u_mrk) ) {
                    face->ReferenceArray(face_lapl_adj).push_back(cell1.getFaces()[0]);
                    face->ReferenceArray(face_lapl_adj).push_back(cell2.getFaces()[1]);

                    Cell left_cell = cell1.Neighbour(faces1[2].getAsFace());
                    Face u_left;
                    if(!left_cell.isValid()) {
                        left_cell = cell2.Neighbour(faces2[2].getAsFace());
                        if(left_cell.isValid()) {
                            u_left = left_cell.getFaces()[0].getAsFace();
                        }
                    } else {
                        u_left = left_cell.getFaces()[1].getAsFace();
                    }

                    face->ReferenceArray(face_lapl_adj).push_back(u_left);

                    Cell right_cell = cell1.Neighbour(faces1[3].getAsFace());
                    Face u_right;
                    if(!right_cell.isValid()) {
                        right_cell = cell2.Neighbour(faces2[3].getAsFace());
                        if(right_cell.isValid()) {
                            u_right = right_cell.getFaces()[0].getAsFace();
                        }
                    } else {
                        u_right = right_cell.getFaces()[1].getAsFace();
                    }

                    face->ReferenceArray(face_lapl_adj).push_back(u_right);

                    Cell bottom_cell = cell1.Neighbour(faces1[4].getAsFace());
                    Face u_bottom;
                    if(!bottom_cell.isValid()) {
                        bottom_cell = cell2.Neighbour(faces2[4].getAsFace());
                        if(bottom_cell.isValid()) {
                            u_bottom = bottom_cell.getFaces()[0].getAsFace();
                        }
                    } else {
                        u_bottom = bottom_cell.getFaces()[1].getAsFace();
                    }

                    face->ReferenceArray(face_lapl_adj).push_back(u_bottom);

                    Cell top_cell = cell1.Neighbour(faces1[5].getAsFace());
                    Face u_top;
                    if(!top_cell.isValid()) {
                        top_cell = cell2.Neighbour(faces2[5].getAsFace());
                        if(top_cell.isValid()) {
                            u_top = top_cell.getFaces()[0].getAsFace();
                        }
                    } else {
                        u_top = top_cell.getFaces()[1].getAsFace();
                    }

                    face->ReferenceArray(face_lapl_adj).push_back(u_top);
                }
                if( face->GetMarker(v_mrk) ) {
                    face->ReferenceArray(face_lapl_adj).push_back(cell1.getFaces()[2]);
                    face->ReferenceArray(face_lapl_adj).push_back(cell2.getFaces()[3]);

                    Cell left_cell = cell1.Neighbour(faces1[0].getAsFace());
                    Face u_left;
                    if(!left_cell.isValid()) {
                        left_cell = cell2.Neighbour(faces2[0].getAsFace());
                        if(left_cell.isValid()) {
                            u_left = left_cell.getFaces()[2].getAsFace();
                        }
                    } else {
                        u_left = left_cell.getFaces()[3].getAsFace();
                    }

                    face->ReferenceArray(face_lapl_adj).push_back(u_left);

                    Cell right_cell = cell1.Neighbour(faces1[1].getAsFace());
                    Face u_right;
                    if(!right_cell.isValid()) {
                        right_cell = cell2.Neighbour(faces2[1].getAsFace());
                        if(right_cell.isValid()) {
                            u_right = right_cell.getFaces()[2].getAsFace();
                        }
                    } else {
                        u_right = right_cell.getFaces()[3].getAsFace();
                    }

                    face->ReferenceArray(face_lapl_adj).push_back(u_right);

                    Cell bottom_cell = cell1.Neighbour(faces1[4].getAsFace());
                    Face u_bottom;
                    if(!bottom_cell.isValid()) {
                        bottom_cell = cell2.Neighbour(faces2[4].getAsFace());
                        if(bottom_cell.isValid()) {
                            u_bottom = bottom_cell.getFaces()[2].getAsFace();
                        }
                    } else {
                        u_bottom = bottom_cell.getFaces()[3].getAsFace();
                    }

                    face->ReferenceArray(face_lapl_adj).push_back(u_bottom);

                    Cell top_cell = cell1.Neighbour(faces1[5].getAsFace());
                    Face u_top;
                    if(!top_cell.isValid()) {
                        top_cell = cell2.Neighbour(faces2[5].getAsFace());
                        if(top_cell.isValid()) {
                            u_top = top_cell.getFaces()[2].getAsFace();
                        }
                    } else {
                        u_top = top_cell.getFaces()[3].getAsFace();
                    }

                    face->ReferenceArray(face_lapl_adj).push_back(u_top);
                }
                if( face->GetMarker(w_mrk) ) {
                    face->ReferenceArray(face_lapl_adj).push_back(cell1.getFaces()[4]);
                    face->ReferenceArray(face_lapl_adj).push_back(cell2.getFaces()[5]);

                    Cell left_cell = cell1.Neighbour(faces1[0].getAsFace());
                    Face u_left;
                    if(!left_cell.isValid()) {
                        left_cell = cell2.Neighbour(faces2[0].getAsFace());
                        if(left_cell.isValid()) {
                            u_left = left_cell.getFaces()[4].getAsFace();
                        }
                    } else {
                        u_left = left_cell.getFaces()[5].getAsFace();
                    }

                    face->ReferenceArray(face_lapl_adj).push_back(u_left);

                    Cell right_cell = cell1.Neighbour(faces1[1].getAsFace());
                    Face u_right;
                    if(!right_cell.isValid()) {
                        right_cell = cell2.Neighbour(faces2[1].getAsFace());
                        if(right_cell.isValid()) {
                            u_right = right_cell.getFaces()[4].getAsFace();
                        }
                    } else {
                        u_right = right_cell.getFaces()[5].getAsFace();
                    }

                    face->ReferenceArray(face_lapl_adj).push_back(u_right);

                    Cell bottom_cell = cell1.Neighbour(faces1[2].getAsFace());
                    Face u_bottom;
                    if(!bottom_cell.isValid()) {
                        bottom_cell = cell2.Neighbour(faces2[2].getAsFace());
                        if(bottom_cell.isValid()) {
                            u_bottom = bottom_cell.getFaces()[4].getAsFace();
                        }
                    } else {
                        u_bottom = bottom_cell.getFaces()[5].getAsFace();
                    }

                    face->ReferenceArray(face_lapl_adj).push_back(u_bottom);

                    Cell top_cell = cell1.Neighbour(faces1[3].getAsFace());
                    Face u_top;
                    if(!top_cell.isValid()) {
                        top_cell = cell2.Neighbour(faces2[3].getAsFace());
                        if(top_cell.isValid()) {
                            u_top = top_cell.getFaces()[4].getAsFace();
                        }
                    } else {
                        u_top = top_cell.getFaces()[5].getAsFace();
                    }

                    face->ReferenceArray(face_lapl_adj).push_back(u_top);
                }
            }
        }
    }
}

void make_cell_lapl_adj(Mesh *m )
{
    TagReference cell_lapl_adj = TagReference( m->GetTag( "cell_lapl_adj" ) );
    for( Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
    {
        if( cell->GetStatus() != Element::Ghost ) {
            ElementArray<Face> faces = cell->getFaces();
            Face back = faces[0].getAsFace();
            Face front = faces[1].getAsFace();
            Face left = faces[2].getAsFace();
            Face right = faces[3].getAsFace();
            Face bottom = faces[4].getAsFace();
            Face top = faces[5].getAsFace();

            cell->ReferenceArray(cell_lapl_adj).push_back(cell->Neighbour(back));
            cell->ReferenceArray(cell_lapl_adj).push_back(cell->Neighbour(front));
            cell->ReferenceArray(cell_lapl_adj).push_back(cell->Neighbour(left));
            cell->ReferenceArray(cell_lapl_adj).push_back(cell->Neighbour(right));
            cell->ReferenceArray(cell_lapl_adj).push_back(cell->Neighbour(bottom));
            cell->ReferenceArray(cell_lapl_adj).push_back(cell->Neighbour(top));
        }
    }
}