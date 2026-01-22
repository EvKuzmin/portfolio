
#include "mark_mesh.hpp"

void mark_inlet_outlet(INMOST::Mesh *m, std::array<double, 6> bounding_box, std::map< std::string, INMOST::MarkerType > &markers) 
{
    INMOST::MarkerType inlet = markers.at("inlet");
    INMOST::MarkerType outlet = markers.at("outlet");
    for( INMOST::Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
    {
        INMOST::Storage::real cnt[3];
        face->Barycenter(cnt);
        if( cnt[2] > bounding_box[4]-1e-7 && cnt[2] < bounding_box[4]+1e-7 ) {
            face->SetMarker(inlet);
        }
        if( cnt[2] > bounding_box[5]-1e-7 && cnt[2] < bounding_box[5]+1e-7 ) {
            face->SetMarker(outlet);
        }
    }
}

void mark_velocities(INMOST::Mesh *m, std::map< std::string, INMOST::MarkerType > &markers)
{
    INMOST::MarkerType u_mrk = markers.at("u_mrk");
    INMOST::MarkerType v_mrk = markers.at("v_mrk");
    INMOST::MarkerType w_mrk = markers.at("w_mrk");
    for( INMOST::Mesh::iteratorCell cell = m->BeginCell(); cell != m->EndCell(); ++cell )
    {
        INMOST::ElementArray<INMOST::Face> faces = cell->getFaces();
        INMOST::Face back = faces[0].getAsFace();
        INMOST::Face front = faces[1].getAsFace();
        INMOST::Face left = faces[2].getAsFace();
        INMOST::Face right = faces[3].getAsFace();
        INMOST::Face bottom = faces[4].getAsFace();
        INMOST::Face top = faces[5].getAsFace();

        back.SetMarker(u_mrk);
        front.SetMarker(u_mrk);
        left.SetMarker(v_mrk);
        right.SetMarker(v_mrk);
        bottom.SetMarker(w_mrk);
        top.SetMarker(w_mrk);
    }
}

void mark_submicro(INMOST::Mesh *m, std::map< std::string, INMOST::MarkerType > &markers)
{
    INMOST::MarkerType submicro_mrk = markers.at("submicro_mrk");
    INMOST::MarkerType pore_mrk = markers.at("pore_mrk");
    INMOST::MarkerType pore_submicro_mrk = markers.at("pore_submicro_mrk");
    for( INMOST::Mesh::iteratorFace face = m->BeginFace(); face != m->EndFace(); ++face )
    {
        INMOST::ElementArray<INMOST::Cell> cells = face->getCells();
        bool cell1_submicro = false;
        bool cell2_submicro = false;
        if(cells.size() > 1) {
            if(cells[0].isValid() && cells[1].isValid()) {
                if( cells[0].GetMarker(submicro_mrk) ) {
                    cell1_submicro = true;
                }
                if( cells[1].GetMarker(submicro_mrk) ) {
                    cell2_submicro = true;
                }

                if((cell1_submicro && cell2_submicro) == true) {
                    face->SetMarker(submicro_mrk);
                } else if( ((!cell1_submicro) && (!cell2_submicro)) == true ) {
                    face->SetMarker(pore_mrk);
                } else if( (cell1_submicro || cell2_submicro) == true ) {
                    face->SetMarker(pore_submicro_mrk);
                }
            }
        }
    }
}