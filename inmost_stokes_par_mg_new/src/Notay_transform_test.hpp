#pragma once

#include"petsc.h"
#include<map>
#include"submatrix_manager.hpp"

void notay_transform_test(Mat A_transformed, 
                          Vec b_wo_test, 
                          Mat Apu_Auu_SPAI, 
                          Mat Auu_SPAI_Aup, 
                          IS Au_IS,
                          IS Ap_IS,
                          Mat Awo, 
                          Vec b_wo_);