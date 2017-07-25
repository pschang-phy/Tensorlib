#ifndef LINALG_H
#define LINALG_H

#include "Tensor.h"


namespace pschangLib
{
    namespace linalg
    {
        void svd( Tensor & iTensor, Tensor & uTensor, Tensor & dTensor, Tensor & vtTensor );
        double norm( const Tensor & iTensor );
        Tensor eye( int dim );
        void expm( const Tensor & iTensor, Tensor & outTensor, int order = 15 );
    }
}

#endif
