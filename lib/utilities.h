#ifndef UTILITIES_H
#define UTILITIES_H

#include "Tensor.h"

namespace pschangLib
{
    VecInt reshuffle( int iRank, const VecInt & axes, bool isBack );
    VecInt Shape( int dim, ... );
    VecInt Range( int iBegin, int iEnd, int step = 1 );

}

#endif
