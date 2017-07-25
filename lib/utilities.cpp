#include <algorithm>

#include <cstdarg>

#include "utilities.h"

namespace pschangLib
{
    VecInt reshuffle( int iRank, const VecInt & axes, bool isBack )
    {
        if ( axes.size() > iRank )
            throw "axes number can not larger than rank number.\n";

        VecInt axesTmp( axes );
        std::sort( axesTmp.rbegin(), axesTmp.rend() );

        VecInt outAxes( iRank );
        int i, j;
        if ( isBack )
        {
            for ( i = 0, j = 0; i < iRank; ++i )
            {
                if ( axesTmp.size() > 0 && axesTmp.back() == i )
                {
                    axesTmp.pop_back();
                    continue;
                }
                outAxes[ j++ ] = i;
            }

            i = iRank - axes.size();
        }
        else
        {
            for ( i = 0, j = axes.size(); i < iRank; ++i )
            {
                if ( axesTmp.size() > 0 && axesTmp.back() == i )
                {
                    axesTmp.pop_back();
                    continue;
                }
                outAxes[ j++ ] = i;
            }

            i = 0;
        }

        for ( int j = 0; j < axes.size(); ++j )
            outAxes[ i++ ] = axes[ j ];

        return outAxes;
    }


    VecInt Shape( int dim, ... )
    {
        va_list shape_list;
        va_start( shape_list, dim );

        VecInt shape_out( dim );

        for ( int i = 0; i < dim; ++i )
            shape_out[ i ] = va_arg( shape_list, int );

        va_end( shape_list );

        return shape_out;
    }


    VecInt Range( int iBegin, int iEnd, int iStep )
    {
        if ( iBegin > iEnd && iStep > 0 )
            throw "invalid argument!\n";

        VecInt range_out( ( iEnd - iBegin ) / iStep );

        for ( int i = 0, val = iBegin; i < range_out.size(); ++i, val += iStep )
            range_out[ i ] = val;

        return range_out;
    }
}
