#include <algorithm>

#include <cmath>

#include <lapacke.h>

#include <cblas.h>

#include "linalg.h"

namespace pschangLib
{
    namespace linalg
    {
        void svd( Tensor & iTensor, Tensor & uTensor, Tensor & dTensor, Tensor & vtTensor )
        {
            if ( iTensor.rank() != 2 )
                throw "The Tensor must be rank 2!\n";

            double * dataPtr;
            if ( iTensor.isOwner() )
                dataPtr = iTensor.data();
            else
                dataPtr = iTensor.copyData();

            const int rows = iTensor.extent( 0 );
            const int cols = iTensor.extent( 1 );

            uTensor.resize( VecInt( 2, rows ) );

            dTensor.resize( VecInt( 1, std::min( rows, cols ) ) );

            vtTensor.resize( VecInt( 2, cols ) );

            double * superb = new double [ std::min( rows, cols ) - 1 ];
            LAPACKE_dgesvd( LAPACK_ROW_MAJOR, 'A', 'A', rows, cols, dataPtr, cols,
                    dTensor.data(), uTensor.data(), rows, vtTensor.data(), cols, superb );

            delete [] superb;
            if ( !iTensor.isOwner() )
                delete [] dataPtr;
        }


        double norm( const Tensor & iTensor )
        {
            const double * dataPtr = iTensor.data();
            const size_t tensorSize = iTensor.size();

            double outValue = 0.0;
            if ( iTensor.isOwner() )
            {
                for ( size_t i = 0; i < tensorSize; ++i )
                    outValue += dataPtr[ i ] * dataPtr[ i ];
            }
            else
            {
                const int tensorRank = iTensor.rank();
                const VecIndex tensorBounds = iTensor.bound();

                size_t * sentinels = new size_t [ tensorRank ];

                size_t offset = 0;
                for ( int i = 0; i < tensorRank; ++i )
                {
                    offset += tensorBounds[ i ].begin_;
                    sentinels[ i ] = ( tensorBounds[ i ].extent_ - 1 ) * tensorBounds[ i ].stride_;
                }

                size_t * indices = new size_t [ tensorRank ];
                for ( int i = 0; i < tensorRank; ++i )
                    indices[ i ] = 0;

                size_t i, j;
                for ( i = 0; i < tensorSize; ++i )
                {
                    outValue += dataPtr[ offset ] * dataPtr[ offset ];
                    for ( j = tensorRank - 1; ++indices[ j ] == tensorBounds[ j ].extent_ && j > 0; --j )
                    {
                        indices[ j ] = 0;
                        offset -= sentinels[ j ];
                    }
                    offset += tensorBounds[ j ].stride_;
                }

                delete [] sentinels;
                delete [] indices;
            }

            return sqrt( outValue );
        }


        Tensor eye( int dim )
        {
            if ( dim < 1 )
                throw "Identity matrix must larger than 0!\n";

            size_t outSize = dim * dim;
            double * dataPtr_out = new double [ outSize ];

            for ( size_t i = 0; i < outSize; ++i )
                dataPtr_out[ i ] = 0.0;

            for ( size_t i = 0; i < dim; ++i )
                dataPtr_out[ i + i * dim ] = 1.0;

            Tensor outTensor( dataPtr_out, VecInt( 2, dim ) );

            return outTensor;
        }


        void expm( const Tensor & iTensor, Tensor & outTensor, int order )
        {
            if ( 2 != iTensor.rank() )
                throw "Tensor rank must be 2!\n";
            else if ( iTensor.extent( 0 ) != iTensor.extent( 1 ) )
                throw "Tensor must be square matrix!\n";
            else if ( order < 0 )
                throw "The order can not less than zero!\n";

            const size_t dim = iTensor.extent( 0 );
            Tensor tmpTensor( eye( dim ) );

            outTensor.resize( VecInt( 2, dim ) );
            outTensor = tmpTensor;

            VecInt axis1( 1, 1 ), axis2( 1, 0 );
            for ( int i = 1; i <= order; ++i )
            {
                tmpTensor /= double( i );
                outTensor += tmpTensor.dot( axis1, iTensor, axis2 );
            }
        }
    }
}
