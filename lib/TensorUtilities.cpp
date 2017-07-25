#include <algorithm>

#include <cstdlib>

#include <cmath>

#include <lapacke.h>

#include <cblas.h>

#include "Tensor.h"
#include "utilities.h"


namespace pschangLib
{
    void randTensor( Tensor & iTensor, size_t rSeed )
    {
        srand( rSeed );
        if ( iTensor.owner_ )
        {
            for ( size_t i = 0; i < iTensor.size_; ++i )
                iTensor.data_[ i ] = rand() / double( RAND_MAX );
        }
        else
        {
            const int tensorRank = iTensor.boundInfo_.size();
            size_t * sentinels = new size_t [ tensorRank ];

            size_t offset = 0;
            for ( int i = 0; i < tensorRank; ++i )
            {
                offset += iTensor.boundInfo_[ i ].begin_;
                sentinels[ i ] = ( iTensor.boundInfo_[ i ].extent_ - 1 ) * iTensor.boundInfo_[ i ].stride_;
            }

            size_t * indices = new size_t [ tensorRank ];
            for ( int i = 0; i < tensorRank; ++i )
                indices[ i ] = 0;

            size_t i;
            int j;
            for ( i = 0; i < iTensor.size_; ++i )
            {
                iTensor.data_[ offset ] = rand() / double( RAND_MAX );
                for ( j = tensorRank - 1; ++indices[ j ] == iTensor.boundInfo_[ j ].extent_ && j > 0; --j )
                {
                    indices[ j ] = 0;
                    offset -= sentinels[ j ];
                }
                offset += iTensor.boundInfo_[ j ].stride_;
            }

            delete [] sentinels;
            delete [] indices;
        }
    }


    Tensor tensordot( const Tensor & iTensor1, const VecInt & axes1, const Tensor & iTensor2, const VecInt & axes2 )
    {
        size_t dotSize = 1;
        {
            if ( axes1.size() != axes2.size() )
                throw "axes length not compatible!\n";

            int axesRank = axes1.size();
            for ( int i = 0; i < axesRank; ++i )
            {
                if ( iTensor1.boundInfo_[ axes1[ i ] ].extent_ != iTensor2.boundInfo_[ axes2[ i ] ].extent_ )
                    throw "axes not compatible!\n";
                dotSize *= iTensor1.boundInfo_[ axes1[ i ] ].extent_;
            }
        }

        const int rankA = iTensor1.boundInfo_.size();
        const int rankB = iTensor2.boundInfo_.size();

        VecInt axesA = reshuffle( rankA, axes1, true );
        VecInt axesB = reshuffle( rankB, axes2, false );

        double * dataPtr_A = iTensor1.duplicate_( axesA );
        double * dataPtr_B = iTensor2.duplicate_( axesB );

        size_t outRows = iTensor1.size_ / dotSize;
        size_t outCols = iTensor2.size_ / dotSize;

        double * dataPtr_out = new double [ outRows * outCols ];


        if ( outRows > 1 && outCols > 1 )
            cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, outRows, outCols, dotSize,
                    1.0, dataPtr_A, dotSize, dataPtr_B, outCols, 0.0, dataPtr_out, outCols );
        else if ( outRows > 1 )
            cblas_dgemv( CblasRowMajor, CblasNoTrans, outRows, dotSize, 1.0, dataPtr_A, dotSize, dataPtr_B, 1, 0.0, dataPtr_out, 1 );
        else
            *dataPtr_out = cblas_ddot( dotSize, dataPtr_A, 1, dataPtr_B, 1 );


        VecInt outShape( rankA + rankB - 2 * axes1.size() );
        if ( 0 == outShape.size() )
            outShape.assign( 1, 1 );
        else
        {
            for ( int i = 0; i < rankA - axes1.size(); ++i )
                outShape[ i ] = iTensor1.boundInfo_[ axesA[ i ] ].extent_;

            for ( int i = rankA - axes1.size(), j = axes2.size(); j < rankB; ++i, ++j )
                outShape[ i ] = iTensor2.boundInfo_[ axesB[ j ] ].extent_;
        }

        delete [] dataPtr_A;
        delete [] dataPtr_B;

        Tensor outTensor( dataPtr_out, outShape, Tensor::setDataAsMember );

        return outTensor;
    }


    Tensor kron( const Tensor & iTensor1, const Tensor & iTensor2 )
    {
        if ( iTensor1.rank() != iTensor2.rank() )
            throw "two tensor must have same rank!\n";

        const int outRank = iTensor1.rank();

        double * dataPtr_out = new double [ iTensor1.size_ * iTensor2.size_ ];
        double * dataPtr_A = iTensor1.owner_ ? iTensor1.data_ : iTensor1.duplicate_();
        double * dataPtr_B = iTensor2.owner_ ? iTensor2.data_ : iTensor2.duplicate_();

        size_t offset;
        size_t i, j;
        for ( i = 0; i < iTensor1.size_; ++i )
        {
            offset = i * iTensor2.size_;
            for ( j = 0; j < iTensor2.size_; ++j, ++offset )
                dataPtr_out[ offset ] = dataPtr_A[ i ] * dataPtr_B[ j ];
        }

        VecInt outShape( iTensor1.shape() );
        for ( i = 0; i < outRank; ++i )
            outShape.push_back( iTensor2.boundInfo_[ i ].extent_ );

        Tensor outTensor( dataPtr_out, outShape, Tensor::setDataAsMember );

        outShape.clear();
        VecInt outAxes( 2 * outRank );
        for ( i = 0; i < outRank; ++i )
        {
            outAxes[ 2 * i ] = i;
            outAxes[ 2 * i + 1 ] = i + outRank;
            outShape.push_back( iTensor1.boundInfo_[ i ].extent_ * iTensor2.boundInfo_[ i ].extent_ );
        }

        if ( !iTensor1.owner_ )
            delete [] dataPtr_A;
        if ( !iTensor2.owner_ )
            delete [] dataPtr_B;

        outTensor.transpose( outAxes );
        outTensor.reshape( outShape );

        return outTensor;
    }


    Tensor trace( const Tensor & iTensor, int axis1, int axis2 )
    {
        const int tensorRank = iTensor.boundInfo_.size();
        if ( tensorRank < 2 )
            throw "The tensor to trace must have rank larger than or equal to 2!\n";
        else if ( axis1 >= tensorRank || axis1 < 0 || axis2 >= tensorRank || axis2 < 0 )
            throw "invalid axes to trace the tensor!\n";
        else if ( iTensor.boundInfo_[ axis1 ].extent_ != iTensor.boundInfo_[ axis2 ].extent_ )
            throw "The two axes to trace must be the same extent\n";


        size_t traceExtent = iTensor.boundInfo_[ axis1 ].extent_;
        size_t traceStride = iTensor.boundInfo_[ axis1 ].stride_ + iTensor.boundInfo_[ axis2 ].stride_;

        int outRank = tensorRank - 2;
        size_t outSize = iTensor.size_ / iTensor.boundInfo_[ axis1 ].extent_ / iTensor.boundInfo_[ axis2 ].extent_;
        double * dataPtr_out = new double [ outSize ];

        VecIndex outBounds;
        for ( int i = 0; i < tensorRank; ++i )
        {
            if ( axis1 == i || axis2 == i )
                continue;
            outBounds.push_back( iTensor.boundInfo_[ i ] );
        }

        VecInt outShape( outRank );

        if ( outRank > 0 )
        {
            size_t * sentinels = new size_t [ outRank ];
            size_t * indices = new size_t [ outRank ];
            for ( int i = 0; i < outRank; ++i )
                indices[ i ] = 0;

            size_t offset = iTensor.boundInfo_[ axis1 ].begin_ + iTensor.boundInfo_[ axis2 ].begin_;
            for ( int i = 0; i < outRank; ++i )
            {
                offset += outBounds[ i ].begin_;
                sentinels[ i ] = ( outBounds[ i ].extent_ - 1 ) * outBounds[ i ].stride_;
            }

            size_t traceOffset;
            size_t i;
            int j;
            for ( i = 0; i < outSize; ++i )
            {
                traceOffset = offset;
                dataPtr_out[ i ] = 0.0;
                for ( j = 0; j < traceExtent; ++j )
                {
                    dataPtr_out[ i ] += iTensor.data_[ traceOffset ];
                    traceOffset += traceStride;
                }

                for ( j = outRank - 1; ++indices[ j ] == outBounds[ j ].extent_ && j > 0; --j )
                {
                    indices[ j ] = 0;
                    offset -= sentinels[ j ];
                }
                offset += outBounds[ j ].stride_;
            }

            for ( i = 0; i < outRank; ++i )
                outShape[ i ] = outBounds[ i ].extent_;

            delete [] sentinels;
            delete [] indices;
        }
        else
        {
            size_t offset = iTensor.boundInfo_[ axis1 ].begin_ + iTensor.boundInfo_[ axis2 ].begin_;

            *dataPtr_out = 0.0;
            for ( size_t i = 0; i < traceExtent; ++i )
            {
                *dataPtr_out += iTensor.data_[ offset ];
                offset += traceStride;
            }

            outShape.push_back( outSize );
        }

        Tensor outTensor( dataPtr_out, outShape, Tensor::setDataAsMember );

        return outTensor;
    }
}
