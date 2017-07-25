#include <iostream>

#include <vector>

#include <cblas.h>

#include "utilities.h"
#include "Tensor.h"


namespace pschangLib
{
    Tensor::Tensor( int iSize )
        : owner_( true ), data_( NULL )
    {
        if ( iSize < 0 )
            throw "The size of Tensor can not less than zero.\n";

        size_ = iSize;
        boundInfo_.push_back( Index( 0, size_ ) );
        if ( size_ > 0 )
        {
            data_ = new double [ size_ ];
            for ( size_t i = 0; i < size_; ++i )
                data_[ i ] = 0.0;
        }
    }


    Tensor::Tensor( int iSize, double iVal )
        : owner_( true ), data_( NULL )
    {
        if ( iSize < 0 )
            throw "The size of Tensor can not less than zero.\n";

        size_ = iSize;
        boundInfo_.push_back( Index( 0, size_ ) );
        if ( size_ > 0 )
        {
            data_ = new double [ size_ ];
            for ( size_t i = 0; i < size_; ++i )
                data_[ i ] = iVal;
        }
    }


    Tensor::Tensor( double * iData, const VecInt & iShape, preExistingMemoryPolicy iPolicy )
        : owner_( true ), data_( NULL )
    {
        size_t iSize = 1;
        for ( int i = 0; i < iShape.size(); ++i )
            iSize *= iShape[ i ];

        if ( iSize < 0 )
            throw "The size of Tensor can not less than zero.\n";

        size_ = iSize;
        if ( size_ > 0 )
            reshape( iShape );
        else
            boundInfo_.push_back( Index( 0, size_ ) );

        switch ( iPolicy )
        {
            case duplicateData:
                if ( size_ > 0 )
                {
                    data_ = new double [ size_ ];
                    for ( int i = 0; i < size_; ++i )
                        data_[ i ] = iData[ i ];
                }

                break;
            case setDataAsMember:
                data_ = iData;

                break;
            case referenceData:
                owner_ = false;
                data_ = iData;

                break;
        }
    }


    Tensor::Tensor( const Tensor & iTensor, preExistingMemoryPolicy iPolicy )
        : owner_( true ), size_( iTensor.size_ )
    {
        switch ( iPolicy )
        {
            case duplicateData:
                if ( iTensor.owner_ )
                {
                    boundInfo_ = iTensor.boundInfo_;
                    data_ = new double [ size_ ];
                    for ( size_t i = 0; i < size_; ++i )
                        data_[ i ] = iTensor.data_[ i ];
                }
                else
                {
                    reshape( iTensor.shape() );
                    data_ = iTensor.duplicate_();
                }

                break;
            case referenceData:
                owner_ = false;
                boundInfo_ = iTensor.boundInfo_;
                data_ = iTensor.data_;

                break;
        }
    }


    Tensor & Tensor::reshape( const VecInt & iShape )
    {
        size_t iSize = 1;
        for ( int j = 0; j < iShape.size(); ++j )
            iSize *= iShape[ j ];

        if ( iSize != size_ )
            throw "Tensor size is not compatible with new shape\n";

        if ( !owner_ )
        {
            data_ = duplicate_();
            owner_ = true;
        }

        boundInfo_.clear();
        boundInfo_.resize( iShape.size() );

        size_t stride = size_;
        for ( int i = 0; i < boundInfo_.size(); ++i )
        {
            boundInfo_[ i ].extent_ = iShape[ i ];
            stride /= iShape[ i ];
            boundInfo_[ i ].stride_ = stride;
        }

        return *this;
    }


    Tensor & Tensor::swapaxes( int axis1, int axis2 )
    {
        const int tensorRank = boundInfo_.size();
        if ( axis1 >= tensorRank || axis1 < 0 || axis2 >= tensorRank || axis2 < 0 )
            throw "Tensor index error!\n";

        VecInt axes( tensorRank );
        for ( int i = 0; i < tensorRank; ++i )
            axes[ i ] = i;
        axes[ axis1 ] = axis2;
        axes[ axis2 ] = axis1;

        double * data_tmp = duplicate_( axes );
        delete [] data_;
        data_ = data_tmp;

        VecInt newShape( tensorRank );
        for ( int i = 0; i < tensorRank; ++i )
            newShape[ i ] = boundInfo_[ i ].extent_;
        newShape[ axis1 ] = boundInfo_[ axis2 ].extent_;
        newShape[ axis2 ] = boundInfo_[ axis1 ].extent_;

        reshape( newShape );

        return *this;
    }


    Tensor & Tensor::transpose( const VecInt & axes )
    {
        const int tensorRank = boundInfo_.size();
        if ( axes.size() != tensorRank )
            throw "The axes size must match the size of tensor rank.\n";

        VecInt newShape( tensorRank );
        for ( int i = 0; i < tensorRank; ++i )
            newShape[ i ] = boundInfo_[ axes[ i ] ].extent_;

        double * data_tmp = duplicate_( axes );
        if ( owner_ )
            delete [] data_;
        else
            owner_ = true;

        data_ = data_tmp;
        reshape( newShape );

        return *this;
    }


    Tensor & Tensor::resize( const VecInt & iShape )
    {
        size_t iSize = 1;
        for ( int j = 0; j < iShape.size(); ++j )
            iSize *= iShape[ j ];

        if ( iSize < 0 )
            throw "The size of Tensor can not less than zero.\n";

        if ( owner_ )
            delete [] data_;
        else
            owner_ = true;

        size_ = iSize;
        if ( size_ > 0 )
        {
            data_ = new double [ size_ ];
            reshape( iShape );
        }
        else
        {
            data_ = NULL;
            boundInfo_.clear();
            boundInfo_.push_back( Index( 0, size_ ) );
        }
        
        return *this;
    }


    Tensor & Tensor::dot( const VecInt & axes1, const Tensor & iTensor, const VecInt & axes2 )
    {
        size_t dotSize = 1;
        {
            if ( axes1.size() != axes2.size() )
                throw "axes length not compatible!\n";

            int axesRank = axes1.size();
            for ( int i = 0; i < axesRank; ++i )
            {
                if ( boundInfo_[ axes1[ i ] ].extent_ != iTensor.boundInfo_[ axes2[ i ] ].extent_ )
                    throw "axes not compatible!\n";
                dotSize *= boundInfo_[ axes1[ i ] ].extent_;
            }

        }

        const int rankA = boundInfo_.size();
        const int rankB = iTensor.boundInfo_.size();

        VecInt axesA = reshuffle( rankA, axes1, true );
        VecInt axesB = reshuffle( rankB, axes2, false );

        VecInt outShape( rankA + rankB - 2 * axes1.size() );
        if ( 0 == outShape.size() )
            outShape.assign( 1, 1 );
        else
        {
            for ( int i = 0; i < rankA - axes1.size(); ++i )
                outShape[ i ] = boundInfo_[ axesA[ i ] ].extent_;

            for ( int i = rankA - axes1.size(), j = axes2.size(); j < rankB; ++i, ++j )
                outShape[ i ] = iTensor.boundInfo_[ axesB[ j ] ].extent_;
        }

        transpose( axesA );
        double * dataPtr_B = iTensor.duplicate_( axesB );

        size_t outRows = size_ / dotSize;
        size_t outCols = iTensor.size_ / dotSize;
        double * data_tmp = new double [ outRows * outCols ];

        if ( outRows > 1 && outCols > 1 )
        {
            cblas_dgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, outRows, outCols, dotSize,
                    1.0, data_, dotSize, dataPtr_B, outCols, 0.0, data_tmp, outCols );
        }
        else if ( outRows > 1 )
        {
            cblas_dgemv( CblasRowMajor, CblasNoTrans, outRows, dotSize, 1.0, data_, dotSize, dataPtr_B, 1, 0.0, data_tmp, 1 );
        }
        else
        {
            *data_tmp = cblas_ddot( dotSize, data_, 1, dataPtr_B, 1 );
        }

        delete [] dataPtr_B;
        delete [] data_;
        data_ = data_tmp;
        size_ = outRows * outCols;
        reshape( outShape );

        return *this;
    }


    Tensor & Tensor::prod( const VecInt & axes, const Tensor & iTensor )
    {
        const int rankA = boundInfo_.size();
        const int rankB = iTensor.boundInfo_.size();
        {
            if ( axes.size() != rankB )
                throw "axes not compatible!\n";

            size_t dotSize = 1;
            for ( int i = 0; i < rankB; ++i )
            {
                if ( boundInfo_[ axes[ i ] ].extent_ != iTensor.boundInfo_[ i ].extent_ )
                    throw "axes not compatible!\n";
                dotSize *= boundInfo_[ axes[ i ] ].extent_;
            }
        }

        size_t * extents = new size_t [ rankA ];
        size_t * strides = new size_t [ rankA ];
        size_t * sentinels_A = new size_t [ rankA ];
        size_t * sentinels_B = new size_t [ rankB ];

        VecInt axes_tmp = reshuffle( rankA, axes, false );
        size_t offset_A = 0;
        for ( int i = 0; i < rankA; ++i )
        {
            offset_A += boundInfo_[ i ].begin_;
            extents[ i ] = boundInfo_[ axes_tmp[ i ] ].extent_;
            strides[ i ] = boundInfo_[ axes_tmp[ i ] ].stride_;
            sentinels_A[ i ] = ( extents[ i ] - 1 ) * strides[ i ];
        }

        size_t offset_B = 0;
        for ( int i = 0; i < rankB; ++i )
        {
            offset_B += iTensor.boundInfo_[ i ].begin_;
            sentinels_B[ i ] = ( iTensor.boundInfo_[ i ].extent_ - 1 ) * iTensor.boundInfo_[ i ].stride_;
        }

        size_t * indices = new size_t [ rankA ];
        for ( int i = 0; i < rankA; ++i )
            indices[ i ] = 0;

        size_t i;
        int j;
        for ( i = 0; i < size_; ++i )
        {
            data_[ offset_A ] *= iTensor.data_[ offset_B ];
            for ( j = rankA - 1; ++indices[ j ] == extents[ j ] && j > 0; --j )
            {
                indices[ j ] = 0;
                offset_A -= sentinels_A[ j ];
                if ( j < rankB )
                    offset_B -= sentinels_B[ j ];
            }
            offset_A += strides[ j ];
            if ( j < rankB )
                offset_B += iTensor.boundInfo_[ j ].stride_;
        }

        delete [] extents;
        delete [] strides;
        delete [] sentinels_A;
        delete [] sentinels_B;
        delete [] indices;

        return *this;
    }


    Tensor & Tensor::trace( int axis1, int axis2 )
    {
        const int tensorRank = boundInfo_.size();
        if ( tensorRank < 2 )
            throw "The tensor to trace must have rank larger than or equal to 2!\n";
        else if ( axis1 >= tensorRank || axis1 < 0 || axis2 >= tensorRank || axis2 < 0 )
            throw "invalid axes to trace the tensor!\n";
        else if ( boundInfo_[ axis1 ].extent_ != boundInfo_[ axis2 ].extent_ )
            throw "The two axes to trace must be the same extent\n";


        size_t traceExtent = boundInfo_[ axis1 ].extent_;
        size_t traceStride = boundInfo_[ axis1 ].stride_ + boundInfo_[ axis2 ].stride_;

        int rank_out = tensorRank - 2;
        size_t size_out = size_ / boundInfo_[ axis1 ].extent_ / boundInfo_[ axis2 ].extent_;
        double * data_out = new double [ size_out ];

        VecIndex bounds_out;
        for ( int i = 0; i < tensorRank; ++i )
        {
            if ( axis1 == i || axis2 == i )
                continue;
            bounds_out.push_back( boundInfo_[ i ] );
        }

        size_t * sentinels = NULL;
        size_t * indices = NULL;
        if ( rank_out > 0 )
            sentinels = new size_t [ rank_out ];

        indices = new size_t [ rank_out ];
        for ( int i = 0; i < rank_out; ++i )
            indices[ i ] = 0;

        size_t offset = boundInfo_[ axis1 ].begin_ + boundInfo_[ axis2 ].begin_;
        for ( int i = 0; i < rank_out; ++i )
        {
            offset += bounds_out[ i ].begin_;
            sentinels[ i ] = ( bounds_out[ i ].extent_ - 1 ) * bounds_out[ i ].stride_;
        }

        size_t traceOffset;
        size_t i;
        int j;
        for ( i = 0; i < size_out; ++i )
        {
            traceOffset = offset;
            data_out[ i ] = 0.0;
            for ( j = 0; j < traceExtent; ++j )
            {
                data_out[ i ] += data_[ traceOffset ];
                traceOffset += traceStride;
            }

            for ( j = rank_out - 1; j >= 0 && ++indices[ j ] == bounds_out[ j ].extent_; --j )
            {
                indices[ j ] = 0;
                offset -= sentinels[ j ];
            }
            if ( j >= 0 )
                offset += bounds_out[ j ].stride_;
        }

        size_ = size_out;
        if ( owner_ )
        {
            delete [] data_;
            data_ = data_out;
        }
        else
        {
            owner_ = true;
            data_ = data_out;
        }

        VecInt shape_out( rank_out );
        for ( i = 0; i < rank_out; ++i )
            shape_out[ i ] = bounds_out[ i ].extent_;
        if ( 0 == rank_out )
            shape_out.push_back( size_out );

        reshape( shape_out );

        delete [] sentinels;
        delete [] indices;

        return *this;
    }

    double * Tensor::duplicate_( const VecInt & axes ) const
    {
        const int tensorRank = boundInfo_.size();
        if ( axes.size() != tensorRank )
            throw "The axes size must match the size of tensor rank.\n";

        size_t * extents = new size_t [ tensorRank ];
        size_t * strides = new size_t [ tensorRank ];
        size_t * sentinels = new size_t [ tensorRank ];

        size_t offset = 0;
        for ( int i = 0; i < tensorRank; ++i )
        {
            offset += boundInfo_[ axes[ i ] ].begin_;
            extents[ i ] = boundInfo_[ axes[ i ] ].extent_;
            strides[ i ] = boundInfo_[ axes[ i ] ].stride_;
            sentinels[ i ] = ( extents[ i ] - 1 ) * strides[ i ];
        }

        size_t * indices = new size_t [ tensorRank ];
        for ( int i = 0; i < tensorRank; ++i )
            indices[ i ] = 0;

        double * data_tmp = new double [ size_ ];
        size_t i, j;
        for ( i = 0; i < size_; ++i )
        {
            data_tmp[ i ] = data_[ offset ];
            for ( j = tensorRank - 1; ++indices[ j ] == extents[ j ] && j > 0; --j )
            {
                indices[ j ] = 0;
                offset -= sentinels[ j ];
            }
            offset += strides[ j ];
        }

        delete [] extents;
        delete [] strides;
        delete [] sentinels;
        delete [] indices;

        return data_tmp;
    }


    double * Tensor::duplicate_() const
    {
        double * data_tmp = new double [ size_ ];

        if ( owner_ )
        {
            for ( size_t i = 0; i < size_; ++i )
                data_tmp[ i ] = data_[ i ];
        }
        else 
        {
            const int tensorRank = boundInfo_.size();
            size_t * sentinels = new size_t [ tensorRank ];

            size_t offset = 0;
            for ( int i = 0; i < tensorRank; ++i )
            {
                offset += boundInfo_[ i ].begin_;
                sentinels[ i ] = ( boundInfo_[ i ].extent_ - 1 ) * boundInfo_[ i ].stride_;
            }

            size_t * indices = new size_t [ tensorRank ];
            for ( int i = 0; i < tensorRank; ++i )
                indices[ i ] = 0;

            size_t i, j;
            for ( i = 0; i < size_; ++i )
            {
                data_tmp[ i ] = data_[ offset ];
                for ( j = tensorRank - 1; ++indices[ j ] == boundInfo_[ j ].extent_ && j > 0; --j )
                {
                    indices[ j ] = 0;
                    offset -= sentinels[ j ];
                }
                offset += boundInfo_[ j ].stride_;
            }

            delete [] sentinels;
            delete [] indices;
        }

        return data_tmp;
    }
}
