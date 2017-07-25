#ifndef BRACKETOPS_H
#define BRACKETOPS_H

#include "Tensor.h"

namespace pschangLib
{
    const double & Tensor::operator[] ( const VecInt & indices ) const
    {
        const int tensorRank = boundInfo_.size();
        if ( indices.size() != tensorRank )
            throw "The input rank of indices must match the rank of tensor!\n";

        size_t offset = 0;
        for ( int i = 0; i < tensorRank; ++i )
            offset += indices[ i ] * boundInfo_[ i ].stride_ + boundInfo_[ i ].begin_;

        return data_[ offset ];
    }


    double & Tensor::operator[] ( const VecInt & indices )
    {
        const int tensorRank = boundInfo_.size();
        if ( indices.size() != tensorRank )
            throw "The input rank of indices must match the rank of tensor!\n";

        size_t offset = 0;
        for ( int i = 0; i < tensorRank; ++i )
            offset += indices[ i ] * boundInfo_[ i ].stride_ + boundInfo_[ i ].begin_;

        return data_[ offset ];
    }


    const Tensor Tensor::operator[] ( const VecIndex & indices ) const
    {
        Tensor outTensor;
        outTensor.boundInfo_.clear();

        outTensor.owner_ = false;
        outTensor.data_ = data_;
        outTensor.size_ = 1;

        int i = 0;
        size_t offset = 0;
        for ( VecIndex::const_iterator it = indices.begin(); it < indices.end(); ++it, ++i )
        {
            offset += boundInfo_[ i ].begin_ + it->begin_ * boundInfo_[ i ].stride_;
            if ( it->extent_ > boundInfo_[ 0 ].extent_ )
                throw "excess the number of elements!\n";
            else if ( 0 == it->extent_ )
                throw "No element be chosen!\n";
            else if ( it->extent_ > 1 )
            {
                outTensor.boundInfo_.push_back( Index( offset,
                            offset + it->extent_ * it->stride_ * boundInfo_[ i ].stride_,
                            it->stride_ * boundInfo_[ i ].stride_ ) );
                offset = 0;
            }

            outTensor.size_ *= it->extent_;
        }

        if ( 0 == outTensor.boundInfo_.size() )
            outTensor.boundInfo_.push_back( Index( offset ) );

        return outTensor;
    }


    Tensor Tensor::operator[] ( const VecIndex & indices )
    {
        Tensor outTensor;
        outTensor.boundInfo_.clear();

        outTensor.owner_ = false;
        outTensor.data_ = data_;
        outTensor.size_ = 1;

        int i = 0;
        size_t offset = 0;
        for ( VecIndex::const_iterator it = indices.begin(); it < indices.end(); ++it, ++i )
        {
            offset += boundInfo_[ i ].begin_ + it->begin_ * boundInfo_[ i ].stride_;
            if ( it->extent_ > boundInfo_[ 0 ].extent_ )
                throw "excess the number of elements!\n";
            else if ( 0 == it->extent_ )
                throw "No element be chosen!\n";
            else if ( it->extent_ > 1 )
            {
                outTensor.boundInfo_.push_back( Index( offset,
                            offset + it->extent_ * it->stride_ * boundInfo_[ i ].stride_,
                            it->stride_ * boundInfo_[ i ].stride_ ) );
                offset = 0;
            }

            outTensor.size_ *= it->extent_;
        }

        if ( 0 == outTensor.boundInfo_.size() )
            outTensor.boundInfo_.push_back( Index( offset ) );

        return outTensor;
    }


    const Tensor Tensor::operator[] ( int index ) const
    {
        if ( index >= boundInfo_[ 0 ].extent_ || index < 0 )
            throw "invalid index!\n";

        Tensor outTensor;
        outTensor.owner_ = false;
        outTensor.size_ = size_ / boundInfo_[ 0 ].extent_;
        outTensor.data_ = data_;

        if ( boundInfo_.size() > 1 )
        {
            outTensor.boundInfo_.assign( boundInfo_.begin() + 1, boundInfo_.end() );
            outTensor.boundInfo_[ 0 ].begin_ += index * boundInfo_[ 0 ].stride_;
        }
        else
        {
            outTensor.boundInfo_[ 0 ].begin_ = index * boundInfo_[ 0 ].stride_;
            outTensor.boundInfo_[ 0 ].extent_ = 1;
        }

        return outTensor;
    }


    Tensor Tensor::operator[] ( int index )
    {
        if ( index >= boundInfo_[ 0 ].extent_ || index < 0 )
            throw "invalid index!\n";

        Tensor outTensor;
        outTensor.owner_ = false;
        outTensor.size_ = size_ / boundInfo_[ 0 ].extent_;
        outTensor.data_ = data_;

        if ( boundInfo_.size() > 1 )
        {
            outTensor.boundInfo_.assign( boundInfo_.begin() + 1, boundInfo_.end() );
            outTensor.boundInfo_[ 0 ].begin_ += index * boundInfo_[ 0 ].stride_;
        }
        else
        {
            outTensor.boundInfo_[ 0 ].begin_ = index * boundInfo_[ 0 ].stride_;
            outTensor.boundInfo_[ 0 ].extent_ = 1;
        }

        return outTensor;
    }


    const Tensor Tensor::operator[] ( const Index & index ) const
    {
        if ( 0 == index.extent_ )
            throw "No element be chosen!\n";
        else if ( index.extent_ > boundInfo_[ 0 ].extent_ )
            throw "excess the number of elements!\n";

        Tensor outTensor;
        outTensor.owner_ = false;
        outTensor.size_ = size_ / boundInfo_[ 0 ].extent_ * index.extent_;
        outTensor.data_ = data_;

        if ( index.extent_ > 1 )
        {
            outTensor.boundInfo_.assign( boundInfo_.begin(), boundInfo_.end() );
            outTensor.boundInfo_[ 0 ].begin_ += index.begin_ * boundInfo_[ 0 ].stride_;
            outTensor.boundInfo_[ 0 ].extent_ = index.extent_;
            outTensor.boundInfo_[ 0 ].stride_ = index.stride_ * boundInfo_[ 0 ].stride_;
        }
        else if ( boundInfo_.size() > 1 )
        {
            outTensor.boundInfo_.assign( boundInfo_.begin() + 1, boundInfo_.end() );
            outTensor.boundInfo_[ 0 ].begin_ += index.begin_ * boundInfo_[ 0 ].stride_;
        }
        else
        {
            outTensor.boundInfo_[ 0 ].begin_ = index.begin_ * boundInfo_[ 0 ].stride_;
            outTensor.boundInfo_[ 0 ].extent_ = 1;
        }

        return outTensor;
    }


    Tensor Tensor::operator[] ( const Index & index )
    {
        if ( 0 == index.extent_ )
            throw "No element be chosen!\n";
        else if ( index.extent_ > boundInfo_[ 0 ].extent_ )
            throw "excess the number of elements!\n";

        Tensor outTensor;
        outTensor.owner_ = false;
        outTensor.size_ = size_ / boundInfo_[ 0 ].extent_ * index.extent_;
        outTensor.data_ = data_;

        if ( index.extent_ > 1 )
        {
            outTensor.boundInfo_.assign( boundInfo_.begin(), boundInfo_.end() );
            outTensor.boundInfo_[ 0 ].begin_ += index.begin_ * boundInfo_[ 0 ].stride_;
            outTensor.boundInfo_[ 0 ].extent_ = index.extent_;
            outTensor.boundInfo_[ 0 ].stride_ = index.stride_ * boundInfo_[ 0 ].stride_;
        }
        else if ( boundInfo_.size() > 1 )
        {
            outTensor.boundInfo_.assign( boundInfo_.begin() + 1, boundInfo_.end() );
            outTensor.boundInfo_[ 0 ].begin_ += index.begin_ * boundInfo_[ 0 ].stride_;
        }
        else
        {
            outTensor.boundInfo_[ 0 ].begin_ = index.begin_ * boundInfo_[ 0 ].stride_;
            outTensor.boundInfo_[ 0 ].extent_ = 1;
        }

        return outTensor;
    }
}

#endif
