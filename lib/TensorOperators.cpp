#include <iostream>

#include "Tensor.h"
#include "arithmeticOps.h"


namespace pschangLib
{
    // some binary operator definitions, one of operand is intrinsic type
    CONST_Tensor_MEMBER_BINARY_OP(+, +=)
        CONST_Tensor_MEMBER_BINARY_OP(-, -=)
        CONST_Tensor_MEMBER_BINARY_OP(*, *=)
        CONST_Tensor_MEMBER_BINARY_OP(/, /=)

        CONST_Tensor_FRIEND_BINARY_OP(+)
        CONST_Tensor_FRIEND_BINARY_OP(-)
        CONST_Tensor_FRIEND_BINARY_OP(*)
        CONST_Tensor_FRIEND_BINARY_OP(/)

        Tensor_MEMBER_BINARY_OP(=)
        Tensor_MEMBER_BINARY_OP(+=)
        Tensor_MEMBER_BINARY_OP(-=)
        Tensor_MEMBER_BINARY_OP(*=)
        Tensor_MEMBER_BINARY_OP(/=)

        CONST_Tensor_Tensor_MEMBER_OP(+, +=)
        CONST_Tensor_Tensor_MEMBER_OP(-, -=)
        CONST_Tensor_Tensor_MEMBER_OP(*, *=)
        CONST_Tensor_Tensor_MEMBER_OP(/, /=)

        Tensor_Tensor_MEMBER_OP(+=)
        Tensor_Tensor_MEMBER_OP(-=)
        Tensor_Tensor_MEMBER_OP(*=)
        Tensor_Tensor_MEMBER_OP(/=)


        Tensor & Tensor::operator= ( const Tensor & iTensor )
        {
            if ( owner_ )
            {
                size_ = iTensor.size_;
                if ( iTensor.owner_ )
                {
                    delete [] data_;
                    boundInfo_ = iTensor.boundInfo_;

                    data_ = new double [ size_ ];
                    for ( size_t i = 0; i < size_; ++i )
                        data_[ i ] = iTensor.data_[ i ];
                }
                else
                {
                    reshape( iTensor.shape() );

                    double * data_tmp = iTensor.duplicate_();
                    delete [] data_;
                    data_ = data_tmp;
                }
            }
            else if ( shape() == iTensor.shape() )
            {
                const int tensorRank = boundInfo_.size();
                size_t * sentinels1 = new size_t [ tensorRank ];
                size_t * sentinels2 = new size_t [ tensorRank ];

                size_t offset1 = 0, offset2 = 0;
                for ( int i = 0; i < tensorRank; ++i )
                {
                    offset1 += boundInfo_[ i ].begin_;
                    offset2 += iTensor.boundInfo_[ i ].begin_;
                    sentinels1[ i ] = ( boundInfo_[ i ].extent_ - 1 ) * boundInfo_[ i ].stride_;
                    sentinels2[ i ] = ( boundInfo_[ i ].extent_ - 1 ) * iTensor.boundInfo_[ i ].stride_;
                }

                size_t * indices = new size_t [ tensorRank ];
                for ( int i = 0; i < tensorRank; ++i )
                    indices[ i ] = 0;

                size_t i, j;
                for ( i = 0; i < size_; ++i )
                {
                    data_[ offset1 ] = iTensor.data_[ offset2 ];
                    for ( j = tensorRank - 1; ++indices[ j ] == boundInfo_[ j ].extent_ && j > 0; --j )
                    {
                        indices[ j ] = 0;
                        offset1 -= sentinels1[ j ];
                        offset2 -= sentinels2[ j ];
                    }
                    offset1 += boundInfo_[ j ].stride_;
                    offset2 += iTensor.boundInfo_[ j ].stride_;
                }

                delete [] sentinels1;
                delete [] sentinels2;
                delete [] indices;
            }
            else
                throw "assigne shape incompatible!\n";

            return *this;
        }


    Tensor Tensor::operator- () const
    {
        Tensor outTensor( *this, Tensor::duplicateData );

        for ( size_t i = 0; i < outTensor.size_; ++i )
            outTensor.data_[ i ] = -outTensor.data_[ i ];

        return outTensor;
    }


    std::ostream & operator<< ( std::ostream & os, const Tensor & iTensor )
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
            os << iTensor.data_[ offset ];

            for ( j = tensorRank - 1; ++indices[ j ] == iTensor.boundInfo_[ j ].extent_ && j > 0; --j )
            {
                indices[ j ] = 0;
                offset -= sentinels[ j ];
            }
            offset += iTensor.boundInfo_[ j ].stride_;

            os << ( tensorRank - 1 == j ? ", " : "\n" );
        }
        os << std::endl;

        delete [] sentinels;
        delete [] indices;

        return os;
    }
}
