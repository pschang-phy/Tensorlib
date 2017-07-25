#ifndef TENSOR_OPERATORS_H
#define TENSOR_OPERATORS_H


#define CONST_Tensor_MEMBER_BINARY_OP(op, oper)           \
    Tensor Tensor::operator op( double iVal ) const       \
    {                                                     \
        Tensor outTensor( *this, Tensor::duplicateData ); \
                                                          \
        for ( size_t i = 0; i < size_; ++i )              \
            outTensor.data_[ i ] oper iVal;               \
                                                          \
        return outTensor;                                 \
    }


#define CONST_Tensor_FRIEND_BINARY_OP(op)                        \
    Tensor operator op( double iVal, const Tensor & iTensor )    \
    {                                                            \
        Tensor outTensor( iTensor, Tensor::duplicateData );      \
                                                                 \
        for ( size_t i = 0; i < iTensor.size_; ++i )             \
            outTensor.data_[ i ] = iVal op outTensor.data_[ i ]; \
                                                                 \
        return outTensor;                                        \
    }


#define Tensor_MEMBER_BINARY_OP(op)                                                                 \
    Tensor & Tensor::operator op( double iVal )                                                     \
    {                                                                                               \
        if ( owner_ )                                                                               \
        {                                                                                           \
            for ( size_t i = 0; i < size_; ++i )                                                    \
                data_[ i ] op iVal;                                                                 \
        }                                                                                           \
        else                                                                                        \
        {                                                                                           \
            const int tensorRank = boundInfo_.size();                                               \
            size_t * sentinels = new size_t [ tensorRank ];                                         \
                                                                                                    \
            size_t offset = 0;                                                                      \
            for ( int i = 0; i < tensorRank; ++i )                                                  \
            {                                                                                       \
                offset += boundInfo_[ i ].begin_;                                                   \
                sentinels[ i ] = ( boundInfo_[ i ].extent_ - 1 ) * boundInfo_[ i ].stride_;         \
            }                                                                                       \
                                                                                                    \
            size_t * indices = new size_t [ tensorRank ];                                           \
            for ( int i = 0; i < tensorRank; ++i )                                                  \
                indices[ i ] = 0;                                                                   \
                                                                                                    \
            size_t i;                                                                               \
            int j;                                                                                  \
            for ( i = 0; i < size_; ++i )                                                           \
            {                                                                                       \
                data_[ offset ] op iVal;                                                            \
                for ( j = tensorRank - 1; ++indices[ j ] == boundInfo_[ j ].extent_ && j > 0; --j ) \
                {                                                                                   \
                    indices[ j ] = 0;                                                               \
                    offset -= sentinels[ j ];                                                       \
                }                                                                                   \
                offset += boundInfo_[ j ].stride_;                                                  \
            }                                                                                       \
                                                                                                    \
            delete [] sentinels;                                                                    \
            delete [] indices;                                                                      \
        }                                                                                           \
                                                                                                    \
        return *this;                                                                               \
    }


#define CONST_Tensor_Tensor_MEMBER_OP(op, oper)                                                     \
    Tensor Tensor::operator op( const Tensor & iTensor ) const                                      \
    {                                                                                               \
        Tensor outTensor;                                                                           \
        if ( shape() != iTensor.shape() )                                                           \
            throw "Tensor shape are not compatiable\n";                                             \
                                                                                                    \
        outTensor.size_ = size_;                                                                    \
        outTensor.data_ = duplicate_();                                                             \
        outTensor.reshape( shape() );                                                               \
        if ( iTensor.owner_ )                                                                       \
        {                                                                                           \
            for ( size_t i = 0; i < size_; ++i )                                                    \
                outTensor.data_[ i ] oper iTensor.data_[ i ];                                       \
         }                                                                                          \
        else                                                                                        \
        {                                                                                           \
            const int tensorRank = boundInfo_.size();                                               \
            size_t * sentinels = new size_t [ tensorRank ];                                         \
                                                                                                    \
            size_t offset = 0;                                                                      \
            for ( int i = 0; i < tensorRank; ++i )                                                  \
            {                                                                                       \
                offset += iTensor.boundInfo_[ i ].begin_;                                           \
                sentinels[ i ] = ( boundInfo_[ i ].extent_ - 1 ) * iTensor.boundInfo_[ i ].stride_; \
            }                                                                                       \
                                                                                                    \
            size_t * indices = new size_t [ tensorRank ];                                           \
            for ( int i = 0; i < tensorRank; ++i )                                                  \
                indices[ i ] = 0;                                                                   \
                                                                                                    \
            size_t i;                                                                               \
            int j;                                                                                  \
            for ( i = 0; i < size_; ++i )                                                           \
            {                                                                                       \
                outTensor.data_[ offset ] oper iTensor.data_[ offset ];                             \
                for ( j = tensorRank - 1; ++indices[ j ] == boundInfo_[ j ].extent_ && j > 0; --j ) \
                {                                                                                   \
                    indices[ j ] = 0;                                                               \
                    offset -= sentinels[ j ];                                                       \
                }                                                                                   \
                offset += iTensor.boundInfo_[ j ].stride_;                                          \
            }                                                                                       \
                                                                                                    \
            delete [] sentinels;                                                                    \
            delete [] indices;                                                                      \
        }                                                                                           \
                                                                                                    \
        return outTensor;                                                                           \
    }


#define Tensor_Tensor_MEMBER_OP(op)                                                                  \
    Tensor & Tensor::operator op( const Tensor & iTensor )                                           \
    {                                                                                                \
        if ( shape() != iTensor.shape() )                                                            \
            throw "Tensor shape are not compatiable\n";                                              \
                                                                                                     \
        if ( owner_ && iTensor.owner_ )                                                              \
        {                                                                                            \
            for ( size_t i = 0; i < size_; ++i )                                                     \
                data_[ i ] op iTensor.data_[ i ];                                                    \
        }                                                                                            \
        else                                                                                         \
        {                                                                                            \
            const int tensorRank = boundInfo_.size();                                                \
            size_t * sentinels1 = new size_t [ tensorRank ];                                         \
            size_t * sentinels2 = new size_t [ tensorRank ];                                         \
                                                                                                     \
            size_t offset1 = 0, offset2 = 0;                                                         \
            for ( int i = 0; i < tensorRank; ++i )                                                   \
            {                                                                                        \
                offset1 += boundInfo_[ i ].begin_;                                                   \
                offset2 += iTensor.boundInfo_[ i ].begin_;                                           \
                sentinels1[ i ] = ( boundInfo_[ i ].extent_ - 1 ) * boundInfo_[ i ].stride_;         \
                sentinels2[ i ] = ( boundInfo_[ i ].extent_ - 1 ) * iTensor.boundInfo_[ i ].stride_; \
            }                                                                                        \
                                                                                                     \
            size_t * indices = new size_t [ tensorRank ];                                            \
            for ( int i = 0; i < tensorRank; ++i )                                                   \
                indices[ i ] = 0;                                                                    \
                                                                                                     \
            size_t i;                                                                                \
            int j;                                                                                   \
            for ( i = 0; i < size_; ++i )                                                            \
            {                                                                                        \
                data_[ offset1 ] op iTensor.data_[ offset2 ];                                        \
                for ( j = tensorRank - 1; ++indices[ j ] == boundInfo_[ j ].extent_ && j > 0; --j )  \
                {                                                                                    \
                    indices[ j ] = 0;                                                                \
                    offset1 -= sentinels1[ j ];                                                      \
                    offset2 -= sentinels2[ j ];                                                      \
                }                                                                                    \
                offset1 += boundInfo_[ j ].stride_;                                                  \
                offset2 += iTensor.boundInfo_[ j ].stride_;                                          \
            }                                                                                        \
                                                                                                     \
            delete [] sentinels1;                                                                    \
            delete [] sentinels2;                                                                    \
            delete [] indices;                                                                       \
        }                                                                                            \
                                                                                                     \
        return *this;                                                                                \
    }

#endif
