#ifndef Tensor_H
#define Tensor_H

#include <iostream>

#include <vector>

#include "Index.h"

namespace pschangLib
{
    typedef std::vector<int> VecInt;
    typedef std::vector<Index> VecIndex;

    class Tensor
    {
        public:
            enum preExistingMemoryPolicy { duplicateData, setDataAsMember, referenceData };

            // class constructors
            explicit Tensor( int iSize = 0 );
            Tensor( int iSize, double iVal );
            Tensor( double * iData, const VecInt & iShape, preExistingMemoryPolicy = duplicateData );

            Tensor( const Tensor & iTensor, preExistingMemoryPolicy = duplicateData );

            Tensor & swapaxes( int axis1, int axis2 );
            Tensor & transpose( const VecInt & iShape );
            Tensor & reshape( const VecInt & iShape );
            Tensor & dot( const VecInt & axes1, const Tensor & iTensor, const VecInt & axes2 );
            Tensor & prod( const VecInt & axes, const Tensor & iTensor );
            Tensor & trace( int axis1 = 0, int axes2 = 1 );
            Tensor & resize( const VecInt & iShape );

            // class operators
            const double & operator[] ( const VecInt & indices ) const;
            double & operator[] ( const VecInt & indices );

            const Tensor operator[] ( int index ) const;
            Tensor operator[] ( int index );

            const Tensor operator[] ( const Index & index ) const;
            Tensor operator[] ( const Index & index );

            const Tensor operator[] ( const VecIndex & indices ) const;
            Tensor operator[] ( const VecIndex & indices );


            Tensor operator+ ( double iVal ) const;
            Tensor operator- ( double iVal ) const;
            Tensor operator* ( double iVal ) const;
            Tensor operator/ ( double iVal ) const;

            Tensor & operator= ( double iVal );
            Tensor & operator+= ( double iVal );
            Tensor & operator-= ( double iVal );
            Tensor & operator*= ( double iVal );
            Tensor & operator/= ( double iVal );

            Tensor operator+ ( const Tensor & iTensor ) const;
            Tensor operator- ( const Tensor & iTensor ) const;
            Tensor operator* ( const Tensor & iTensor ) const;
            Tensor operator/ ( const Tensor & iTensor ) const;

            Tensor & operator+= ( const Tensor & iTensor );
            Tensor & operator-= ( const Tensor & iTensor );
            Tensor & operator*= ( const Tensor & iTensor );
            Tensor & operator/= ( const Tensor & iTensor );

            Tensor & operator= ( const Tensor & iTensor );

            Tensor operator- () const;


            // inline member functions
            size_t rank() const
            { return boundInfo_.size(); }

            size_t extent( int iDim ) const
            {
                if ( iDim < 0 || iDim >= boundInfo_.size() )
                    throw "invalid index!\n";

                return boundInfo_[ iDim ].extent_;
            }

            size_t size() const
            { return size_; }

            bool isOwner() const
            { return owner_; }

            VecIndex bound() const
            { return boundInfo_; }

            double * copyData() const
            { return duplicate_(); }

            const double * data() const
            { return data_; }

            double * data()
            { return data_; }

            VecInt shape() const
            {
                VecInt outShape( boundInfo_.size() );
                for ( int i = 0; i < boundInfo_.size(); ++i )
                    outShape[ i ] = boundInfo_[ i ].extent_;

                return outShape;
            }

            Tensor & flatten()
            {
                reshape( VecInt( 1, size_ ) );
                return *this;
            }

            operator double() const
            {
                if ( 1 != size_ )
                    throw "Only tensor with only one element can be casted to double!\n";

                size_t offset = 0;
                for ( int i = 0; i < boundInfo_.size(); ++i )
                    offset += boundInfo_[ i ].begin_;

                return data_[ offset ];
            }

            // class destructor
            ~Tensor()
            {
                if ( owner_ )
                    delete [] data_;
            }

        private:
            VecIndex boundInfo_;
            double * data_;
            size_t size_;
            bool owner_;
            double * duplicate_( const VecInt & axes ) const;
            double * duplicate_() const;

            // class friends
            friend std::ostream & operator<< ( std::ostream & os, const Tensor & iTensor );
            friend Tensor operator+ ( double iVal, const Tensor & iTensor );
            friend Tensor operator- ( double iVal, const Tensor & iTensor );
            friend Tensor operator* ( double iVal, const Tensor & iTensor );
            friend Tensor operator/ ( double iVal, const Tensor & iTensor );

            friend void randTensor( Tensor & iTensor, size_t rSeed = 0 );
            friend Tensor tensordot( const Tensor & iTensor1, const VecInt & axes1, const Tensor & iTensor2, const VecInt & axes2 );
            friend Tensor kron( const Tensor & iTensor1, const Tensor & iTensor2 );
            friend Tensor trace( const Tensor & iTensor, int axis1 = 0, int axis2 = 1 );
    };


    namespace axes
    {
        const VecInt axis0( 1, 0 );
        const VecInt axis1( 1, 1 );
        const VecInt axis2( 1, 2 );
        const VecInt axis3( 1, 3 );
        const VecInt axis4( 1, 4 );
        const VecInt axis5( 1, 5 );
    }
}

#endif
