#include <iostream>
using std::cout;
using std::cin;
using std::endl;

#include "Tensor.h"
#include "linalg.h"
#include "utilities.h"

using namespace pschangLib;

double sx[ 2 * 2 ] = { 0.0, 1.0, 1.0, 0.0 };
double sz[ 2 * 2 ] = { 1.0, 0.0, 0.0, -1.0 };

const Tensor Sx( sx, VecInt( 2, 2 ) );
const Tensor Sz( sz, VecInt( 2, 2 ) );

double expValue( const Tensor gamma[], const Tensor l[], const Tensor & op );


int main()
{
    const int dimSite = 2;
    double xfield = 0.0;
    double zfield = 0.0;

    cout << "Input xfield: ";
    cin >> xfield;

    cout << "Input zfield: ";
    cin >> zfield;

    Tensor H = -kron( Sz, Sz ) +
        xfield / 2.0 * ( kron( Sx, linalg::eye( 2 ) ) + kron( linalg::eye( 2 ), Sx ) ) +
        zfield / 2.0 * ( kron( Sz, linalg::eye( 2 ) ) + kron( linalg::eye( 2 ), Sz ) );


    int dimBound = 100;
    double deltaT = 0.005;

    Tensor gamma[ 2 ] = { Tensor( dimBound * dimSite * dimBound ),
                          Tensor( dimBound * dimSite * dimBound ) };

    Tensor l[ 2 ] = { Tensor( dimBound ), Tensor( dimBound ) };

    randTensor( gamma[ 0 ] );
    randTensor( gamma[ 1 ] );
    randTensor( l[ 0 ] );
    randTensor( l[ 1 ] );

    gamma[ 0 ].reshape( Shape( 3, dimBound, dimSite, dimBound ) );
    gamma[ 1 ].reshape( Shape( 3, dimBound, dimSite, dimBound ) );

    Tensor U;
    linalg::expm( -deltaT * H, U );

    VecIndex slice( 2 );
    slice[ 0 ] = Index( 0, dimBound * dimSite );
    slice[ 1 ] = Index( 0, dimBound );

    VecInt shape_1 = Shape( 3, dimBound, dimSite * dimSite, dimBound );
    VecInt shape_2 = Shape( 3, dimBound, dimSite, dimBound );
    VecInt shape_3( 2, dimBound * dimSite );

    Tensor u, s, v;
    Tensor lB_inv;
    Tensor theta;
    try
    {
        int A, B;
        double t = 0.0;
        while ( t < 5.0 )
        {
            for ( A = 0; A < 2; ++A )
            {
                B = ( A + 1 ) % 2;

                theta = gamma[ A ];
                theta.prod( axes::axis0, l[ B ] ).prod( axes::axis2, l[ A ] );
                theta.dot( axes::axis2, gamma[ B ], axes::axis0 );
                theta.prod( axes::axis3, l[ B ] );

                theta.reshape( shape_1 );
                theta.dot( axes::axis1, U, axes::axis1 ).swapaxes( 1, 2 );

                linalg::svd( theta.reshape( shape_3 ), u, s, v );

                l[ A ] = s[ slice[ 1 ] ] / linalg::norm( s[ slice[ 1 ] ] );
                lB_inv = 1.0 / l[ B ];
                gamma[ A ] = u[ slice ].reshape( shape_2 ).prod( axes::axis0, lB_inv );
                gamma[ B ] = v[ slice[ 1 ] ].reshape( shape_2 ).prod( axes::axis2, lB_inv );
            }

            t += deltaT;
        }

        cout << expValue( gamma, l, Sz ) << endl;
    }
    catch ( const char * s )
    {
        cout << s << endl;
    }

    return 0;
}


double expValue( const Tensor gamma[], const Tensor l[], const Tensor & op )
{
    const int dimBound = l[ 0 ].size();

    Tensor theta;
    Tensor expVal;
    int A, B;
    double outExpValue = 0.0;
    for ( A = 0; A < 2; ++A )
    {
        B = ( A + 1 ) % 2;

        theta = gamma[ A ];
        theta.prod( axes::axis0, l[ B ] ).prod( axes::axis2, l[ A ] );
        theta.dot( axes::axis1, tensordot( theta, axes::axis1, op, axes::axis1 ), axes::axis2 );
        outExpValue += theta.trace( 0, 2 ).trace();
    }

    return outExpValue / 2.0;
}
