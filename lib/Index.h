
namespace pschangLib
{
    class Index
    {
        public:
            explicit Index( int iBegin = 0 )
            {
                if ( iBegin < 0 )
                    throw "begin value can not less than zero.\n";

                begin_ = iBegin;
                extent_ = 1;
                stride_ = 1;
            }

            Index( int iBegin, int iEnd, int iStride = 1 )
            {
                if ( iBegin < 0 )
                    throw "begin value can not less than zero.\n";

                begin_ = iBegin;
                if ( iEnd < begin_ )
                    throw "iEnd value must larger than or equal to iBegin.\n";
                if ( iStride <= 0 )
                    throw "iStride must larger than zero.\n";

                stride_ = iStride;
                extent_ = ( iEnd - begin_ ) / stride_;
            }

            Index & operator= ( int iBegin )
            {
                if ( iBegin < 0 )
                    throw "begin value can not less than zero.\n";

                begin_ = iBegin;
                extent_ = 1;
                stride_ = 1;

                return *this;
            }

            size_t begin_;
            size_t extent_;
            size_t stride_;
    };
}
