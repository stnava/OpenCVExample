#include "opencv2/highgui/highgui.hpp"
using namespace cv;
int main (int argc, char **argv)
  {
  Mat img = imread(argv[1] );
  if( ! img.data )                              // Check for invalid input
    {
    return -1;
    }
  imshow( "Hello", img );
  waitKey( 0 );
  return 0;
  }
