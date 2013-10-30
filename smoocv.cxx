/**
 * file Smoothing.cpp
 * brief Sample code for simple filters
 * author OpenCV team
 */
#include <iostream>
#include <vector>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"

using namespace std;
using namespace cv;

Mat src;
Mat dst;
/// Global Variables
int DELAY_CAPTION = 1500;
int DELAY_BLUR = 100;
int MAX_KERNEL_LENGTH = 31;

char window_name[] = "Smoothing Demo";

/// Function headers
int display_caption( const char* caption );
int display_dst( int delay );

/* 8 bit, color or gray - deprecated, use CV_LOAD_IMAGE_ANYCOLOR */
#define CV_LOAD_IMAGE_UNCHANGED  -1
/* 8 bit, gray */
#define CV_LOAD_IMAGE_GRAYSCALE   0
/* 8 bit unless combined with CV_LOAD_IMAGE_ANYDEPTH, color */
#define CV_LOAD_IMAGE_COLOR       1
/* any depth, if specified on its own gray */
#define CV_LOAD_IMAGE_ANYDEPTH    2
/* by itself equivalent to CV_LOAD_IMAGE_UNCHANGED
   but can be modified with CV_LOAD_IMAGE_ANYDEPTH */
#define CV_LOAD_IMAGE_ANYCOLOR    4

/**
 * function main
 */
int main( int argc, char** argv )
{
    if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }

    src = imread(argv[1], CV_LOAD_IMAGE_ANYCOLOR);   // Read the file

    if(! src.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

   if( display_caption( "Original Image" ) != 0 ) { return 0; }

   dst = src.clone();
   if( display_dst( DELAY_CAPTION ) != 0 ) { return 0; }

   /// Applying Homogeneous blur
   if( display_caption( "Homogeneous Blur" ) != 0 ) { return 0; }

  for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
      { blur( src, dst, Size( i, i ), Point(-1,-1) );
        if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } }


  /// Applying Gaussian blur
  if( display_caption( "Gaussian Blur" ) != 0 ) { return 0; }

  for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
      { GaussianBlur( src, dst, Size( i, i ), 0, 0 );
        if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } }


  /// Applying Median blur
  if( display_caption( "Median Blur" ) != 0 ) { return 0; }

  for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
      { medianBlur ( src, dst, i );
        if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } }


  /// Applying Bilateral Filter
  if( display_caption( "Bilateral Blur" ) != 0 ) { return 0; }

  for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
      { bilateralFilter ( src, dst, i, i*2, i/2 );
        if( display_dst( DELAY_BLUR ) != 0 ) { return 0; } }

  /// Wait until user press a key
  display_caption( "End: Press a key!" );

  waitKey(0);

  return 0;
}

/**
 * @function display_caption
 */
int display_caption( const char* caption )
{
  dst = Mat::zeros( src.size(), src.type() );
  putText( dst, caption,
           Point( src.cols/4, src.rows/2),
           FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255) );

  imshow( window_name, dst );
  int c = waitKey( DELAY_CAPTION );
  if( c >= 0 ) { return -1; }
  return 0;
}

/**
 * @function display_dst
 */
int display_dst( int delay )
{
  imshow( window_name, dst );
  int c = waitKey ( delay );
  if( c >= 0 ) { return -1; }
  return 0;
}
