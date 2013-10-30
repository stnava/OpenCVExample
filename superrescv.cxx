/**
 * file Smoothing.cpp
 * brief Sample code for simple filters
 * author OpenCV team
 */
#include <iostream>
#include <iomanip>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/contrib.hpp"
#include "opencv2/superres.hpp"
#include "opencv2/superres/optical_flow.hpp"
#include "opencv2/opencv_modules.hpp"
using namespace std;
using namespace cv;
using namespace cv::superres;

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

  bool useOclChanged = false;
  CommandLineParser cmd(argc, argv,
        "{ v video      |           | Input video }"
        "{ o output     | temp.jpg  | Output video }"
        "{ s scale      | 4         | Scale factor }"
        "{ i iterations | 180       | Iteration count }"
        "{ t temporal   | 4         | Radius of the temporal search area }"
        "{ f flow       | farneback | Optical flow algorithm (farneback, simple, tvl1, brox, pyrlk) }"
        "{ g            | false     | CPU as default device, cuda for CUDA and ocl for OpenCL }"
        "{ h help       | false     | Print help message }"
    );

    if (cmd.get<bool>("help"))
    {
        cout << "This sample demonstrates Super Resolution algorithms for video sequence" << endl;
        cmd.printMessage();
        return 0;
    }
 
    src = imread(argv[1], CV_LOAD_IMAGE_COLOR);   // Read the file

    if(! src.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    const string inputVideoName = cmd.get<string>("video");
    const string outputVideoName = cmd.get<string>("output");
    const int scale = cmd.get<int>("scale");
    const int iterations = cmd.get<int>("iterations");
    const int temporalAreaRadius = cmd.get<int>("temporal");
    const string optFlow = cmd.get<string>("flow");

    std::cout <<" scale " << scale << std::endl;

    Ptr<SuperResolution> superRes;
    superRes = createSuperResolution_BTVL1();
    superRes->set("scale", scale);
    superRes->set("iterations", iterations);
    superRes->set("temporalAreaRadius", temporalAreaRadius);
    //    superRes->setInput( src );

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
