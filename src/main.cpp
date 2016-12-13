#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main(int argc, char** argv )
{

    string filename = "data/vehicleDetection.mp4";
    VideoCapture capture(filename);
    Mat frame;


    int totalframe = capture.get(CV_CAP_PROP_FRAME_COUNT);

    std::cout<<"toralframe = "<< totalframe << std::endl;

    if( !capture.isOpened() )
        throw "Error when reading steam_avi";

    namedWindow( "window", 1);
    for( ; ; )
    {
        capture >> frame;
        if(frame.empty())
            break;
        imshow("window", frame);
        waitKey(20); // waits to display frame
    }
    waitKey(0); // key press to close window

    return 0;

}
