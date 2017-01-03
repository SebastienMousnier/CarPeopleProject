/*
 * Project:  Inverse Perspective Mapping
 *
 * File:     main.cpp
 *
 * Contents: Creation, initialisation and usage of IPM object
 *           for the generation of Inverse Perspective Mappings of images or videos
 *
 * Author:   Marcos Nieto <marcos.nieto.doncel@gmail.com>
 * Date:	 22/02/2014
 * Homepage: http://marcosnietoblog.wordpress.com/
 */

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <ctime>

#include "IPM.h"
#include "Blob.h"

using namespace cv;
using namespace std;



// global variables ///////////////////////////////////////////////////////////////////////////////
const cv::Scalar SCALAR_BLACK = cv::Scalar(0.0, 0.0, 0.0);
const cv::Scalar SCALAR_WHITE = cv::Scalar(255.0, 255.0, 255.0);
const cv::Scalar SCALAR_YELLOW = cv::Scalar(0.0, 255.0, 255.0);
const cv::Scalar SCALAR_GREEN = cv::Scalar(0.0, 200.0, 0.0);
const cv::Scalar SCALAR_RED = cv::Scalar(0.0, 0.0, 255.0);


// function prototypes ////////////////////////////////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs);
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex);
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs);
double distanceBetweenPoints(cv::Point point1, cv::Point point2);
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName);
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName);
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount);
void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy);
void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy);



int main( int _argc, char** _argv )
{
	// Images
	Mat inputImg, inputImgGray;
	Mat outputImg;

    Mat test_hough;
    Mat output_test_hough;

    test_hough = imread("test_hough.png");
    output_test_hough = test_hough;

	if( _argc != 2 )
	{
		cout << "Usage: ipm.exe <videofile>" << endl;
        return 1;
	}

	// Video
	string videoFileName = _argv[1];	
	cv::VideoCapture video;
	if( !video.open(videoFileName) )
		return 1;

    // Show video information
    int width = 0, height = 0, fps = 0, fourcc = 0;
    width = static_cast<int>(video.get(CV_CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(video.get(CV_CAP_PROP_FRAME_HEIGHT));
    fps = static_cast<int>(video.get(CV_CAP_PROP_FPS));
    fourcc = static_cast<int>(video.get(CV_CAP_PROP_FOURCC));

	cout << "Input video: (" << width << "x" << height << ") at " << fps << ", fourcc = " << fourcc << endl;
	
	// The 4-points at the input image	
	vector<Point2f> origPoints;
	origPoints.push_back( Point2f(0, height) );
	origPoints.push_back( Point2f(width, height) );
    origPoints.push_back( Point2f(width/2+30, 300) );
    origPoints.push_back( Point2f(width/2-50, 300) );

	// The 4-points correspondences in the destination image
	vector<Point2f> dstPoints;
	dstPoints.push_back( Point2f(0, height) );
	dstPoints.push_back( Point2f(width, height) );
	dstPoints.push_back( Point2f(width, 0) );
	dstPoints.push_back( Point2f(0, 0) );
		
	// IPM object
	IPM ipm( Size(width, height), Size(width, height), origPoints, dstPoints );
	
	// Main loop
    int frameNum = 1;
    video >> inputImg;

    outputImg = inputImg;

    char chCheckForEscKey = 0;

    // Color Conversion
    if(test_hough.channels() == 3)
        cvtColor(test_hough, test_hough, CV_BGR2GRAY);
    else
        test_hough.copyTo(test_hough);


    //Apply thresholding
    //cv::threshold(test_hough, test_hough, 100, 255, cv::THRESH_BINARY);

    Canny(test_hough,test_hough,500,300);

    imshow("Input", test_hough);

    // Transforme l'image pour unifié les contour d'une même ligne détecté
     cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));

     for (unsigned int i = 0; i < 2; i++) {
         cv::dilate(test_hough, test_hough, structuringElement);
         cv::dilate(test_hough, test_hough, structuringElement);
         cv::erode(test_hough, test_hough, structuringElement);
     }

     imshow("Output", test_hough);


     std::vector<std::vector<cv::Point> > contours;

     cv::findContours(test_hough, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

     drawAndShowContours(test_hough.size(), contours, "imgContours");

     std::vector<std::vector<cv::Point> > convexHulls(contours.size());

     for (unsigned int i = 0; i < contours.size(); i++) {
         cv::convexHull(contours[i], convexHulls[i]);
     }

     drawAndShowContours(test_hough.size(), convexHulls, "imgConvexHulls");


     /**
       * Pour chaque différents blobs détectés, on test si il correspond à un objet en mouvement ou à du bruit
       */

     std::vector<Blob> currentFrameBlobs;

     // detecte les lignes
     for (auto &convexHull : convexHulls) {
         Blob possibleBlob(convexHull);

         if ((cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) < 0.50     ){
            currentFrameBlobs.push_back(possibleBlob);
         }
     }

     std::vector<Blob> blobs(currentFrameBlobs);

     drawAndShowContours(test_hough.size(), blobs, "imgBlobs");

     //drawBlobInfoOnImage(blobs, output_test_hough);


     Blob line1(blobs.at(0));
     Blob line2(blobs.at(1));
     for (Blob &blob : blobs)
     {
        if(blob.currentBoundingRect.height > line1.currentBoundingRect.height)
        {
            line1 = blob;
        }
        else if (blob.currentBoundingRect.height > line2.currentBoundingRect.height)
        {
            line2 = blob;
        }
     }

  //  blobs.empty();
  //  blobs.push_back(line1);
  //  blobs.push_back(line2);
    drawBlobInfoOnImage(blobs, output_test_hough);

    imshow("Output 2", output_test_hough);


    while (video.isOpened() && chCheckForEscKey != 27)
    {/*
		printf("FRAME #%6d ", frameNum);
		fflush(stdout);
		frameNum++;

		// Get current image		
        video >> inputImg;

		if( inputImg.empty() )
			break;

		 // Color Conversion
		 if(inputImg.channels() == 3)		 
			 cvtColor(inputImg, inputImgGray, CV_BGR2GRAY);				 		 
		 else	 
             inputImg.copyTo(inputImgGray);


         Canny(inputImgGray,inputImgGray,600,300);

		 // Process
         clock_t begin = clock();
         //ipm.applyHomography( inputImg, outputImg );
		 clock_t end = clock();
         double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
		 printf("%.2f (ms)\r", 1000*elapsed_secs);


         vector<Vec2f> lines;
        // imshow("Input",test_hough);
         HoughLines(test_hough, lines, 1, M_PI/18, 281/7);

     //    std::cout<< "lines.size = "<<lines.size()<<"  " <<  std::endl;


         for (size_t i = 0; i < lines.size(); i++)
         {
             float rho = lines[i][0];
             float theta = lines[i][1];

             double a = cos(theta), b = sin(theta);
             double x0 = a * rho, y0 = b * rho;

             Point pt1(cvRound(x0 + 1000 * (-b)),
                       cvRound(y0 + 1000 * (a)) );

             Point pt2(cvRound(x0 - 1000 * (-b)),
                       cvRound(y0 - 1000 * (a)) );

             clipLine(output_test_hough.size(), pt1, pt2);


             if(!output_test_hough.empty())
                 line(output_test_hough,pt1,pt2,Scalar(0,0,255),1,8);

             imshow("Output",output_test_hough);
             imshow("Input",test_hough);
             resizeWindow("Input",300,300);

         }

*/
         /*ipm.drawPoints(origPoints, inputImg );


		 // View		
		 imshow("Input", inputImg);
         imshow("Output", outputImg);*/
        // imshow("Input",inputImgGray);
         chCheckForEscKey = cv::waitKey(1);
    }

	return 0;	
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void matchCurrentFrameBlobsToExistingBlobs(std::vector<Blob> &existingBlobs, std::vector<Blob> &currentFrameBlobs) {

    for (auto &existingBlob : existingBlobs) {

        existingBlob.blnCurrentMatchFoundOrNewBlob = false;

        existingBlob.predictNextPosition();
    }

    for (auto &currentFrameBlob : currentFrameBlobs) {

        int intIndexOfLeastDistance = 0;
        double dblLeastDistance = 100000.0; // init = 100000

        for (unsigned int i = 0; i < existingBlobs.size(); i++) {

            if (existingBlobs[i].blnStillBeingTracked == true) {

                double dblDistance = distanceBetweenPoints(currentFrameBlob.centerPositions.back(), existingBlobs[i].predictedNextPosition);

                if (dblDistance < dblLeastDistance) {
                    dblLeastDistance = dblDistance;
                    intIndexOfLeastDistance = i;
                }
            }
        }

        if (dblLeastDistance < currentFrameBlob.dblCurrentDiagonalSize * 0.5) {
            addBlobToExistingBlobs(currentFrameBlob, existingBlobs, intIndexOfLeastDistance);
        }
        else {
            addNewBlob(currentFrameBlob, existingBlobs);
        }

    }

    for (auto &existingBlob : existingBlobs) {

        if (existingBlob.blnCurrentMatchFoundOrNewBlob == false) {
            existingBlob.intNumOfConsecutiveFramesWithoutAMatch++;
        }

        if (existingBlob.intNumOfConsecutiveFramesWithoutAMatch >= 5) {
            existingBlob.blnStillBeingTracked = false;
        }

    }

}

///////////////////////////////////////////////////////////////////////////////////////////////////
void addBlobToExistingBlobs(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex) {

    existingBlobs[intIndex].currentContour = currentFrameBlob.currentContour;
    existingBlobs[intIndex].currentBoundingRect = currentFrameBlob.currentBoundingRect;

    existingBlobs[intIndex].centerPositions.push_back(currentFrameBlob.centerPositions.back());

    existingBlobs[intIndex].dblCurrentDiagonalSize = currentFrameBlob.dblCurrentDiagonalSize;
    existingBlobs[intIndex].dblCurrentAspectRatio = currentFrameBlob.dblCurrentAspectRatio;

    existingBlobs[intIndex].blnStillBeingTracked = true;
    existingBlobs[intIndex].blnCurrentMatchFoundOrNewBlob = true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void addNewBlob(Blob &currentFrameBlob, std::vector<Blob> &existingBlobs) {

    currentFrameBlob.blnCurrentMatchFoundOrNewBlob = true;

    existingBlobs.push_back(currentFrameBlob);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
double distanceBetweenPoints(cv::Point point1, cv::Point point2) {

    int intX = abs(point1.x - point2.x);
    int intY = abs(point1.y - point2.y);

    return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName) {
    cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

    //cv::imshow(strImageName, image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<Blob> blobs, std::string strImageName) {

    cv::Mat image(imageSize, CV_8UC3, SCALAR_BLACK);

    std::vector<std::vector<cv::Point> > contours;

    for (auto &blob : blobs) {
        if (blob.blnStillBeingTracked == true) {
            contours.push_back(blob.currentContour);
        }
    }

    cv::drawContours(image, contours, -1, SCALAR_WHITE, -1);

    //cv::imshow(strImageName, image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, int &intHorizontalLinePosition, int &carCount) {
    bool blnAtLeastOneBlobCrossedTheLine = false;

    for (auto blob : blobs) {

        if (blob.blnStillBeingTracked == true && blob.centerPositions.size() >= 2) {
            int prevFrameIndex = (int)blob.centerPositions.size() - 2;
            int currFrameIndex = (int)blob.centerPositions.size() - 1;

            if (blob.centerPositions[prevFrameIndex].y < intHorizontalLinePosition && blob.centerPositions[currFrameIndex].y >= intHorizontalLinePosition) {  // haut vers bas
            //if (blob.centerPositions[prevFrameIndex].y >= intHorizontalLinePosition && blob.centerPositions[currFrameIndex].y < intHorizontalLinePosition) {  // bas vers haut
                carCount++;
                blnAtLeastOneBlobCrossedTheLine = true;
            }
        }

    }

    return blnAtLeastOneBlobCrossedTheLine;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy) {

    for (unsigned int i = 0; i < blobs.size(); i++) {

        if (blobs[i].blnStillBeingTracked == true) {
            cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, SCALAR_RED, 2);

            int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
            double dblFontScale = blobs[i].dblCurrentDiagonalSize / 60.0;
            int intFontThickness = (int)std::round(dblFontScale * 1.0);

        //    cv::putText(imgFrame2Copy, std::to_string(i), blobs[i].centerPositions.back(), intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy) {

    int intFontFace = CV_FONT_HERSHEY_SIMPLEX;
    double dblFontScale = (imgFrame2Copy.rows * imgFrame2Copy.cols) / 300000.0;
    int intFontThickness = (int)std::round(dblFontScale * 1.5);

    cv::Size textSize = cv::getTextSize(std::to_string(carCount), intFontFace, dblFontScale, intFontThickness, 0);

    cv::Point ptTextBottomLeftPosition;

    ptTextBottomLeftPosition.x = imgFrame2Copy.cols - 1 - (int)((double)textSize.width * 1.25);
    ptTextBottomLeftPosition.y = (int)((double)textSize.height * 1.25);

    cv::putText(imgFrame2Copy, std::to_string(carCount), ptTextBottomLeftPosition, intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);

}


