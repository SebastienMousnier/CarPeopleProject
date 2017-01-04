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
    Mat input_bw;
    Mat output_canny;
    Mat output_with_blobs;

    //test_hough = imread("test_hough.png");
/*
    if( _argc != 2 )
	{
		cout << "Usage: ipm.exe <videofile>" << endl;
        return 1;
    }
*/
	// Video
    string videoFileName = "ourVideo.mp4";
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


	// Main loop
    int frameNum = 1;
    video >> inputImg;


    inputImg.copyTo(outputImg);
    inputImg.copyTo(output_with_blobs);

    char chCheckForEscKey = 0;

    // Color Conversion
    if(outputImg.channels() == 3)
        cvtColor(outputImg, input_bw, CV_BGR2GRAY);
    else
        outputImg.copyTo(input_bw);

    Canny(input_bw, output_canny,500,300);

    // Transforme l'image pour unifier les contours d'une même ligne détectée
     cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));

     for (unsigned int i = 0; i < 2; i++) {
         cv::dilate(output_canny, output_canny, structuringElement);
         cv::dilate(output_canny, output_canny, structuringElement);
         cv::erode(output_canny, output_canny, structuringElement);
     }

     imshow("Output canny + morpho", output_canny);


     std::vector<std::vector<cv::Point> > contours;

     Mat tmp_contour;  // besoin d'une copie car finContours() écrase la matrice
     output_canny.copyTo(tmp_contour);

     cv::findContours(tmp_contour, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

     drawAndShowContours(output_canny.size(), contours, "imgContours");

     std::vector<std::vector<cv::Point> > convexHulls(contours.size());

     for (unsigned int i = 0; i < contours.size(); i++) {
         cv::convexHull(contours[i], convexHulls[i]);
     }

     drawAndShowContours(output_canny.size(), convexHulls, "imgConvexHulls");


     /**
       * Pour chaque différents blobs détectés, on test si il correspond à une ligne ou pas ( en fonction de l'espace occupé par le blob blanc dans le rectabgle)
       */

     std::vector<Blob> currentFrameBlobs;

     // detecte les lignes
     for (auto &convexHull : convexHulls) {
         Blob possibleBlob(convexHull);

         if ((cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) < 0.30     ){
            currentFrameBlobs.push_back(possibleBlob);
         }
     }

     std::vector<Blob> blobs(currentFrameBlobs);

     drawAndShowContours(output_canny.size(), blobs, "imgBlobs");


     /**
      * On récupere les 2 plus grandes lignes (celles qui ont la plus grande hauteur) afin de les utiliser pour l'homographie)
      */
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

    blobs.clear();
    blobs.push_back(line1);
    blobs.push_back(line2);
    drawBlobInfoOnImage(blobs, output_with_blobs);

    /**
      * Recherche des 2 lignes de fuite
      */

    cv::Point center1, center2;
    center1 = line1.centerPositions.back();
    center2 = line2.centerPositions.back();

    cv::Point high_corner1_left, high_corner2_left;

    high_corner1_left.x = center1.x - line1.currentBoundingRect.width /2; high_corner1_left.y = center1.y - line1.currentBoundingRect.height /2;

    high_corner2_left.x = center2.x - line2.currentBoundingRect.width /2; high_corner2_left.y = center2.y + line2.currentBoundingRect.height /2;

    ///Stockage coeff dir
    float coef1 = float(line1.currentBoundingRect.width )/ float(line1.currentBoundingRect.height); // Calcul en valeur absolue avec la bounding box
    float coef2 = float(line2.currentBoundingRect.width) / float(line2.currentBoundingRect.height);

    // si le coin haut gauche est noir, on passe le coeff en négatif
    std::cout<< "output_canny.at<uchar>(high_corner1_left)= "<< int(output_canny.at<uchar>(high_corner1_left))<< std::endl;
    std::cout<< "output_canny.at<uchar>(high_corner2_left)= "<< int(output_canny.at<uchar>(high_corner2_left))<< std::endl;
    if(output_canny.at<uchar>(high_corner1_left) == 0)
    {
        std::cout<< "yesssss"<< std::endl;
        coef1 *= -1.0;
    }
    if(output_canny.at<uchar>(high_corner2_left) == 0)
    {
        std::cout<< "yesssss222"<< std::endl;
        coef2 *= -1.0;
    }

    // Choix côté
    if(coef1 > 0 && coef2 < 0)
    {
        float tmp = coef1;
        coef1 = coef2;
        coef2 = tmp;
    }
    else if (coef1 * coef2 > 0)
    {
        if (coef1 < coef2)
        {
            float tmp = coef1;
            coef1 = coef2;
            coef2 = tmp;
        }
    }

    Point2f orig_left(0, height);
    Point2f point_left(500, int (500*coef1));

    Point2f orig_right(width, height);
    Point2f point_right(width-500, int (coef2*(width - 500)));

    std::cout<< "coef1= "<< coef1<< "  coef2= "<< coef2<< std::endl;
    std::cout<< "int (coef2*(width - 500))= "<< int (coef2*(width - 500))<< std::endl;
  //  std::cout<< "coef1= "<< coef1<< "  coef2= "<< coef2<< std::endl;

    /**
      * Application de l'homographie
      */
    // The 4-points at the input image
    vector<Point2f> origPoints;

    origPoints.push_back( orig_left );
    origPoints.push_back( orig_right );
    origPoints.push_back( point_right );
    origPoints.push_back( point_left );

    // The 4-points correspondences in the destination image
    vector<Point2f> dstPoints;
    dstPoints.push_back( Point2f(0, height) );
    dstPoints.push_back( Point2f(width, height) );
    dstPoints.push_back( Point2f(width, 0) );
    dstPoints.push_back( Point2f(0, 0) );

    // IPM object
    IPM ipm( Size(width, height), Size(width, height), origPoints, dstPoints );

    ipm.applyHomography( inputImg, outputImg );

    imshow("Output homographie", outputImg);

    ipm.drawPoints(origPoints, output_with_blobs );

    imshow("Input blobs + ligne de fuite", output_with_blobs);

    while (video.isOpened() && chCheckForEscKey != 27)
    {
/*		printf("FRAME #%6d ", frameNum);
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

             clipLine(output_with_blobs.size(), pt1, pt2);


             if(!output_with_blobs.empty())
                 line(output_with_blobs,pt1,pt2,Scalar(0,0,255),1,8);

             imshow("Output",output_with_blobs);
             imshow("Input",test_hough);
             resizeWindow("Input",300,300);

         }


         /*ipm.drawPoints(origPoints, inputImg );


		 // View		
		 imshow("Input", inputImg);
         imshow("Output", outputImg);*/
        // imshow("Input",inputImgGray);
        chCheckForEscKey = cv::waitKey(1);
    }

	return 0;	
}

/**
  * Applicatione de l'homographie
  */
/*
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

ipm.applyHomography( test_hough, outputImg );
*/




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


