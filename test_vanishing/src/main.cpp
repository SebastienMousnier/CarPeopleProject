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
IPM createHomography(Mat inputImg);


int main( int _argc, char** _argv )
{
	// Images
    Mat inputImg, inputImgGray;
	Mat outputImg;

    Mat input_bw;
    Mat output_canny;
    Mat output_with_blobs;


	// Video
    string videoFileName = "bvd11novembre.mp4";
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
    video >> inputImg;

    inputImg.copyTo(outputImg);
    inputImg.copyTo(output_with_blobs);

    char chCheckForEscKey = 0;


    IPM ipm = createHomography( outputImg);


    while (video.isOpened() && chCheckForEscKey != 27)
    {

		// Get current image		
        video >> inputImg;

		if( inputImg.empty() )
			break;

        ipm.applyHomography( inputImg, outputImg );

        imshow("Output final", outputImg);

        chCheckForEscKey = cv::waitKey(1);
    }

	return 0;	
}

IPM createHomography(Mat inputImg)
{

    Mat input_bw;
    Mat outputImg;
    Mat output_canny;
    Mat output_with_blobs;

    inputImg.copyTo(output_with_blobs);

    int width = 0, height = 0;
    width = inputImg.cols;
    height = inputImg.rows;

    // Color Conversion
    if(inputImg.channels() == 3)
        cvtColor(inputImg, input_bw, CV_BGR2GRAY);
    else
        inputImg.copyTo(input_bw);

    Canny(input_bw, output_canny,500,300);

    // Transforme l'image pour unifier les contours d'une même ligne détectée
     cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));

     for (unsigned int i = 0; i < 2; i++) {
         cv::dilate(output_canny, output_canny, structuringElement);
         cv::dilate(output_canny, output_canny, structuringElement);
         cv::erode(output_canny, output_canny, structuringElement);
     }

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
       * Pour chaque différents blobs détectés, on test si il correspond à une ligne ou pas ( en fonction de l'espace occupé par le blob blanc dans le rectangle)
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
            // On enregistre la nouvelle ligne en ecrasent la plus petite entre line1 et line2
            if (line2.currentBoundingRect.height > line1.currentBoundingRect.height)
                line1 = blob;
            else
                line2 = blob;
        }
        else if (blob.currentBoundingRect.height > line2.currentBoundingRect.height)
        {
            if (line2.currentBoundingRect.height > line1.currentBoundingRect.height)
                line1 = blob;
            else
                line2 = blob;
            line2 = blob;
        }
     }

    // On vide blobs, et on le remplit avec les 2 meilleures lignes de fuite trouvées
    blobs.clear();
    blobs.push_back(line1);
    blobs.push_back(line2);
  //  drawBlobInfoOnImage(blobs, output_with_blobs);

    /**
      * Recherche des 2 lignes de fuite
      */

    cv::Point center1, center2;
    center1 = line1.centerPositions.back();
    center2 = line2.centerPositions.back();

    cv::Point high_corner1_left, high_corner2_left;
    high_corner1_left.x = center1.x - line1.currentBoundingRect.width /2; high_corner1_left.y = center1.y - line1.currentBoundingRect.height /2;
    high_corner2_left.x = center2.x - line2.currentBoundingRect.width /2; high_corner2_left.y = center2.y - line2.currentBoundingRect.height /2;

    ///Stockage coeff dir
    float coef1 = float(line1.currentBoundingRect.height )/ float(line1.currentBoundingRect.width); // Calcul en valeur absolue avec la bounding box
    float coef2 = float(line2.currentBoundingRect.height) / float(line2.currentBoundingRect.width);

    // si le coin haut gauche est noir, on passe le coeff en négatif
    if(output_canny.at<uchar>(high_corner1_left) == 0)
    {
        coef1 *= -1.0;
    }
    if(output_canny.at<uchar>(high_corner2_left) == 0)
    {
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

    /**
     * Recherche des 4 points pour l'homographie
     *
     */
    /// Résolution de l'equation pour les points de départ

    // Ecart, parametre à ajuster en fonction de la video
    const unsigned int ecart = 80;

    Point2f center_left(center1.x, center1.y);
    Point2f center_right(center2.x, center2.y);

    Point2f pt_left_down(center_left.x + (height - center_left.y) / coef1, height);
   // Point2f pt_left_up(width/2 - ecart, int (coef1 * (width/2 - ecart) + height));
    Point2f pt_left_up(0, 0);

    Point2f pt_right_down(center_right.x + (height - center_right.y) / coef2, height);
   // Point2f pt_right_up(width/2 + ecart, int (coef1 * (width/2 - ecart) + height));
    Point2f pt_right_up(0, 0);

    std::cout<< "coef1= "<< coef1<< "  coef2= "<< coef2<< std::endl;


    /// Resolution de l'equation point d'arrives

    pt_left_up.x = (coef1*pt_left_down.x - coef2*pt_right_down.x + pt_right_down.y - pt_left_down.y + coef2 * ecart) / (coef1 - coef2);
    pt_right_up.x = ecart + pt_left_up.x;
    pt_left_up.y = coef1 * (pt_left_up.x - pt_left_down.x) + pt_left_down.y;
    pt_right_up.y = coef2 * (pt_right_up.x - pt_right_down.x) + pt_left_down.y;   // should have pt_right_up.y == pt_left_up.y

    /**
      * Application de l'homographie
      */
    // The 4-points at the input image
    vector<Point2f> origPoints;

/*    // Point placé manuellement, bon
    origPoints.push_back( Point2f(50, height) );
    origPoints.push_back( Point2f(width-300, height) );
    origPoints.push_back( Point2f(width/2 , 230) );
    origPoints.push_back( Point2f(width/2 - 70, 230) );
*/
  // Point trouvé par resolution d'equation
    origPoints.push_back( pt_left_down );
    origPoints.push_back( pt_right_down );
    origPoints.push_back( pt_right_up );
    origPoints.push_back( pt_left_up );

    /////////////////


    // The 4-points correspondences in the destination image
    vector<Point2f> dstPoints;

    dstPoints.push_back( Point2f(50, height) );
    dstPoints.push_back( Point2f(width-300, height) );
    dstPoints.push_back( Point2f(width-300, 0) );
    dstPoints.push_back( Point2f(50, 0) );

    // IPM object
    IPM ipm( Size(width, height), Size(width, height), origPoints, dstPoints );

    ipm.applyHomography( inputImg, outputImg );

    ipm.drawPoints(origPoints, output_with_blobs, 0 );
    ipm.drawPoints(dstPoints, output_with_blobs, 1 );

    imshow("Input blobs + ligne de fuite", output_with_blobs);

    return ipm;
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


