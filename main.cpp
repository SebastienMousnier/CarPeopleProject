// main.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>
//#include<conio.h>           // it may be necessary to change or remove this line if not using Windows

#include "Blob.h"

#define SHOW_STEPS            // un-comment or comment this line to show steps or not
using namespace std;
using namespace cv;


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

///////////////////////////////////////////////////////////////////////////////////////////////////
int main(void) {

    cv::VideoCapture capVideo;

    cv::Mat imgFrame1;
    cv::Mat imgFrame2;

    Mat imgFrameNB;

    std::vector<Blob> blobs;

    cv::Point crossingLine[2];

    int carCount = 0;

    capVideo.open("bvd11novembre.mp4");

    if (!capVideo.isOpened()) {                                                 // if unable to open video file
        std::cout << "error reading video file" << std::endl << std::endl;      // show error message
        //_getch();                   // it may be necessary to change or remove this line if not using Windows
        return(0);                                                              // and exit program
    }





    if (capVideo.get(CV_CAP_PROP_FRAME_COUNT) < 2) {
        std::cout << "error: video file must have at least two frames";
        //_getch();                   // it may be necessary to change or remove this line if not using Windows
        return(0);
    }


    // Show video information
    int width = 0, height = 0, fps = 0, fourcc = 0;
    width = static_cast<int>(capVideo.get(CV_CAP_PROP_FRAME_WIDTH));
    height = static_cast<int>(capVideo.get(CV_CAP_PROP_FRAME_HEIGHT));
    fps = static_cast<int>(capVideo.get(CV_CAP_PROP_FPS));
    fourcc = static_cast<int>(capVideo.get(CV_CAP_PROP_FOURCC));

    cout << "Input video: (" << width << "x" << height << ") at " << fps << ", fourcc = " << fourcc << endl;

    capVideo.read(imgFrame1);
    capVideo.read(imgFrame2);

    //Mat new_image( imgFrame1.size(), imgFrame1.type() );

    int intHorizontalLinePosition = (int)std::round((double)imgFrame1.rows * 0.35);

    crossingLine[0].x = 0;
    crossingLine[0].y = intHorizontalLinePosition;

    crossingLine[1].x = imgFrame1.cols - 1;
    crossingLine[1].y = intHorizontalLinePosition;

    char chCheckForEscKey = 0;

    bool blnFirstFrame = true;

    int frameCount = 2;



    while (capVideo.isOpened() && chCheckForEscKey != 27) {

        std::vector<Blob> currentFrameBlobs;

        cv::Mat imgFrame1Copy = imgFrame1.clone();
        cv::Mat imgFrame2Copy = imgFrame2.clone();

        cv::Mat imgDifference;
        cv::Mat imgThresh;

        cv::cvtColor(imgFrame1Copy, imgFrame1Copy, CV_BGR2GRAY);
        cv::cvtColor(imgFrame2Copy, imgFrame2Copy, CV_BGR2GRAY);

        cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0);
        cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(5, 5), 0);

        cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);

        cv::threshold(imgDifference, imgThresh, 30, 255.0, CV_THRESH_BINARY);

        //cv::imshow("imgThresh", imgThresh);

        // Plus l'élément est grand plus on élargit la zone de blanc
        cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        cv::Mat structuringElement15x15 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));

        for (unsigned int i = 0; i < 2; i++) {
            cv::dilate(imgThresh, imgThresh, structuringElement15x15);
            cv::dilate(imgThresh, imgThresh, structuringElement15x15);
            cv::erode(imgThresh, imgThresh, structuringElement15x15);
        }

        cv::Mat imgThreshCopy = imgThresh.clone();

        std::vector<std::vector<cv::Point> > contours;

        cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        drawAndShowContours(imgThresh.size(), contours, "imgContours");

        std::vector<std::vector<cv::Point> > convexHulls(contours.size());

        for (unsigned int i = 0; i < contours.size(); i++) {
            cv::convexHull(contours[i], convexHulls[i]);
        }

        drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");

        for (auto &convexHull : convexHulls) {
            Blob possibleBlob(convexHull);

            if (possibleBlob.currentBoundingRect.area() > 400 &&
                possibleBlob.dblCurrentAspectRatio > 0.2 &&
                possibleBlob.dblCurrentAspectRatio < 4.0 &&
                possibleBlob.currentBoundingRect.width > 30 &&
                possibleBlob.currentBoundingRect.height > 30 &&
                possibleBlob.dblCurrentDiagonalSize > 60.0 &&
                (cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.50) {
                currentFrameBlobs.push_back(possibleBlob);
            }
        }

        drawAndShowContours(imgThresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");

        if (blnFirstFrame == true) {
            for (auto &currentFrameBlob : currentFrameBlobs) {
                blobs.push_back(currentFrameBlob);
            }
        } else {
            matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs);
        }

        drawAndShowContours(imgThresh.size(), blobs, "imgBlobs");



        imgFrame2Copy = imgFrame2.clone();          // get another copy of frame 2 since we changed the previous frame 2 copy in the processing above

        drawBlobInfoOnImage(blobs, imgFrame2Copy);

        bool blnAtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLine(blobs, intHorizontalLinePosition, carCount);

        if (blnAtLeastOneBlobCrossedTheLine == true) {
            cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_GREEN, 2);
        }
        else {
            cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_RED, 2);
        }

        drawCarCountOnImage(carCount, imgFrame2Copy);


        // blob tracking tracking

        cv::Point centreBlob;

        // déclaration d'un vecteur de point pour l'image noir (utilisation K-means)
        std::vector<std::vector<cv::Point>> stockageCenterBlob;
        cv::vector<cv::Point> stockageCenterIntermediaire;



        //
        for (Blob &blobtest : blobs)
        {
            stockageCenterIntermediaire.clear();
           for (cv::Point &center : blobtest.centerPositions)
           {
               //cv::vector<cv::Point> stockageBlob;
                cv::circle(imgFrame2Copy, center, 1, SCALAR_WHITE, 1);
                centreBlob = center; //blobtest.centerPositions.front();
                // on stocke nos centre dans le vecteur stockage
                stockageCenterIntermediaire.push_back(center);
            }
           stockageCenterBlob.push_back(stockageCenterIntermediaire);
        }


        // K-means
        // déclaration d'une image noir
        cv::cvtColor(imgFrame2Copy, imgFrameNB, CV_BGR2GRAY);
        Mat imgFrameBlobTrackingNoir(imgFrame2Copy.rows, imgFrame2Copy.cols, CV_8UC3, Scalar(0,0,0));
        //imshow("imgFrameBlobTrackingNoir", imgFrameBlobTrackingNoir);
        imgFrameNB = imgFrameBlobTrackingNoir.clone();



for (std::vector<cv::Point> &vecteurCentre : stockageCenterBlob)
{
    cv::Point Pointsauvegarde = vecteurCentre.front();

    //cv::Point pointCenter =  pointCentre.at(<Point>);
    for (cv::Point &pointCentre : vecteurCentre)
    {
        line(imgFrameNB,Pointsauvegarde,pointCentre, SCALAR_WHITE, 1,8, 0);
        Pointsauvegarde = pointCentre;
       // cv::circle(imgFrameNB, pointCentre, 1, SCALAR_WHITE, 1);


    }

}




       //imshow("clone", imgFrameNB );
       imwrite("frame.png",imgFrameNB);

       // debut Kmeans

       Mat src = imread("frame.png", 1);
       Mat samples(src.rows * src.cols, 3, CV_32F);
       for( int y = 0; y < src.rows; y++ )
         for( int x = 0; x < src.cols; x++ )
           for( int z = 0; z < 3; z++)
             samples.at<float>(y + x*src.rows, z) = src.at<Vec3b>(y,x)[z];


         int clusterCount = 4;
         Mat labels;
         int attempts = 5;
         Mat centers;
         kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 0.0001, 10000), attempts, KMEANS_PP_CENTERS, centers );


         Mat new_image( src.size(), src.type() );
         //new_image.reshape(src.cols,src.rows);
         for( int y = 0; y < src.rows; y++ )
         {
           for( int x = 0; x < src.cols; x++ )
           {
             int cluster_idx = labels.at<int>(y + x*src.rows,0);
             new_image.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
             new_image.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
             new_image.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
           }
         }
         //imshow("original", src);
         imwrite("Kmeans.png",new_image);

         //std::cout << "centers " << centers  <<std::endl;

   //fin Kmeans



   /* //std::vector<Blob> existingBlobs;
    int intTailleVecteurBlobs;
    cv::Point pointBlobDistance1;
    cv::Point pointBlobDistance2;
    double distance;
    for (Blob &blobtest : blobs){
    //cv:Point pointBlob;
    cv::Point pointcenter = blobtest.centerPositions.back();


  //blobtest.centerPositions = color(1.0,1.0,1.0);
 if ( pointcenter.x < bord0 - 0.3*width  && pointcenter.x > 0.1*width && pointcenter.y >500 && pointcenter.y < bord1 - 500)
 {
    // std::cout << "yesss" <<std::endl;
     //std::cout << "pointcenter.x= " << pointcenter.x << " pointcenter.y= " << pointcenter.y <<std::endl<<std::endl;

     for (cv::Point &center : blobtest.centerPositions)
     {
          cv::circle(imgFrameNB, center, 3, SCALAR_WHITE, 5);

        //  std::cout << "test hello" <<std::endl;
          // calcul vitesse
        //cv::Point &centerPrecedent : blobtest.predictedNextPosition.



  //std::cout << "hors de la boucle" << blobtest.centerPositions <<std::endl<<std::endl;
          intTailleVecteurBlobs = blobtest.centerPositions.size();
   //cv::vector<pointBlob>  existingBlobs = blobtest.centerPositions.s;
  //std::cout << "taille existingBlobs" <<  blobtest.centerPositions.size()  <<std::endl;
          pointBlobDistance1 = blobtest.centerPositions.back();
          pointBlobDistance2 = blobtest.centerPositions.at(intTailleVecteurBlobs-1);
          distance = distanceBetweenPoints(pointBlobDistance1,pointBlobDistance2);
  // std::cout << "distance " << distance  <<std::endl;


      }

          // fin calcul vitesse

  }
 else
 {
     //std::cout << "noo" <<std::endl;
 }

}*/










        //cv::imshow("imgFrame2Copy", imgFrame2Copy);

        //cv::waitKey(0);                 // uncomment this line to go frame by frame for debugging
        
                // now we prepare for the next iteration

        currentFrameBlobs.clear();

        imgFrame1 = imgFrame2.clone();           // move frame 1 up to where frame 2 is

        if ((capVideo.get(CV_CAP_PROP_POS_FRAMES) + 1) < capVideo.get(CV_CAP_PROP_FRAME_COUNT)) {

            for(int i=0 ; i<5; ++i)
                capVideo.read(imgFrame2);
        }
        else {
            std::cout << "end of video\n";
            break;
        }
        blnFirstFrame = false;
        frameCount++;
        chCheckForEscKey = cv::waitKey(1);
    }

    //imshow( "clustered image", new_image );

    imwrite("frame.png",imgFrameNB);





    if (chCheckForEscKey != 27) {               // if the user did not press esc (i.e. we reached the end of the video)
        cv::waitKey(0);                         // hold the windows open to allow the "end of video" message to show
    }
    // note that if the user did press esc, we don't need to hold the windows open, we can simply let the program end which will close the windows

    return(0);
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

////////////////////////////////////////////////////////////////////////////////////////////////
/*void blobtracking( Blob &currentFrameBlob, std::vector<Blob> &existingBlobs, int &intIndex) {

        existingBlobs[intIndex].


}*/









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

            cv::putText(imgFrame2Copy, std::to_string(i), blobs[i].centerPositions.back(), intFontFace, dblFontScale, SCALAR_GREEN, intFontThickness);
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










