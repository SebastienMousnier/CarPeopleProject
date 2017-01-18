// main.cpp

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<iostream>
//#include<conio.h>           // it may be necessary to change or remove this line if not using Windows

#include "Blob.h"
#include "IPM.h"

#define SHOW_STEPS            // un-comment or comment this line to show steps or not

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
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, std::vector<Blob> &blobs_crossed_line, int &intHorizontalLinePosition, int &carCount);
void drawBlobInfoOnImage(std::vector<Blob> &blobs, cv::Mat &imgFrame2Copy);
void drawCarCountOnImage(int &carCount, cv::Mat &imgFrame2Copy);
IPM createHomography(cv::Mat inputImg);
void firstProcess(cv::Mat inputFrame1, cv::Mat inputFrame2,cv::Mat &outputThresh, IPM &ipm);
void updateTrajectoire(cv::Mat inputFrame, cv::String title, const std::vector<Blob> &existingBlobs);
void countPedestriansCarsTrucks(const std::vector<Blob> &Blobs);

///////////////////////////////////////////////////////////////////////////////////////////////////
int main(void) {

    cv::VideoCapture capVideo;

    cv::Mat imgFrame1;
    cv::Mat imgFrame2;
    cv::Mat outputImg;

    std::vector<Blob> blobs;

    // when a blob crossed the line, we stock it a vector in order that at the end to fine if it is a piedestran, car, or trunk
    std::vector<Blob> blobs_crossed_line;

    cv::Point crossingLine[2];

    int carCount = 0;

    //capVideo.open("CarsDrivingUnderBridge.mp4");

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

    capVideo.read(imgFrame1);
    capVideo.read(imgFrame2);


    /**
     * @brief intHorizontalLinePosition: creation de la ligne horizontal qui comptera les voitures
     */

    int intHorizontalLinePosition = (int)std::round((double)imgFrame1.rows * 0.55);

    crossingLine[0].x = 0;
    crossingLine[0].y = intHorizontalLinePosition;

    crossingLine[1].x = imgFrame1.cols - 1;
    crossingLine[1].y = intHorizontalLinePosition;

    char chCheckForEscKey = 0;

    bool blnFirstFrame = true;

    int frameCount = 2;


    imgFrame1.copyTo(outputImg);
    IPM ipm = createHomography( outputImg);

    while (capVideo.isOpened() && chCheckForEscKey != 27) {

        std::vector<Blob> currentFrameBlobs;

        cv::Mat imgFrame1Copy = imgFrame1.clone();
        cv::Mat imgFrame2Copy = imgFrame2.clone();

        cv::Mat imgThresh;

        firstProcess(imgFrame1Copy, imgFrame2Copy, imgThresh, ipm);

        /**
         *  Entoure par un rectangle les objets en mouvement
         */

        // besoin d'une copie car finContours() écrase la matrice
        cv::Mat imgThreshCopy = imgThresh.clone();

        std::vector<std::vector<cv::Point> > contours;

        cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        drawAndShowContours(imgThresh.size(), contours, "imgContours");

        std::vector<std::vector<cv::Point> > convexHulls(contours.size());

        for (unsigned int i = 0; i < contours.size(); i++) {
            cv::convexHull(contours[i], convexHulls[i]);
        }

        drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");


        /**
          * Pour chaque différent blob détecté, on teste si il correspond à un objet en mouvement ou à du bruit
          */

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

        bool blnAtLeastOneBlobCrossedTheLine = checkIfBlobsCrossedTheLine(blobs, blobs_crossed_line, intHorizontalLinePosition, carCount);


        // Enregistre la trajectoire des différents objets
        updateTrajectoire(imgFrame2Copy, "trajectoire.png", blobs);
        imgFrame2Copy = imgFrame2.clone();          // get another copy of frame 2 since we changed the previous frame 2 copy in the processing above

        /// affiche les informations sur l'image de départ sans homographie
        /*
        drawBlobInfoOnImage(blobs, imgFrame2Copy);

        if (blnAtLeastOneBlobCrossedTheLine == true) {
            cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_GREEN, 2);
        }
        else {
            cv::line(imgFrame2Copy, crossingLine[0], crossingLine[1], SCALAR_RED, 2);
        }
        drawCarCountOnImage(carCount, imgFrame2Copy);

        cv::imshow("imgFrame2Copy", imgFrame2Copy);
*/

        /// affiche les informations sur l'image de départ après homographie

        cv::Mat outputImg_homography_blobs = imgFrame2Copy.clone();
        ipm.applyHomography( imgFrame2Copy, outputImg_homography_blobs );

        drawBlobInfoOnImage(blobs, outputImg_homography_blobs);

        if (blnAtLeastOneBlobCrossedTheLine == true) {
            cv::line(outputImg_homography_blobs, crossingLine[0], crossingLine[1], SCALAR_GREEN, 2);
        }
        else {
            cv::line(outputImg_homography_blobs, crossingLine[0], crossingLine[1], SCALAR_RED, 2);
        }

        drawCarCountOnImage(carCount, outputImg_homography_blobs);

        cv::imshow("outputImg_homography_blobs", outputImg_homography_blobs);
        cv::imshow("input Video", imgFrame1Copy);

        cv::Mat tmp;
        ipm.applyHomography( imgFrame2Copy, tmp );

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

    if (chCheckForEscKey != 27) {               // if the user did not press esc (i.e. we reached the end of the video)
        cv::waitKey(0);                         // hold the windows open to allow the "end of video" message to show
    }
    // note that if the user did press esc, we don't need to hold the windows open, we can simply let the program end which will close the windows


    countPedestriansCarsTrucks(blobs_crossed_line);

    return(0);
}

/**
 * @brief countPedestriansCarsTrucks
 * @param blobs
 *
 * Différencie les piétons, les voitures et les camions par l'algorithme des kmeans.
 * Les paramètre du kmeans sont la largeur et la longueur des blobs
 */
void countPedestriansCarsTrucks(const std::vector<Blob> &blobs)
{
    cv::Mat drawCluster(400, 400, CV_8UC3, SCALAR_WHITE);

    std::vector<cv::Point2f> points_width_height;

    // Parcours tout les blobs afin de récuperer leur largeur et leur longeur que l'on sauvegarde dans un vecteur afin d'appliquer l'algo des kmeans
    for (const Blob &blob : blobs)
    {
        const cv::Point2f point(blob.currentBoundingRect.width, blob.currentBoundingRect.height);
      //  cv::circle(drawCluster, point, 1, SCALAR_BLACK, 2);
        points_width_height.push_back(point);
    }

    // Need to use a matrice for kmeans (assert fail i case of vector
    cv::Mat data_mat(points_width_height.size(), 2, CV_32FC1, &points_width_height[0]), labels, centers;

    /// k = 3 => Pedestrians, cars, trunks
    int k = 3;
    if (blobs.size()<3)
    {
        std::cout<<"Not enough samples in order to differenciate Pedestrians from cars " << std::endl;
        return;
    }
    cv::kmeans(data_mat, k, labels,
               cv::TermCriteria( cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
                  3, cv::KMEANS_PP_CENTERS, centers);


    std::vector<float> area_cluster;
    // Parcours tout les centre de cluster pour les afficher aussi
    for (int i = 0; i < k; ++i)
    {
        cv::Point c = centers.at<cv::Point2f>(i);
        cv::circle(drawCluster, c, 3, SCALAR_BLACK, 6);
        area_cluster.push_back(c.x * c.y);
    }

    int nb_sample_cluster1 = 0, nb_sample_cluster2 = 0, nb_sample_cluster3 = 0;

    // Parcours tous les echantillons et determine dans quels clusters ils se trouvent
    for (int i = 0; i < blobs.size(); ++i)
    {
        int clusterIdx = labels.at<int>(i);
        cv::Point ipt = data_mat.at<cv::Point2f>(i);

        if (clusterIdx == 0)
        {
            cv::circle( drawCluster, ipt, 1, SCALAR_GREEN, 2);  // Dessine un point sur l'image de sortie
            nb_sample_cluster1++;                           // Compte le nombre de point
        }
        else if (clusterIdx == 1)
        {
            cv::circle( drawCluster, ipt, 1, SCALAR_RED, 2);
            nb_sample_cluster2++;
        }
        else
        {
            cv::circle( drawCluster, ipt, 1, SCALAR_YELLOW, 2);
            nb_sample_cluster3++;
        }
    }

    float area_pedestrians, area_cars, area_trucks;

    /*if (area_cluster(0) > area_cluster(1) )
    {
        if (area_cluster(0) > area_cluster(2))
        {
            air_pedestrians = area_cluster(0);
            if (area_cluster(1) > area_cluster(2))
            {
                air_cars
                air_pedestrians = area_cluster(2);
            }
        }

    }*/


    std::map<float,int> area_and_number;
    area_and_number.insert(std::pair<float,int>(area_cluster[0],nb_sample_cluster1));
    area_and_number.insert(std::pair<float,int>(area_cluster[1],nb_sample_cluster2));
    area_and_number.insert(std::pair<float,int>(area_cluster[2],nb_sample_cluster3));

    std::map<float,int>::iterator it;
    int i=0;


    // pour parcourir toutes les paires de la map
    for(it=area_and_number.begin() ; it!=area_and_number.end() ; ++it)
    {
        if (i ==0)
            std::cout<<"Piétons: "  << it->second << std::endl;
        else if (i == 1)
            std::cout<<"Voitures: " << it->second << std::endl;
        else
            std::cout<<"Camions: "  << it->second << std::endl;
        i++;

    }



    cv::imwrite("differentiation_pietons_voiture_camion.png", drawCluster);


  //  cv::imshow("kmeans", drawCluster);

    return ;
}


/**
 * @brief updateTrajectoire
 * @param inputOuputFrame
 *      => Image sur laquelle on dessine les trajectoires
 * @param title
 *      => titre de l'image que l'on enregistre, si title == "" ; on n'enregistre pas l'image
 * @param existingBlobs
 */
void updateTrajectoire(cv::Mat inputOuputFrame, cv::String title, const std::vector<Blob> &existingBlobs)
{
    cv::Point centreBlob;

    // déclaration d'un vecteur de point pour l'image noir
    std::vector<std::vector<cv::Point>> stockageCenterBlob;
    cv::vector<cv::Point> stockageCenterIntermediaire;

    for (const Blob &blobtest : existingBlobs)
    {
       stockageCenterIntermediaire.clear();
       for (const cv::Point &center : blobtest.centerPositions)
       {
            cv::circle(inputOuputFrame, center, 1, SCALAR_WHITE, 1);
            centreBlob = center;
            // on stocke nos centre dans le vecteur stockage
            stockageCenterIntermediaire.push_back(center);
        }
       stockageCenterBlob.push_back(stockageCenterIntermediaire);
    }

    // déclaration d'une image noire
    cv::Mat imgFrameNB;
    cv::cvtColor(inputOuputFrame, imgFrameNB, CV_BGR2GRAY);
    cv::Mat imgFrameBlobTrackingNoir(inputOuputFrame.rows, inputOuputFrame.cols, CV_8UC3, cv::Scalar(0,0,0));
    //imshow("imgFrameBlobTrackingNoir", imgFrameBlobTrackingNoir);
    imgFrameNB = imgFrameBlobTrackingNoir.clone();

    for (std::vector<cv::Point> &vecteurCentre : stockageCenterBlob)
    {
        cv::Point Pointsauvegarde = vecteurCentre.front();
        for (cv::Point &pointCentre : vecteurCentre)
        {
            cv::line(imgFrameNB, Pointsauvegarde, pointCentre, SCALAR_WHITE, 1,8, 0);
            Pointsauvegarde = pointCentre;
           // cv::circle(imgFrameNB, pointCentre, 1, SCALAR_WHITE, 1);
        }
    }

    if (title != "")
    {
        cv::imwrite(title, imgFrameNB);
    }

    return;
}

/// Fonction pour récupérer les paramètres de l'homographie
IPM createHomography(cv::Mat inputImg)
{

    cv::Mat input_bw;
    cv::Mat outputImg;
    cv::Mat output_canny;
    cv::Mat output_with_blobs;

    std::cout<<"major version= " << CV_MAJOR_VERSION << std::endl;
    std::cout<<"major version= " << CV_MINOR_VERSION << std::endl;

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

     cv::Mat tmp_contour;  // besoin d'une copie car finContours() écrase la matrice
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
    drawBlobInfoOnImage(blobs, output_with_blobs);

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

    cv::Point2f center_left(center1.x, center1.y);
    cv::Point2f center_right(center2.x, center2.y);

    cv::Point2f pt_left_down(center_left.x + (height - center_left.y) / coef1, height);
    cv::Point2f pt_left_up(0, 0);

    cv::Point2f pt_right_down(center_right.x + (height - center_right.y) / coef2, height);
    cv::Point2f pt_right_up(0, 0);

    /// Resolution de l'equation point d'arrives

    pt_left_up.x = (coef1*pt_left_down.x - coef2*pt_right_down.x + pt_right_down.y - pt_left_down.y + coef2 * ecart) / (coef1 - coef2);
    pt_right_up.x = ecart + pt_left_up.x;
    pt_left_up.y = coef1 * (pt_left_up.x - pt_left_down.x) + pt_left_down.y;
    pt_right_up.y = coef2 * (pt_right_up.x - pt_right_down.x) + pt_left_down.y;   // should have pt_right_up.y == pt_left_up.y

    /**
      * Application de l'homographie
      */
    // The 4-points at the input image

    std::vector<cv::Point2f> origPoints;
    origPoints.push_back( pt_left_down );
    origPoints.push_back( pt_right_down );
    origPoints.push_back( pt_right_up );
    origPoints.push_back( pt_left_up );

    /////////////////

    // The 4-points correspondences in the destination image
    std::vector<cv::Point2f> dstPoints;
 /*   dstPoints.push_back( cv::Point2f(500, height) );
    dstPoints.push_back( cv::Point2f(width-300, height) );
    dstPoints.push_back( cv::Point2f(width-300, 0) );
    dstPoints.push_back( cv::Point2f(500, 0) );*/

    dstPoints.push_back( cv::Point2f((pt_left_down.x + pt_left_up.x)/2, height) );
    dstPoints.push_back( cv::Point2f((pt_right_down.x + pt_right_up.x)/2, height) );
    dstPoints.push_back( cv::Point2f((pt_right_down.x + pt_right_up.x)/2, 0) );
    dstPoints.push_back( cv::Point2f((pt_left_down.x + pt_left_up.x)/2, 0) );

    // IPM object
    IPM ipm( cv::Size(width, height), cv::Size(width, height), origPoints, dstPoints );

    ipm.applyHomography( inputImg, outputImg );

    ipm.drawPoints(origPoints, output_with_blobs, 0 );
    ipm.drawPoints(dstPoints, output_with_blobs, 1 );

    imshow("ligne de fuite pour l'homographie", output_with_blobs);

    return ipm;
}


/// Fonction de pré traitement de l'image
void firstProcess(cv::Mat inputFrame1, cv::Mat inputFrame2, cv::Mat &outputThresh, IPM &ipm)
{
    cv::Mat imgDifference;


    cv::cvtColor(inputFrame1, inputFrame1, CV_BGR2GRAY);
    cv::cvtColor(inputFrame2, inputFrame2, CV_BGR2GRAY);

    cv::GaussianBlur(inputFrame1, inputFrame1, cv::Size(5, 5), 0);
    cv::GaussianBlur(inputFrame2, inputFrame2, cv::Size(5, 5), 0);

    cv::absdiff(inputFrame1, inputFrame2, imgDifference);

    ipm.applyHomography( imgDifference, imgDifference );


    // Segmente la différence entre les 2 images (binarise la différence)
    cv::threshold(imgDifference, outputThresh, 30, 255.0, CV_THRESH_BINARY);


    // affiche l'image binarisé brut (cad pas de transformation morphologique apliqué)
    // cv::imshow("outputThresh before applying morphology", outputThresh);


    /**
     *  Application des transformations morphologiqes afin de unifié les différents blobs qui composent un élément
     *      -> la background substractor n'étant pas très précis, il segmente les objets mobiles en plusieurs blobs au lieu d'un seul
     */

    // Plus l'élément est grand plus on élargit la zone de blanc
    cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat structuringElement15x15 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));

    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 7));

    for (unsigned int i = 0; i < 2; i++) {
        cv::dilate(outputThresh, outputThresh, structuringElement);
        cv::dilate(outputThresh, outputThresh, structuringElement);
        cv::erode(outputThresh, outputThresh, structuringElement);
    }

    // affiche l'image binarisé après les opérations morphologique (1 block blanc = 1 objet en mouvement)
    cv::imshow("outputThresh after applying morphology", outputThresh);
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
bool checkIfBlobsCrossedTheLine(std::vector<Blob> &blobs, std::vector<Blob> &blobs_crossed_line, int &intHorizontalLinePosition, int &carCount) {
    bool blnAtLeastOneBlobCrossedTheLine = false;

    for (auto blob : blobs) {

        if (blob.blnStillBeingTracked == true && blob.centerPositions.size() >= 2) {
            int prevFrameIndex = (int)blob.centerPositions.size() - 2;
            int currFrameIndex = (int)blob.centerPositions.size() - 1;

            if (blob.centerPositions[prevFrameIndex].y < intHorizontalLinePosition && blob.centerPositions[currFrameIndex].y >= intHorizontalLinePosition) {  // haut vers bas
            //if (blob.centerPositions[prevFrameIndex].y >= intHorizontalLinePosition && blob.centerPositions[currFrameIndex].y < intHorizontalLinePosition) {  // bas vers haut
                carCount++;
                blnAtLeastOneBlobCrossedTheLine = true;
                blobs_crossed_line.push_back(blob);
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
