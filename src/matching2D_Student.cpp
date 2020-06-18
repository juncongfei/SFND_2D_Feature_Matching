#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType;
        if(descriptorType.compare("DES_BINARY") == 0)
        {
            normType = cv::NORM_HAMMING;
        }
        else if(descriptorType.compare("DES_HOG") == 0)
        {
            normType = cv::NORM_L2;
        }
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if(descSource.type() != CV_32F)
        {
            descSource.convertTo(descSource, CV_32F);
        }
        if(descRef.type() != CV_32F)
        {
            descRef.convertTo(descRef, CV_32F);
        }

        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, 2); // k=2

        const float ratio_thresh = 0.8f;
        for (size_t i = 0; i < knn_matches.size(); i++)
        {
            if(knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
            {
                matches.push_back(knn_matches[i][0]);
            }
        }
        cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, double &descriptorTime)
{
    // select appropriate descriptor from BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if(descriptorType.compare("BRIEF") == 0)
    {

        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if(descriptorType.compare("ORB") == 0)
    {

        extractor = cv::ORB::create();
    }
    else if(descriptorType.compare("FREAK") == 0)
    {

        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if(descriptorType.compare("AKAZE") == 0)
    {

        extractor = cv::AKAZE::create();
    }
    else if(descriptorType.compare("SIFT") == 0)
    {

        extractor = cv::xfeatures2d::SIFT::create();
    }

    // perform feature description
    descriptorTime = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    descriptorTime = ((double)cv::getTickCount() - descriptorTime) / cv::getTickFrequency();
    descriptorTime = 1000 * descriptorTime /1.0;
    cout << descriptorType << " descriptor extraction in " << descriptorTime << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis, double &detectorTime, int &detectedKpts)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    detectorTime = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    detectorTime = ((double)cv::getTickCount() - detectorTime) / cv::getTickFrequency();
    detectorTime = 1000 * detectorTime /1.0;
    detectedKpts = keypoints.size();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << detectorTime << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the traditional Harris detector
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis, double &detectorTime, int &detectedKpts)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize  x  blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    detectorTime = (double)cv::getTickCount();
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Apply NMS
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (size_t r = 0; r < dst_norm.rows; r++)
    {
        for (size_t c = 0; c < dst_norm.cols; c++)
        {
            int response = (int)dst_norm.at<float>(r, c);
            if (response > minResponse)
            {
                cv::KeyPoint currKpt;
                currKpt.pt = cv::Point2f(c, r);
                currKpt.size = 2 * apertureSize;
                currKpt.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool b_overlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double overlap = cv::KeyPoint::overlap(currKpt, *it);
                    if (overlap > maxOverlap)
                    {
                        b_overlap = true;
                        if (currKpt.response > it->response)
                        {
                            *it = currKpt;
                            break;
                        }
                    }
                }
                if (!b_overlap)
                {                                     // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(currKpt); // store new keypoint in dynamic list
                }
            }
        }
    }
  
    detectorTime = ((double)cv::getTickCount() - detectorTime) / cv::getTickFrequency();
    detectorTime = 1000 * detectorTime /1.0;
    detectedKpts = keypoints.size();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << detectorTime << " ms" << endl;


    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

//FAST, BRISK, ORB, AKAZE, SIFT
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis, double &detectorTime, int &detectedKpts)
{
    string windowName;
    if (detectorType.compare("FAST") == 0)
    {
        int threshold = 30;
        bool bNMS = true;
        cv::FastFeatureDetector::DetectorType type = cv::FastFeatureDetector::TYPE_9_16; // TYPE_9_16, TYPE_7_12, TYPE_5_8
        cv::Ptr<cv::FeatureDetector> detector = cv::FastFeatureDetector::create(threshold, bNMS, type);
        detectorTime = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        detectorTime = ((double)cv::getTickCount() - detectorTime) / cv::getTickFrequency();
        detectorTime = 1000 * detectorTime /1.0;
        detectedKpts = keypoints.size();
        cout << "FAST detection with n= " << keypoints.size() << " keypoints in " << detectorTime << " ms" << endl;
        windowName = "FAST  Detector Results";
    }  
    else if (detectorType.compare("BRISK") == 0)
    {
        cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
        detectorTime = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        detectorTime = ((double)cv::getTickCount() - detectorTime) / cv::getTickFrequency();
        detectorTime = 1000 * detectorTime /1.0;
        detectedKpts = keypoints.size();
        cout << "BRISK detection with n= " << keypoints.size() << " keypoints in " << detectorTime << " ms" << endl;
        windowName = "BRISK  Detector Results";
    }
    else if (detectorType.compare("ORB") == 0)
    {
        cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
        detectorTime = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        detectorTime = ((double)cv::getTickCount() - detectorTime) / cv::getTickFrequency();
        detectorTime = 1000 * detectorTime /1.0;
        detectedKpts = keypoints.size();
        cout << "ORB detection with n= " << keypoints.size() << " keypoints in " << detectorTime << " ms" << endl;
        windowName = "ORB  Detector Results";
    }  
    else if (detectorType.compare("AKAZE") == 0)
    {
        cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
        detectorTime = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        detectorTime = ((double)cv::getTickCount() - detectorTime) / cv::getTickFrequency();
        detectorTime = 1000 * detectorTime /1.0;
        detectedKpts = keypoints.size();
        cout << "AKAZE detection with n= " << keypoints.size() << " keypoints in " << detectorTime << " ms" << endl;
        windowName = "AKAZE  Detector Results";
    }  
    else if (detectorType.compare("SIFT") == 0)
    {
        cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create();
        detectorTime = (double)cv::getTickCount();
        detector->detect(img, keypoints);
        detectorTime = ((double)cv::getTickCount() - detectorTime) / cv::getTickFrequency();
        detectorTime = 1000 * detectorTime /1.0;
        detectedKpts = keypoints.size();
        cout << "SIFT detection with n= " << keypoints.size() << " keypoints in " << detectorTime << " ms" << endl;
         windowName = "SIFT  Detector Results";
    }  
    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
