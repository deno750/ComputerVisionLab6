#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/ccalib.hpp>
#include <opencv2/stitching.hpp>

using namespace std;
using namespace cv::xfeatures2d;

bool useSift = true;

cv::Point calculateProjectedPoint(cv::Mat projectionMat, cv::Point objectPoint);

int main() {
    cv::VideoCapture videoCapture("data/video.mov");
    vector<cv::Mat> frames;
    bool firstFrame = true;
    cv::Mat object = cv::imread("data/objects/obj4.png");
    if (object.empty()) {
        return -1;
    }
    vector<cv::KeyPoint> objKeypoints;
    vector<cv::KeyPoint> frameKeypoints;
    cv::Ptr<cv::Feature2D> detector;
    cv::Ptr<cv::BFMatcher> matcher;
    if (useSift) {
        detector = SIFT::create();
        matcher = cv::BFMatcher::create(cv::NORM_L2, true);
    } else {
        detector = cv::ORB::create();
        matcher = cv::BFMatcher::create(cv::NORM_HAMMING, true);
    }
    cv::Mat objectDescriptors;
    detector->detect(object, objKeypoints);
    detector->compute(object, objKeypoints, objectDescriptors);
    
    cv::Point point1 = cv::Point(150, 300);
    cv::Point point2 = cv::Point(560, 300);
    
    
    if (videoCapture.isOpened()) {
        for (;;) {
            cv::Mat frame;
            videoCapture >> frame;
            if (frame.empty()) {
                cout << "Finished video!" << endl;
                break;
            }
            if (firstFrame) {
                
                cv::Mat frameDescriptors;
                detector->detect(frame, frameKeypoints);
                detector->compute(frame, frameKeypoints, frameDescriptors);
                
                vector<cv::DMatch> matches;
                matcher->match(objectDescriptors, frameDescriptors, matches);
                
                cv::Mat showMatches;
                
                vector<int> mask;
                vector<cv::Point2f> objectPoints, framePoints;
                for (int i = 0; i < matches.size(); ++i) {
                    objectPoints.push_back(objKeypoints[matches[i].queryIdx].pt);
                    framePoints.push_back(frameKeypoints[matches[i].trainIdx].pt);
                }
                cv::Mat H = cv::findHomography(objectPoints, framePoints, cv::RANSAC, 3, mask);
                cout << H << endl;
                //vector<cv::KeyPoint> objcInliers, frameInliers;
                vector<cv::DMatch> goodMatches;
                for (int i = 0; i < mask.size(); ++i) {
                    if (mask[i]) {
                        //objcInliers.push_back(objKeypoints[matches[i].queryIdx]);
                        //frameInliers.push_back(frameKeypoints[matches[i].trainIdx]);
                        goodMatches.push_back(matches[i]);
                    }
                }
                
                cv::Point projectedPoint1 = calculateProjectedPoint(H, point1);
                cv::Point projectedPoint2 = calculateProjectedPoint(H, point2);
                
                
                cv::line(frame, projectedPoint1, projectedPoint2, cv::Scalar(0, 0, 255));
                cv::line(object, point1, point2, cv::Scalar(0, 0, 255));
                
                cv::imshow("Object", object);
                cv::imshow("Frame", frame);
                
                firstFrame = false;
            }
            
            //cv::imshow("VIDEO", frame);
            
            if (cv::waitKey(5) == 0) {
                break;
            }
        }
    }
    cv::waitKey(0);
    return 0;
}

cv::Point calculateProjectedPoint(cv::Mat projectionMat, cv::Point objectPoint) {
    cv::Mat newMat(3, 1, projectionMat.type());
    newMat.at<double>(0, 0) = objectPoint.x;
    newMat.at<double>(0, 1) = objectPoint.y;
    newMat.at<double>(0, 2) = 1;
    cv::Mat resMat = projectionMat * newMat;
    double k = resMat.at<double>(0, 2);
    int x = static_cast<int>(resMat.at<double>(0, 0) / k);
    int y = static_cast<int>(resMat.at<double>(0, 1) / k);
    return cv::Point(x, y);
}
