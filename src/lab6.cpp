#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/ccalib.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

    
cv::Mat H;
vector<cv::Point2f> objectPoints, framePoints;
vector<cv::DMatch> goodMatches;

bool useSift = true;

void drawRect(cv::Mat image, std::vector<Point2f> corners, Scalar color = Scalar(0, 0, 255), int lineWidth = 4);

int main() {
    cv::VideoCapture videoCapture("data/video.mov");
    vector<cv::Mat> frames;
    
    cv::Mat object = cv::imread("data/objects/obj2.png");
    
    if (object.empty()) {
        cout << "Unabe to read object file" << endl;
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
    
    
    if (videoCapture.isOpened()) {
        //Starting the computation on the first frame
        cv::Mat firstFrame;
        videoCapture >> firstFrame;
        //videoCapture.read(firstFrame);
        if (firstFrame.empty()) {
            cout << "Empty video!" << endl;
            return -1;;
        }
        
        //Compute keypoints of the frame
        cv::Mat frameDescriptors;
        detector->detect(firstFrame, frameKeypoints);
        detector->compute(firstFrame, frameKeypoints, frameDescriptors);
        
        //Compute the matched between object and frame
        vector<cv::DMatch> matches;
        matcher->match(objectDescriptors, frameDescriptors, matches);
        
        for (int i = 0; i < matches.size(); ++i)
        {
            objectPoints.push_back(objKeypoints[matches[i].queryIdx].pt);
            framePoints.push_back(frameKeypoints[matches[i].trainIdx].pt);
        }
        
        vector<int> mask;
        H = cv::findHomography(objectPoints, framePoints, cv::RANSAC, 3, mask);
        
        //The old frame points are not needed anymore. Now we store the points which are inliers
        framePoints.clear();
        for (int i = 0; i < mask.size(); ++i) {
            if (mask[i]) {
                framePoints.push_back(frameKeypoints[matches[i].trainIdx].pt);
                goodMatches.push_back(matches[i]);
            }
        }
        //==========Object recognition on frame is ended here==============
        
        //Get the corners from the frame ( the object to be "detected" )
        std::vector<Point2f> obj_corners(4);
        obj_corners[0] = Point2f(0, 0);
        obj_corners[1] = Point2f( (float)object.cols, 0 );
        obj_corners[2] = Point2f( (float)object.cols, (float)object.rows );
        obj_corners[3] = Point2f( 0, (float)object.rows );
        std::vector<Point2f> scene_corners(4);
        perspectiveTransform( obj_corners, scene_corners, H);
    
        //Draw red lines between the corners of the frame object detected
        drawRect(firstFrame, scene_corners);
        
        Mat img_matches;
        drawMatches(object, objKeypoints, firstFrame, frameKeypoints, goodMatches, img_matches);

        //Show detected matches
        imshow("Object detection", img_matches );

        cv::Mat frame_old;
        videoCapture >> frame_old;

        //Create a mask image for drawing purposes
        //Mat mask = Mat::zeros(frame_old.size(), frame_old.type());
        
        std::vector<Point2f> p0 = framePoints;
        std::vector<Point2f> oldRectPoints = scene_corners; //Rect points of the old frame
        
        while(true) {
            cv::Mat frame;
            videoCapture >> frame;
            if (frame.empty())
                break;

            //calculate optical flow
            std::vector<Point2f>  p1;
            
            vector<uchar> status;
            vector<float> err;
            vector<cv::Mat> pyramid;
            TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
            
            buildOpticalFlowPyramid(frame_old, pyramid, Size(7, 7), 3, true, BORDER_REFLECT_101, BORDER_CONSTANT, true);
            calcOpticalFlowPyrLK(pyramid, frame, p0, p1, status, err, Size(7,7), 3, criteria);
            
            
            //Stores the good points of the new and current frames that matches well
            vector<Point2f> good_old;
            vector<Point2f> good_new;
            for(uint i = 0; i < p0.size(); i++) {
                
                double distance = sqrt(pow(p0[i].x - p1[i].x, 2) + pow(p0[i].y - p1[i].y, 2)); //Distance that is used to seek the stable points
                //Select good points
                if(status[i] == 1 && distance <= 10) {
                    
                    good_old.push_back(p0[i]);
                    good_new.push_back(p1[i]);
                    
                    //draw the keypoints
                    circle(frame, p1[i], 3, Scalar( 0, 0, 255), -1);
                }
            }
            
            cv::Mat H = cv::findHomography(good_old, good_new); //calculate homography matrix between old frame and new frame
            std::vector<Point2f> newRectPoints; //Points of the rect of the new frame
            perspectiveTransform( oldRectPoints, newRectPoints, H);
            
            //Draw red lines
            drawRect(frame, newRectPoints);

            imshow("Frame", frame);
            int keyboard = waitKey(30);
            if (keyboard == 'q' || keyboard == 27)
                break;
            
            //Now update the previous frame and previous points
            frame_old = frame.clone();
            p0 = good_new;
            oldRectPoints = newRectPoints;
        }
    }
    cv::waitKey(0);
    return 0;
}

void drawRect(cv::Mat image, std::vector<Point2f> corners, Scalar color, int lineWidth) {
    line(image, corners[0], corners[1], color, lineWidth);
    line(image, corners[1], corners[2], color, lineWidth);
    line(image, corners[2], corners[3], color, lineWidth);
    line(image, corners[3], corners[0], color, lineWidth);
}

