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
    
void objectDescriptor(vector<Mat> objects, vector<vector<KeyPoint>> &keyPoints, vector<Mat> &descriptors) {
    cv::Ptr<cv::Feature2D> detector = SIFT::create();
    for (Mat object : objects) {
        vector<KeyPoint> objKeypoints;
        Mat objectDescriptors;
        detector->detect(object, objKeypoints);
        detector->compute(object, objKeypoints, objectDescriptors);
        keyPoints.push_back(objKeypoints);
        descriptors.push_back(objectDescriptors);
    }
}

void frameDescriptor(Mat frame, vector<KeyPoint> &keyPoints, Mat &descriptors) {
    cv::Ptr<cv::Feature2D> detector = SIFT::create();
    detector->detect(frame, keyPoints);
    detector->compute(frame, keyPoints, descriptors);
    
}

vector<vector<cv::DMatch>> matchObjectsAndFrame(vector<Mat> objectsdescriptors, Mat frame, Mat frameDescriptors) {
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2, true);
    vector<vector<cv::DMatch>> dmatches;
    for (Mat descriptors : objectsdescriptors) {
        vector<cv::DMatch> matches;
        matcher->match(descriptors, frameDescriptors, matches);
        dmatches.push_back(matches);
    }
    return dmatches;
}

vector<Mat> findPointsHomographies(vector<vector<KeyPoint>> objKeypoints, vector<KeyPoint> frameKeypoints, vector<vector<DMatch>> matches, vector<vector<int>> &maskes) {
    vector<Mat> homographies;
    for (int i = 0; i < matches.size(); ++i) {
        vector<DMatch> dMatch = matches[i];
        vector<Point2f> objectPoints, framePoints;
        vector<KeyPoint> objectKeypoint = objKeypoints[i];
        for (int j = 0; j < dMatch.size(); ++j) {
            objectPoints.push_back(objectKeypoint[dMatch[i].queryIdx].pt);
            framePoints.push_back(frameKeypoints[dMatch[i].trainIdx].pt);
        }
        vector<int> mask;
        Mat H = cv::findHomography(objectPoints, framePoints, cv::RANSAC, 3, mask);
        homographies.push_back(H);
        maskes.push_back(mask);
    }
    return homographies;
}

vector<vector<Point2f>> computeRectCorners(vector<Point2f> obj_corners, vector<Mat> Hs) {
    vector<vector<Point2f>> scene_corners;
    for (Mat H : Hs) {
        vector<Point2f> corners;
        perspectiveTransform( obj_corners, corners, H);
        scene_corners.push_back(corners);
    }
    return scene_corners;
}


int main() {
    cv::VideoCapture videoCapture("data/video.mov");
    vector<cv::Mat> frames;
    
    cv::Mat object = cv::imread("data/objects/obj1.png");
    cv::Mat object2 = cv::imread("data/objects/obj2.png");
    cv::Mat object3 = cv::imread("data/objects/obj3.png");
    cv::Mat object4 = cv::imread("data/objects/obj4.png");
    
    vector<Mat> objects = {object, object2, object3, object4};
    
    for (Mat object : objects) {
        if (object.empty()) {
            cout << "Unabe to read object file" << endl;
            return -1;
        }
    }
    
    vector<vector<cv::KeyPoint>> objKeypoints;
    vector<Mat> objDescriptors;
    vector<cv::KeyPoint> frameKeypoints;
    objectDescriptor(objects, objKeypoints, objDescriptors);
    
    
    if (videoCapture.isOpened()) {
        //Starting the computation on the first frame
        cv::Mat firstFrame;
        videoCapture >> firstFrame;
        //videoCapture.read(firstFrame);
        if (firstFrame.empty()) {
            cout << "Empty video!" << endl;
            return -1;;
        }
        cv::Mat frameDescriptors;
        frameDescriptor(firstFrame, frameKeypoints, frameDescriptors);
        vector<vector<cv::DMatch>> dmatches = matchObjectsAndFrame(objects, firstFrame, frameDescriptors);
        
        vector<vector<int>> maskes;
        vector<Mat> Hs = findPointsHomographies(objKeypoints, frameKeypoints, dmatches, maskes);
        
        vector<vector<DMatch>> goodMathces;
        for (int i = 0; i < maskes.size(); ++i) {
            vector<int> mask = maskes[i];
            vector<DMatch> matches = dmatches[i];
            vector<DMatch> good_matches;
            for (int j = 0; j < mask.size(); ++j) {
                if (mask[i]) {
                    framePoints.push_back(frameKeypoints[matches[j].trainIdx].pt);
                    good_matches.push_back(matches[j]);
                }
            }
            goodMathces.push_back(good_matches);
        }
        //==========Object recognition on frame is ended here==============
        
        //Get the corners from the frame ( the object to be "detected" )
        std::vector<Point2f> obj_corners(4);
        obj_corners[0] = Point2f(0, 0);
        obj_corners[1] = Point2f( (float)object.cols, 0 );
        obj_corners[2] = Point2f( (float)object.cols, (float)object.rows );
        obj_corners[3] = Point2f( 0, (float)object.rows );
        
        vector<vector<Point2f>> scene_corners = computeRectCorners(obj_corners, Hs);
    
        //Draw red lines between the corners of the frame object detected
        for (vector<Point2f> corners : scene_corners) {
            drawRect(firstFrame, corners);
        }
        imshow("PROVA", firstFrame);
        
        /*Mat img_matches;
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
        }*/
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

