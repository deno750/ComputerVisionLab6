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

cv::Point calculateProjectedPoint(cv::Mat projectionMat, cv::Point objectPoint);

int main() {
    cv::VideoCapture videoCapture("data/video.mov");
    vector<cv::Mat> frames;
    bool firstFrame = true;
    
    cv::Mat object = cv::imread("data/objects/obj2.png");
    
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
    
    
    if (videoCapture.isOpened()) {
        for (;;) {
            cv::Mat frame;
            videoCapture >> frame;
            videoCapture.read(frame);
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
                
                vector<int> mask;
                //vector<cv::Point2f> objectPoints, framePoints;
                for (int i = 0; i < matches.size(); ++i)
                {
                    objectPoints.push_back(objKeypoints[matches[i].queryIdx].pt);
                    framePoints.push_back(frameKeypoints[matches[i].trainIdx].pt);
                }
                //cv::Mat H = cv::findHomography(objectPoints, framePoints, cv::RANSAC, 3, mask);
                H = cv::findHomography(objectPoints, framePoints, cv::RANSAC, 3, mask);
                cout << H << endl;
                vector<cv::KeyPoint> objcInliers, frameInliers;
                //vector<cv::DMatch> goodMatches;
                framePoints.clear();
                for (int i = 0; i < mask.size(); ++i)
                {
                    if (mask[i])
                    {
                        objcInliers.push_back(objKeypoints[matches[i].queryIdx]);
                        frameInliers.push_back(frameKeypoints[matches[i].trainIdx]);
                        framePoints.push_back(frameKeypoints[matches[i].trainIdx].pt);
                        goodMatches.push_back(matches[i]);
                    }
                }
                firstFrame = false;
            }
            
            if (cv::waitKey(5) == 0) {
                break;
            }
            //SE CHIUDIAMO IL CICLO QUI SODDISFIAMO QUELLO CHE DICE IL PROF RIGUARDO SMETTERLA DI ESTRARRE FEATURE? PUNTO 2 ALLA FINE
            Mat img_matches;
            
            drawMatches(object, objKeypoints, frame, frameKeypoints, goodMatches, img_matches);
            
            
            //Get the corners from the frame ( the object to be "detected" )
            std::vector<Point2f> obj_corners(4);
            obj_corners[0] = Point2f(0, 0);
            obj_corners[1] = Point2f( (float)object.cols, 0 );
            obj_corners[2] = Point2f( (float)object.cols, (float)object.rows );
            obj_corners[3] = Point2f( 0, (float)object.rows );
            std::vector<Point2f> scene_corners(4);
            perspectiveTransform( obj_corners, scene_corners, H);
        
            //Draw red lines between the corners of the frame object detected
            line( img_matches, scene_corners[0] + Point2f((float)object.cols, 0),
                  scene_corners[1] + Point2f((float)object.cols, 0), Scalar(0, 0, 255), 4 );
            line( img_matches, scene_corners[1] + Point2f((float)object.cols, 0),
                  scene_corners[2] + Point2f((float)object.cols, 0), Scalar( 0, 0, 255), 4 );
            line( img_matches, scene_corners[2] + Point2f((float)object.cols, 0),
                  scene_corners[3] + Point2f((float)object.cols, 0), Scalar( 0, 0, 255), 4 );
            line( img_matches, scene_corners[3] + Point2f((float)object.cols, 0),
                  scene_corners[0] + Point2f((float)object.cols, 0), Scalar( 0, 0, 255), 4 );
    
            //Show detected matches
            imshow("Object detection", img_matches );
            waitKey(0);

            cv::Mat frame_old;
            videoCapture >> frame_old;
    
            //Create a mask image for drawing purposes
            Mat mask = Mat::zeros(frame_old.size(), frame_old.type());
            std::vector<Point2f> p0 = framePoints;
            cv::Mat testImage = frame.clone();
            for (int i = 0; i < framePoints.size(); ++i) {
                circle(testImage, framePoints[i], 3, Scalar( 0, 0, 255), -1);
            }
            imshow("TEST", testImage);
            std::vector<Point2f> oldLinePoints = scene_corners;
            while(true) {
                cv::Mat frameX;
                videoCapture >> frameX;
                if (frame.empty())
                    break;

                //calculate optical flow
                std::vector<Point2f>  p1;
                //p0 = framePoints;
                
                vector<uchar> status;
                vector<float> err;
                vector<cv::Mat> pyramid;
                TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
                
                buildOpticalFlowPyramid(frame_old, pyramid, Size(7, 7), 3, true, BORDER_REFLECT_101, BORDER_CONSTANT, true);
                calcOpticalFlowPyrLK(pyramid, frameX, p0, p1, status, err, Size(7,7), 3, criteria);
                
                
                vector<Point2f> good_old;
                vector<Point2f> good_new;
                for(uint i = 0; i < p0.size(); i++)
                {
                    double distance = sqrt(pow(p0[i].x - p1[i].x, 2) + pow(p0[i].y - p1[i].y, 2));
                    //Select good points
                    if(status[i] == 1 && distance <= 10)
                    {
                        good_old.push_back(p0[i]);
                        good_new.push_back(p1[i]);
                        //draw the keypoints
                        circle(frameX, p1[i], 3, Scalar( 0, 0, 255), -1);
                    }
                }
                cv::Mat H = cv::findHomography(good_old, good_new);
                std::vector<Point2f> newLinePoints;
                perspectiveTransform( oldLinePoints, newLinePoints, H);
                oldLinePoints = newLinePoints;
                //cout << H << endl;
                
                //Draw red lines between the corners of the frame object detected
                line(frameX, newLinePoints[0], newLinePoints[1], Scalar(0, 0, 255));
                line(frameX, newLinePoints[1], newLinePoints[2], Scalar(0, 0, 255));
                line(frameX, newLinePoints[2], newLinePoints[3], Scalar(0, 0, 255));
                line(frameX, newLinePoints[3], newLinePoints[0], Scalar(0, 0, 255));
                //line( frameX, newLinePoints[0], 0),newLinePoints[1], Scalar(0, 0, 255), 4 );
                /*line( frameX, scene_corners[1] + Point2f((float)object.cols, 0),
                      scene_corners[2] + Point2f((float)object.cols, 0), Scalar( 0, 0, 255), 4 );
                line( frameX, scene_corners[2] + Point2f((float)object.cols, 0),
                      scene_corners[3] + Point2f((float)object.cols, 0), Scalar( 0, 0, 255), 4 );
                line( frameX, scene_corners[3] + Point2f((float)object.cols, 0),
                      scene_corners[0] + Point2f((float)object.cols, 0), Scalar( 0, 0, 255), 4 );*/
                
                Mat img;
                add(frameX, mask, img);

                imshow("Frame", img);
                //waitKey(0);
                int keyboard = waitKey(30);
                if (keyboard == 'q' || keyboard == 27)
                    break;
                
                //Now update the previous frame and previous points
                frame_old = frameX.clone();
                p0 = good_new;
            }
        }
    }
    cv::waitKey(0);
    return 0;
}

/*void drawRect(cv::Mat image, std::vector<Point2f> scene_corners, std::vector<Point2f> obj_corners) {
    line( image, scene_corners[0] + Point2f((float)object.cols, 0),
          scene_corners[1] + Point2f((float)object.cols, 0), Scalar(0, 0, 255), 4 );
    line( image, scene_corners[1] + Point2f((float)object.cols, 0),
          scene_corners[2] + Point2f((float)object.cols, 0), Scalar( 0, 0, 255), 4 );
    line( image, scene_corners[2] + Point2f((float)object.cols, 0),
          scene_corners[3] + Point2f((float)object.cols, 0), Scalar( 0, 0, 255), 4 );
    line( image, scene_corners[3] + Point2f((float)object.cols, 0),
          scene_corners[0] + Point2f((float)object.cols, 0), Scalar( 0, 0, 255), 4 );
}*/

