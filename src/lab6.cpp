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


//int main() {
    
//    cv::VideoCapture cap("data/video.mov");
//
//    if(cap.isOpened()) //check if we success
//    {
//        for(;;)
//        {
//            cv::Mat frame;
//            cap >> frame;
//            cap.read(frame);
////             check if we succeeded
//            if (frame.empty()) {
//                cerr << "ERROR! blank frame grabbed\n";
//                break;
//            }
////             show live and wait for a key with timeout long enough to show images
//            imshow("Live", frame);
//            if (cv::waitKey(5) >= 0)
//                break;
//        }
//    }
//
//    cv::Mat src = imread("data/objects/obj2.png");
//
//    cv::Mat first_frame;
//    cap>>first_frame;
//
////    Detect the keypoint using SIFT detector
//    Ptr<SIFT> detector = SIFT::create();
//    std::vector<KeyPoint> keypoints, keypoints2;
//    cv::Mat descriptors, descriptors2;
//    detector->detectAndCompute(first_frame,  noArray(), keypoints, descriptors);
//    detector->detectAndCompute(src,  noArray(), keypoints2, descriptors2);
//
//////    Draw keypoints
////    Mat img_keypoints, img_keypoints2;
////    drawKeypoints(first_frame, keypoints, img_keypoints);
////    drawKeypoints(src, keypoints2, img_keypoints2);
////
//////    Show detected (drawn) keypoints
////    imshow("SIFT Keypoints", img_keypoints );
////    imshow("SIFT Keypoints2", img_keypoints2);
////    waitKey();
//
//
////     Matching descriptor vectors with a brute force matcher
//    BFMatcher matcher(NORM_L2);
//    std::vector< DMatch > matches;
//    matcher.match(descriptors, descriptors2, matches);
//
////    Draw matches
////    Mat img_matches;
////    drawMatches( first_frame, keypoints, src, keypoints2, matches, img_matches );
//
////    Show detected matches
////    imshow("Matches", img_matches );
////    waitKey(0);
//
//
//    double max_dist = 0;
//    double min_dist = 100;
//
//    for( int i = 0; i < descriptors.rows; i++ )
//    { double dist = matches[i].distance;
//      if( dist < min_dist ) min_dist = dist;
//      if( dist > max_dist ) max_dist = dist;
//    }
//
//    printf("-- Max dist : %f \n", max_dist );
//    printf("-- Min dist : %f \n", min_dist );
//
////    Draw only good matches
//    std::vector< DMatch > good_matches;
//
//    for( int i = 0; i < descriptors.rows; i++ )
//    { if( matches[i].distance < 3*min_dist )
//       {
//           good_matches.push_back(matches[i]);
//       }
//    }
//
//    Mat img_matches;
//    drawMatches( src, keypoints, first_frame, keypoints2, good_matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
//
////     Localize the object
//    std::vector<Point2f> obj;
//    std::vector<Point2f> scene;
//
//    for( int i = 0; i < good_matches.size(); i++ )
//    {
//      //-- Get the keypoints from the good matches
//      obj.push_back( keypoints[ good_matches[i].queryIdx ].pt );
//      scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
//    }
//
//    cv::Mat H = findHomography( obj, scene, RANSAC);

//
//    return 0;
    
    cv::Mat H;
    vector<cv::Point2f> objectPoints, framePoints;
    vector<cv::DMatch> goodMatches;

    bool useSift = true;

    cv::Point calculateProjectedPoint(cv::Mat projectionMat, cv::Point objectPoint);

    int main() {
        cv::VideoCapture videoCapture("data/video.mov");
        vector<cv::Mat> frames;
        bool firstFrame = true;
        
        cv::Mat object = cv::imread("data/objects/obj1.png");
//        cv::Mat object = cv::imread("data/objects/obj2.png");
//        cv::Mat object = cv::imread("data/objects/obj3.png");
//        cv::Mat object = cv::imread("data/objects/obj4.png");
        
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
                    
                    cv::Mat showMatches;
                    
                    vector<int> mask;
//                    vector<cv::Point2f> objectPoints, framePoints;
                    for (int i = 0; i < matches.size(); ++i)
                    {
                        objectPoints.push_back(objKeypoints[matches[i].queryIdx].pt);
                        framePoints.push_back(frameKeypoints[matches[i].trainIdx].pt);
                    }
//                    cv::Mat H = cv::findHomography(objectPoints, framePoints, cv::RANSAC, 3, mask);
                    H = cv::findHomography(objectPoints, framePoints, cv::RANSAC, 3, mask);
                    cout << H << endl;
                    vector<cv::KeyPoint> objcInliers, frameInliers;
//                    vector<cv::DMatch> goodMatches;
                    for (int i = 0; i < mask.size(); ++i)
                    {
                        if (mask[i])
                        {
//                            objcInliers.push_back(objKeypoints[matches[i].queryIdx]);
//                            frameInliers.push_back(frameKeypoints[matches[i].trainIdx]);
                            goodMatches.push_back(matches[i]);
                        }
                    }
                    firstFrame = false;
                    }
                
                    if (cv::waitKey(5) == 0) {
                        break;
            }
//                 SE CHIUDIAMO IL CICLO QUI SODDISFIAMO QUELLO CHE DICE IL PROF RIGUARDO SMETTERLA DI ESTRARRE FEATURE? PUNTO 2 ALLA FINE
                    Mat img_matches;
                    drawMatches(object, objKeypoints, frame, frameKeypoints, goodMatches, img_matches);
                    
                    
//                    Get the corners from the frame ( the object to be "detected" )
                        std::vector<Point2f> obj_corners(4);
                        obj_corners[0] = Point2f(0, 0);
                        obj_corners[1] = Point2f( (float)object.cols, 0 );
                        obj_corners[2] = Point2f( (float)object.cols, (float)object.rows );
                        obj_corners[3] = Point2f( 0, (float)object.rows );
                        std::vector<Point2f> scene_corners(4);
                        perspectiveTransform( obj_corners, scene_corners, H);
                    
//                         Draw red lines between the corners of the frame object detected
                        line( img_matches, scene_corners[0] + Point2f((float)object.cols, 0),
                              scene_corners[1] + Point2f((float)object.cols, 0), Scalar(0, 0, 255), 4 );
                        line( img_matches, scene_corners[1] + Point2f((float)object.cols, 0),
                              scene_corners[2] + Point2f((float)object.cols, 0), Scalar( 0, 0, 255), 4 );
                        line( img_matches, scene_corners[2] + Point2f((float)object.cols, 0),
                              scene_corners[3] + Point2f((float)object.cols, 0), Scalar( 0, 0, 255), 4 );
                        line( img_matches, scene_corners[3] + Point2f((float)object.cols, 0),
                              scene_corners[0] + Point2f((float)object.cols, 0), Scalar( 0, 0, 255), 4 );
                
//                         Show detected matches
                        imshow("Object detection", img_matches );
                        waitKey(0);

                        cv::Mat frame_old;
                        videoCapture >> frame_old;
                
                //        Create a mask image for drawing purposes
                        Mat mask = Mat::zeros(frame_old.size(), frame_old.type());
                        std::vector<Point2f> p0 = framePoints;
                        while(true)
                        {
                            cv::Mat frameX;
                            videoCapture >> frameX;
                            if (frame.empty())
                                break;

//                         calculate optical flow
                            std::vector<Point2f>  p1;
//                            p0 = framePoints;
                            
                            vector<uchar> status;
                            vector<float> err;
                            vector<cv::Mat> pyramid;
                            TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
                            
                            buildOpticalFlowPyramid(frame_old, pyramid, Size(7, 7), 3, true, BORDER_REFLECT_101, BORDER_CONSTANT, true);
                            calcOpticalFlowPyrLK(pyramid, frameX, p0, p1, status, err, Size(7,7), 3, criteria);
                            
                            vector<Point2f> good_new;
                            for(uint i = 0; i < p0.size(); i++)
                            {
//                             Select good points
                                if(status[i] == 1)
                                {
                                    good_new.push_back(p1[i]);
//                             draw the keypoints
                                    circle(frameX, p1[i], 3, Scalar( 0, 0, 255), -1);
                                }
                            }
                            
                            Mat img;
                            add(frameX, mask, img);

                            imshow("Frame", img);
//                            waitKey(0);
                            int keyboard = waitKey(30);
                            if (keyboard == 'q' || keyboard == 27)
                                break;
                            
//                     Now update the previous frame and previous points
                            frame_old = frameX.clone();
                            p0 = good_new;
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

