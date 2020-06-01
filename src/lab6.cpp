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

void drawRect(Mat image, std::vector<Point2f> corners, Scalar color = Scalar(0, 0, 255), int lineWidth = 4) {
    line(image, corners[0], corners[1], color, lineWidth);
    line(image, corners[1], corners[2], color, lineWidth);
    line(image, corners[2], corners[3], color, lineWidth);
    line(image, corners[3], corners[0], color, lineWidth);
}
    
void objectDescriptor(vector<Mat> objects, vector<vector<KeyPoint>> &keyPoints, vector<Mat> &descriptors) {
    Ptr<Feature2D> detector = SIFT::create();
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
    Ptr<Feature2D> detector = SIFT::create();
    detector->detect(frame, keyPoints);
    detector->compute(frame, keyPoints, descriptors);
    
}

vector<vector<DMatch>> matchObjectsAndFrame(vector<Mat> objectsdescriptors, Mat frameDescriptors) {
    Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2, true);
    vector<vector<DMatch>> dmatches;
    for (Mat descriptors : objectsdescriptors) {
        vector<DMatch> matches;
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
            objectPoints.push_back(objectKeypoint[dMatch[j].queryIdx].pt);
            framePoints.push_back(frameKeypoints[dMatch[j].trainIdx].pt);
        }
        vector<int> mask;
        Mat H = findHomography(objectPoints, framePoints, RANSAC, 3, mask);
        homographies.push_back(H);
        maskes.push_back(mask);
    }
    return homographies;
}

vector<vector<Point2f>> computeRectCorners(vector<vector<Point2f>> obj_corners, vector<Mat> Hs) {
    vector<vector<Point2f>> scene_corners;
    for (int i = 0; i< obj_corners.size(); ++i) {
        Mat H = Hs[i];
        vector<Point2f> corners;
        vector<Point2f> object_corners = obj_corners[i];
        perspectiveTransform( object_corners, corners, H);
        scene_corners.push_back(corners);
    }
    return scene_corners;
}


int main() {
    VideoCapture videoCapture("data/video.mov");
    vector<Mat> frames;
    
    Mat object = imread("data/objects/obj1.png");
    Mat object2 = imread("data/objects/obj2.png");
    Mat object3 = imread("data/objects/obj3.png");
    Mat object4 = imread("data/objects/obj4.png");
    
    vector<Mat> objects = {object, object2, object3, object4};
    vector<Scalar> colors = {Scalar(0, 0, 255), Scalar(255, 0, 0), Scalar(255, 255, 0), Scalar(255, 255, 255) };
    
    for (Mat object : objects) {
        if (object.empty()) {
            cout << "Unabe to read object file" << endl;
            return -1;
        }
    }
    
    vector<vector<KeyPoint>> objKeypoints;
    vector<Mat> objDescriptors;
    vector<KeyPoint> frameKeypoints;
    objectDescriptor(objects, objKeypoints, objDescriptors);
    
    
    if (videoCapture.isOpened()) {
        //Starting the computation on the first frame
        Mat firstFrame;
        videoCapture >> firstFrame;
        //videoCapture.read(firstFrame);
        if (firstFrame.empty()) {
            cout << "Empty video!" << endl;
            return -1;;
        }
        resize(firstFrame, firstFrame, Size(640, 360), 0, 0, INTER_CUBIC);
        Mat frameDescriptors;
        frameDescriptor(firstFrame, frameKeypoints, frameDescriptors);
        vector<vector<DMatch>> dmatches = matchObjectsAndFrame(objDescriptors, frameDescriptors);
        
        vector<vector<int>> maskes;
        vector<Mat> Hs = findPointsHomographies(objKeypoints, frameKeypoints, dmatches, maskes);
        
        vector<vector<Point2f>> framePoints;
        vector<vector<DMatch>> goodMathces;
        for (int i = 0; i < objects.size(); ++i) {
            vector<int> mask = maskes[i];
            vector<DMatch> matches = dmatches[i];
            vector<DMatch> good_matches;
            vector<Point2f> frame_points;
            for (int j = 0; j < mask.size(); ++j) {
                if (mask[j]) {
                    frame_points.push_back(frameKeypoints[matches[j].trainIdx].pt);
                    good_matches.push_back(matches[j]);
                }
            }
            framePoints.push_back(frame_points);
            goodMathces.push_back(good_matches);
        }
        //==========Object recognition on first frame is ended here==============
        
        //Get the corners from the frame ( the object to be "detected" )
        
        vector<vector<Point2f>> objectCorners;
        for (Mat object : objects) {
            std::vector<Point2f> obj_corners = {
                Point2f(0, 0),
                Point2f((float) object.cols, 0),
                Point2f((float) object.cols, (float) object.rows),
                Point2f(0, (float) object.rows),
            };
            objectCorners.push_back(obj_corners);
        }

        
        vector<vector<Point2f>> scene_corners = computeRectCorners(objectCorners, Hs);
    
        //Draw the keypoints of the obejcts
        Mat firsFrameDraw = firstFrame.clone();
        for (int i = 0; i < scene_corners.size(); ++i) {
            
            vector<Point2f> points = framePoints[i];
            for (int j = 0; j < points.size(); ++j) {
                Point2f p = points[j];
                circle(firsFrameDraw, p, 3, colors[i], -1);
            }
            
        }
        
        imshow("Keypoints", firsFrameDraw);
        
        for (int i = 0; i < objects.size(); ++i) {
            Mat img_matches;
            drawMatches(objects[i], objKeypoints[i], firstFrame, frameKeypoints, goodMathces[i], img_matches);
            imshow("Matches with object" + to_string(i + 1), img_matches);
        }

        Mat frame_old;
        videoCapture >> frame_old;
        resize(frame_old, frame_old, Size(640, 360), 0, 0, INTER_CUBIC);
        
        vector<vector<Point2f>> oldRectPoints = scene_corners; //Rect points of the old frame
        
        while(true) {
            Mat frame;
            videoCapture >> frame;
            if (frame.empty())
                break;
            resize(frame, frame, Size(640, 360), 0, 0, INTER_CUBIC);
            Mat drawFrame = frame.clone();
            vector<vector<Point2f>> newPoints;
            vector<vector<Point2f>> newRectPoints;
            vector<vector<Point2f>> goodNewPoints;
            
            //Frame pyramid calculation
            vector<Mat> pyramid;
            buildOpticalFlowPyramid(frame_old, pyramid, Size(7, 7), 3, true, BORDER_REFLECT_101, BORDER_CONSTANT, true);
            
            for (int i = 0; i < objects.size(); ++i) {
                Scalar color = colors[i];
                std::vector<Point2f> p0 = framePoints[i];
                //calculate optical flow
                std::vector<Point2f>  p1;
                
                
                vector<uchar> status;
                vector<float> err;
                TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
                
                calcOpticalFlowPyrLK(pyramid, frame, p0, p1, status, err, Size(7,7), 3, criteria);
                
                newPoints.push_back(p1);
                
                //Stores the good points of the new and current frames that matches well
                vector<Point2f> good_old;
                vector<Point2f> good_new;
                for(uint j = 0; j < p0.size(); j++) {
                    
                    double distance = (pow(p0[j].x - p1[j].x, 2) + pow(p0[j].y - p1[j].y, 2)); //Distance that is used to seek the stable points. Sqrt is not computed just to maintain the computation of the distance faster.
                    //Select good points
                    if(status[j] == 1 && distance <= 25) { //distance <= 5 pixels squared
                        
                        good_old.push_back(p0[j]);
                        good_new.push_back(p1[j]);
                        
                        //draw the keypoints
                        circle(drawFrame, p1[j], 3, color, -1);
                    }
                }
                goodNewPoints.push_back(good_new);
                Mat H = findHomography(good_old, good_new);
                
                vector<Point2f> new_rect_points; //Points of the rect of the new frame
                vector<Point2f> oldPoints = oldRectPoints[i];
                perspectiveTransform(oldPoints, new_rect_points, H);
                newRectPoints.push_back(new_rect_points);
                
                //Draw rects
                drawRect(drawFrame, new_rect_points, color);
            }

            imshow("Frame", drawFrame);
            int keyboard = waitKey(30);
            if (keyboard == 'q' || keyboard == 27)
                break;
            
            //Now update the previous frame and previous points
            frame_old = frame.clone();
            framePoints = goodNewPoints;
            oldRectPoints = newRectPoints;
        }
    }
    waitKey(0);
    return 0;
}

