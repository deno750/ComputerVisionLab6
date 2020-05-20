#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;
using namespace cv::xfeatures2d;


int main() {
    
    cv::VideoCapture cap("../../src/data/video.mov");
    if(cap.isOpened()) //check if we success
    {
        for(;;)
        {
            cv::Mat frame;
            cap >> frame;
            cap.read(frame);
            // check if we succeeded
            if (frame.empty()) {
                cerr << "ERROR! blank frame grabbed\n";
                break;
            }
            // show live and wait for a key with timeout long enough to show images
            imshow("Live", frame);
            if (cv::waitKey(5) >= 0)
                break;
        }
    }
    
    Ptr<SIFT> detector = SIFT::create();
    std::vector<KeyPoint> keypoints;
    detector->detect( , keypoints );
    return 0;
}
