#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include <unistd.h>

using namespace std;

int main() {
    cv::VideoCapture videoCapture("data/video.mov");
    if (videoCapture.isOpened()) {
        for (;;) {
            cv::Mat frame;
            videoCapture >> frame;
            cv::imshow("VIDEO", frame);
            break;
        }
    }
    cv::waitKey(0);
    return 0;
}
