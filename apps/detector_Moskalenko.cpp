#include <iostream>
#include <string>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;

const char* params =
     "{ h | help     | false | print usage                                   }"
     "{   | detector |       | XML file with a cascade detector              }"
     "{   | image    |       | image to detect objects on                    }"
     "{   | video    |       | video file to detect on                       }"
     "{   | camera   | false | whether to detect on video stream from camera }";


void drawDetections(const vector<Rect>& detections,
                    const Scalar& color,
                    Mat& image)
{
    for (size_t i = 0; i < detections.size(); ++i)
    {
        rectangle(image, detections[i], color, 2);
    }
}

const Scalar white(255,255, 255);
const Scalar red(0, 0, 255);
const Scalar green(0, 255, 0);
const Scalar blue(255, 0, 0);
const Scalar colors[] = {red, green, blue};

int main(int argc, char** argv)
{
    // Parse command line arguments.
    CommandLineParser parser(argc, argv, params);
    // If help flag is present, print help message and exit.
    if (parser.get<bool>("help"))
    {
        parser.printParams();
        return 0;
    }

	string detector_unn_str = "./unn_logo_cascade/cascade.xml";
	string detector_itseez_str = "./itseez_logo_w_caption_cascade/cascade.xml";
	string detector_opencv_str = "./opencv_logo_cascade/cascade.xml";

    string detector_file = parser.get<string>("detector");
    CV_Assert(!detector_file.empty());
    string image_file = parser.get<string>("image");
    string video_file = parser.get<string>("video");
    bool use_camera = parser.get<bool>("camera");

	cv::CascadeClassifier detector;
	cv::CascadeClassifier detector_unn;
	cv::CascadeClassifier detector_itseez;
	cv::CascadeClassifier detector_opencv;
	detector.load(detector_file);
	detector_unn.load(detector_unn_str);
	detector_itseez.load(detector_itseez_str);
	detector_opencv.load(detector_opencv_str);

    if (!image_file.empty())
    {
		Mat image;
		image = cv::imread(image_file);
		vector<Rect> result;
		detector.detectMultiScale(image,result);
		drawDetections(result,red,image);
		imshow("Image", image);
		waitKey(0);   
    }
    else if (!video_file.empty())
    {
		namedWindow("Video",1);
		cv::VideoCapture cap(video_file);
		CV_Assert(cap.isOpened());
		Mat image;
		cap >> image;
		while (cap.isOpened())
		{
			cap >> image;
			vector<Rect> result;
			detector.detectMultiScale(image,result);
			drawDetections(result,red,image);
			imshow("Video", image);
			if(waitKey(30) >= 0) break;
		};
    }
    else if (use_camera)
    {
		namedWindow("Video",1);
        cv::VideoCapture cap = cv::VideoCapture(0);
		cap.open(0);
		Mat image;
		cap >> image;
		for(;;)
		{
			cap >> image;
			vector<Rect> result;
			vector<Rect> resultUnn;
			vector<Rect> resultOpencv;
			vector<Rect> resultItseez;
			detector.detectMultiScale(image,result);
			detector_unn.detectMultiScale(image,resultUnn);
			detector_opencv.detectMultiScale(image,resultOpencv);
			detector_itseez.detectMultiScale(image,resultItseez);
			drawDetections(resultUnn,red,image);
			drawDetections(resultOpencv,blue,image);
			drawDetections(resultItseez,green,image);
			drawDetections(result,white,image);
			imshow("Video", image);
			if(waitKey(30) >= 0) break;
		};
    }
    else
    {
        cout << "Declare a source of images to detect on." << endl;
    }

    return 0;
}



