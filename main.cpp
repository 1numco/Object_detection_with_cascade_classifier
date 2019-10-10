#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
using namespace std;
using namespace cv;
void detectAndDisplay(Mat frame);
CascadeClassifier cascade;
int main(int argc, const char** argv)
{
	CommandLineParser parser(argc, argv,
		"{help h||}"
		"{cascade||Path to face cascade.}"
		"{camera|0|Camera device number.}");
	parser.about("\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (palms) in a video stream.\n"
		"You can use Haar or LBP features.\n\n");
	parser.printMessage();
	String cascade_name = parser.get<String>("cascade");
	//-- 1. Load the cascades
	if (!cascade.load(cascade_name))
	{
		cout << "--(!)Error loading face cascade\n";
		return -1;
	};

	int camera_device = parser.get<int>("camera");
	VideoCapture capture;
	//-- 2. Read the video stream
	capture.open(camera_device);
	if (!capture.isOpened())
	{
		cout << "--(!)Error opening video capture\n";
		return -1;
	}
	Mat frame;
	while (capture.read(frame))
	{
		if (frame.empty())
		{
			cout << "--(!) No captured frame -- Break!\n";
			break;
		}
		//-- 3. Apply the classifier to the frame
		detectAndDisplay(frame);
		if (waitKey(10) == 27)
		{
			break; // escape
		}
	}
	return 0;
}
void detectAndDisplay(Mat frame)
{
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);
	//-- Detect faces
	std::vector<Rect> objects;
	cascade.detectMultiScale(frame_gray, objects);
	for (size_t i = 0; i < objects.size(); i++)
	{
		Point center(objects[i].x + objects[i].width / 2, objects[i].y + objects[i].height / 2);
		ellipse(frame, center, Size(objects[i].width / 2, objects[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4);
	}
	//-- Show what you got
	imshow("Capture - Face detection", frame);
}