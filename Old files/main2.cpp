#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <ctype.h>
#include<iostream>
#include <baseapi.h> //do tesseracct OCR
#include <allheaders.h> //do tesseracct OCR

using namespace std;
using namespace cv;
using namespace tesseract;



Mat reduce_image(Mat const& img, int K)
{
	int n = img.rows * img.cols;
	cv::Mat data = img.reshape(1, n);
	data.convertTo(data, CV_32F);

	vector<int> labels;
	Mat1f colors;
	kmeans(data, K, labels, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10000, 0.0001), 5, KMEANS_PP_CENTERS, colors);

	for (int i = 0; i < n; ++i) {
		data.at<float>(i, 0) = colors(labels[i], 0);
	}

	Mat reduced = data.reshape(1, img.rows);
	reduced.convertTo(reduced, CV_8U);

	return reduced;
}

Mat clean_plate(Mat const& img)
{
	Mat resized_img;
	resize(img, resized_img, Size(), 5.0, 5.0, INTER_CUBIC);//5.0, 5.0

	Mat equalized_img;
	equalizeHist(resized_img, equalized_img);

	Mat reduced_img(reduce_image(equalized_img, 8));

	Mat mask;
	adaptiveThreshold(reduced_img, mask, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 191, 20);
	Mat kernel(getStructuringElement(MORPH_RECT, Size(3, 3)));
	erode(mask, mask, kernel, Point(-1, -1), 1);

	return mask;
}

void endTesseractAPI(TessBaseAPI** api) {
	if (api != NULL && *api != NULL) {
		(*api)->End();
		delete (*api);
		*api = NULL;
	}
}

int initApp(TessBaseAPI** api, CascadeClassifier& plateCascade, VideoCapture& cap, string const& videoPath) {
	*api = new TessBaseAPI();
	if ((*api)->Init("C:\\Program Files\\Tesseract-OCR\\tessdata", "eng")) {
		cout << "Could not initialize tesseract." << endl;
		return 1;
	}

	plateCascade.load("Resources\\haarcascade_russian_plate_number.xml");
	if (plateCascade.empty()) {
		cout << "XML file not loaded" << endl;
		endTesseractAPI(api);
		return 1;
	}

	cap = VideoCapture(videoPath);
	if (!cap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		endTesseractAPI(api);
		return 1;
	}
	return 0;
}

void extractPlateNumber(char** input, string& plateNumber) {
	if (input == NULL || *input == NULL) return;

	for (size_t i = 0; *(*input + i) != '\0'; i++)
	{
		if (isalpha(*(*input + i)) || isdigit(*(*input + i))) {

			plateNumber.push_back(*(*input + i));
		}
	}
}

string readCharactersFromPlate(Mat const& plate, TessBaseAPI** api) {
	Mat gray;
	cvtColor(plate, gray, COLOR_BGR2GRAY);
	resize(gray, gray, Size(340, 104));
	Mat cleanedPlate(clean_plate(gray));

	imwrite("Resources\\cleanedPlate.png", cleanedPlate);

	Pix* image = pixRead("Resources\\cleanedPlate.png");
	(*api)->SetImage(image);

	char* output = (*api)->GetUTF8Text();
	string plateNumber;
	extractPlateNumber(&output, plateNumber);
	delete[] output;

	return plateNumber;
}

int main() {
	TessBaseAPI* api;
	CascadeClassifier plateCascade;
	VideoCapture cap;
	string mediaPath = "Resources\\video3.MOV";

	if (initApp(&api, plateCascade, cap, mediaPath)) return -1;

	Mat img, img_gray;
	vector<Rect> plates;
	while (cap.read(img))
	{
		Rect roi(0, 200, 1080, 1920 - 200);
		img = img(roi);
		resize(img, img, Size(), 0.7, 0.5);
		cvtColor(img, img_gray, COLOR_BGR2GRAY);

		plateCascade.detectMultiScale(img_gray, plates, 1.1, 4, 0, Size(100, 30), Size(170, 60));

		for (int i = 0; i < plates.size(); i++)
		{
			Mat imgCrop = img(plates[i]);
			string number = readCharactersFromPlate(imgCrop, &api);
			cout << number << endl;
			rectangle(img, plates[i].tl(), plates[i].br(), Scalar(255, 0, 255), 3);
		}

		imshow("Video", img);
		if (waitKey(50) == 13) break;
	}

	endTesseractAPI(&api);
	cap.release();
	destroyAllWindows();
	return 0;
}