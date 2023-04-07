
#include "opencv.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <ctype.h>
#include<iostream>
#include <baseapi.h> //do tesseracct OCR
#include <allheaders.h> //do tesseracct OCR
#include "provinces.cpp"
#include<algorithm>
#include <iostream>
#include <map>
#include <unordered_map>
#include <cstdio>
#include <vector>
#include <iostream>
#include <map>
#include <string>
#include <string_view>

using namespace std;
using namespace cv;
using namespace tesseract;


map<string, string> province = {
		{"BI", "BIAŁYSTOK"},
		{"BIA", "Powiat białostocki"},
		{"BS", "Suwałki"},
		{"BL", "Łomża"},
		{"BAU", "powiat augustowski"},
		{"BIA", "powiat białostocki"},
		{"BBI", "powiat bielski"},
		{"BGR", "powiat grajewski"},
		{"BHA", "powiat hajnowski"},
		{"BKL", "powiat kolneński"},
		{"BMN", " powiat moniecki"},
		{"BSE", "powiat sejneński"},
		{"BSI", "powiat siemiatycki"},
		{"BSK", "powiat sokólski"},
		{"BSU", "powiat suwalski"},
		{"BWM", "powiat wysokomazowiecki"},
		{"BZA", "powiat zambrowski"},
		{"BLM", "powiat łomżyński"},

		{"CB", "Bydgoszcz"},
		{"CG", "Grudziądz"},
		{"CT", "Toruń"},
		{"CW", "Włocławek"},
		{"CAL", "powiat aleksandrowski"},
		{"CBR", "powiat brodnicki"},
		{"CBY", "powiat bydgoskii"},
		{"CCH", "powiat chełmiński"},
		{"CGD", "powiat golubsko-dobrzyński"},
		{"CGR", "powiat grudziądzki"},
		{"CIN", "powiat inowrocławski"},
		{"CLI", " powiat lipnowski"},
		{"CMG", "powiat mogileński"},
		{"CNA", "powiat nakielski"},
		{"CRA", "powiat radziejowski"},
		{"CRY", " powiat rypiński"},
		{"CSE", "powiat sępoleński"},
		{"CSW", "powiat świecki"},
		{"CTR", "powiat toruński"},
		{"CTU", "powiat tucholski"},
		{"CWA", "powiat wąbrzeski"},
		{"CWL", "powiat włocławski"},
		{"CZN", "powiat żniński"},

		{"DJ", " Jelenia Góra"},
		{"DL", "Legnica"},
		{"DB", "Wałbrzych"},
		{"DW", "Wrocław"},
		{"DBL", "powiat bolesławiecki"},
		{"DDZ", "powiat dzierżoniowski"},
		{"DGR", "powiat górowski"},
		{"DGL", "powiat głogowski"},
		{"DJA", "powiat jaworski"},
		{"DJE", "powiat jeleniogórski"},
		{"DKA", "powiat kamiennogórski"},
		{"DKL", "powiat kłodzki"},
		{"DLE", "powiat legnicki"},
		{"DLB", "powiat lubański"},
		{"DLU", "powiat lubiński"},
		{"DLW", "powiat lwówecki"},
		{"DMI", "powiat milicki"},
		{"DMI", "powiat milicki"},
		{"DMI", "powiat milicki"},
		{"DMI", "powiat milicki"},
		{"DMI", "powiat milicki"},
		{"DMI", "powiat milicki"},
		{"DMI", "powiat milicki"},
		{"DMI", "powiat milicki"},
		{"DMI", "powiat milicki"},
		{"DMI", "powiat milicki"},

};

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

	/*plateCascade.load("Resources\\haarcascade_russian_plate_number.xml");
	if (plateCascade.empty()) {
		cout << "XML file not loaded" << endl;
		endTesseractAPI(api);
		return 1;
	}*/

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

string inputCleaner(string input) {

	string temp = input;

	for (int i = 0; i < temp.size(); i++) {
		if (!isalpha(temp[i]) && !isdigit(temp[i])) {
			temp.erase(remove(temp.begin(), temp.end(), temp[i]), temp.end());
			i = 0;
		}
	}

	while (isdigit(temp[0]))
		temp.erase(remove(temp.begin(), temp.end(), temp[0]), temp.end());

	return temp;
}

string provinceFinder(string pre) {

	char twoLetters[4];
	char threeLetters[4];

	pre.copy(threeLetters, 3, 0);
	threeLetters[3] = '\0';

	pre.copy(twoLetters, 2, 0);
	twoLetters[2] = '\0';


	if (province[threeLetters] != "")
		return province[threeLetters];

	if (province[twoLetters] != "")
		return province[twoLetters];

	return NULL;
}



int main() {
	vector < string > data;
	vector < string > output;
	TessBaseAPI* api;
	CascadeClassifier plateCascade;
	VideoCapture cap;
	string mediaPath = "video3.MOV";
	map<string, int> foundLocation;
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
			data.push_back(number);
			rectangle(img, plates[i].tl(), plates[i].br(), Scalar(255, 0, 255), 3);
		}

		for (int i = 0; i < data.size(); i++) {
			string insert = provinceFinder(inputCleaner(data[i]));
			if (insert != "") {
				foundLocation[insert]++;
				output.push_back(insert);
				break;
			}
		}

		/*for (const auto& [key, value] : foundLocation) {
			std::cout << key << " hav value " << value << "; ";
		}*/

		for (int i = 0; i < output.size(); i++) {
			cout << output[i] << " has value " << foundLocation[output[i]] << "; ";
		}


		data.clear();

		imshow("Video", img);
		if (waitKey(50) == 13) break;
	}

	endTesseractAPI(&api);
	cap.release();
	destroyAllWindows();
	return 0;
}
