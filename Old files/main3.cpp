#include "opencv.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <ctype.h>
#include <baseapi.h> //do tesseracct OCR
#include <allheaders.h> //do tesseracct OCR
#include <algorithm>
#include <iostream>
#include <map>
#include <unordered_map>
#include <cstdio>
#include <vector>
#include <string>
#include <string_view>

using namespace std;
using namespace cv;
using namespace tesseract;


map<string, string> province = {
		{"BI", "Bialystok"},
		{"BIA", "Powiat bialostocki"},
		{"BS", "Suwalki"},
		{"BL", "Łomża"},
		{"BAU", "powiat augustowski"},
		{"BBI", "powiat bielski"},
		{"BGR", "powiat grajewski"},
		{"BHA", "powiat hajnowski"},
		{"BKL", "powiat kolnenski"},
		{"BMN", " powiat moniecki"},
		{"BSE", "powiat sejneński"},
		{"BSI", "powiat siemiatycki"},
		{"BSK", "powiat sokolski"},
		{"BSU", "powiat suwalski"},
		{"BWM", "powiat wysokomazowiecki"},
		{"BZA", "powiat zambrowski"},
		{"BLM", "powiat łomżyński"},

		{"CB", "Bydgoszcz"},
		{"CG", "Grudziadz"},
		{"CT", "Torun"},
		{"CW", "Wloclawek"},
		{"CAL", "powiat aleksandrowski"},
		{"CBR", "powiat brodnicki"},
		{"CBY", "powiat bydgoskii"},
		{"CCH", "powiat chelminski"},
		{"CGD", "powiat golubsko-dobrzynski"},
		{"CGR", "powiat grudziadzki"},
		{"CIN", "powiat inowroclawski"},
		{"CLI", " powiat lipnowski"},
		{"CMG", "powiat mogilenski"},
		{"CNA", "powiat nakielski"},
		{"CRA", "powiat radziejowski"},
		{"CRY", " powiat rypinski"},
		{"CSE", "powiat sępolenski"},
		{"CSW", "powiat swiecki"},
		{"CTR", "powiat torunski"},
		{"CTU", "powiat tucholski"},
		{"CWA", "powiat wabrzeski"},
		{"CWL", "powiat wloclawski"},
		{"CZN", "powiat zninski"},
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
		{"EP", "Piotrków Trybunalski"},
		{"ES", "Skierniewice"},
		{"EL", "Lodz"},
		{"EBE", "powiat bełchatowski"},
		{"EBR", "powiat brzeziński"},
		{"EKU", "powiat kutnowski"},
		{"EOP", "powiat opoczyński"},
		{"EPA", "powiat pabianicki"},
		{"EPJ", "powiat pajęczański"},
		{"EPI", "powiat piotrkowski"},
		{"EPD", "powiat poddębicki"},

		{"ERA", "powiat radomszczański"},
		{"ERW", "powiat rawski"},
		{"ESI", "powiat sieradzki"},
		{"ESK", "powiat skierniewicki"},
		{"ETM", "powiat tomaszowski"},
		{"EWI", "powiat wieluński"},
		{"EWE", "powiat wieruszowski"},
		{"EZD", "powiat zduńskowolski"},
		{"EZG", "powiat zgierski"},
		{"ELA", "powiat łaski"},
		{"ELE", "powiat łęczycki"},
		{"ELW", "powiat łódzki wschodni"},
		{"ELC", "powiat łowicki"},
		{"RST", "powiat stalowa wola"},
		{"WPY", "powiat przysuski"},
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
	//return resized_img;
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

	//std::cout << plate << endl;

	Mat plateCopy = plate.clone();

	cv::Size plateSize = plate.size();
	int plateHeight = plateSize.height;
	int plateWidth = plateSize.width;
 
	int MAXR = 55;
	int MAXB = 55;
	int MAXG = 55;

	for (size_t i = 0; i < plateHeight; i++){
		for (size_t j = 0; j < plateWidth; j++){
			cv::Vec3b RGBPixel = plateCopy.at<Vec3b>(i, j);
			if (RGBPixel[0] > MAXR && RGBPixel[1] > MAXB && RGBPixel[2] > MAXG) {
				RGBPixel[0] = 255;
				RGBPixel[1] = 255;
				RGBPixel[2] = 255;
				plateCopy.at<Vec3b>(i, j) = RGBPixel;
			}
		}
	}
	//imshow("tablica", plateCopy);
	//waitKey(0);

	cvtColor(plateCopy, gray, COLOR_BGR2GRAY);
	
	resize(gray, gray, Size(340, 120));
	Mat cleanedPlate(clean_plate(gray));
	cleanedPlate = cleanedPlate(cv::Rect(0, 200, cleanedPlate.size().width, cleanedPlate.size().height - 341));
	

	Mat blurred;

	GaussianBlur(cleanedPlate, blurred, cv::Size(0, 0), 1.5);
	addWeighted(cleanedPlate, 3, blurred, -1.5, 0, blurred);

	int a = 10;
	int b = 10;

	Scalar value(255, 255, 255);
	copyMakeBorder(cleanedPlate, cleanedPlate, a, a, b, b, BORDER_CONSTANT, value);
	//imshow("final", cleanedPlate);
	//waitKey(0);
	
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

	return "";
}



int main() {
	vector < string > data;
	vector < string > output;
	TessBaseAPI* api;
	CascadeClassifier plateCascade;
	VideoCapture cap;
	string mediaPath = "Resources\\video3.MOV";
	map<string, int> foundLocation;
	if (initApp(&api, plateCascade, cap, mediaPath)) return -1;

	Mat img, img_gray;
	vector<Rect> plates;
	bool lock = false;
	while (cap.read(img))
	{
		Rect roi(0, 200, 1080, 1920 - 200);
		img = img(roi);
		resize(img, img, Size(), 0.7, 0.5);
		cvtColor(img, img_gray, COLOR_BGR2GRAY);

		plateCascade.detectMultiScale(img_gray, plates, 1.1, 4, 0, Size(130, 30), Size(240, 60));

		if (!lock) {
			for (int i = 0; i < plates.size(); i++)
			{
				Mat imgCrop = img(plates[i]);
				string number = readCharactersFromPlate(imgCrop, &api);

				std::transform(number.begin(), number.end(), number.begin(), ::toupper);
				//cout << "Odczytano tablice1: " << number << endl;


				std::transform(number.begin(), number.end(), number.begin(), ::toupper);
				if (number.length() <= 8 && (province.count(number.substr(0, 2)) == 1 || province.count(number.substr(0, 3))) == 1) {
					//cout << "Odczytano tablice: " << number.substr(0, 8) << endl;
					data.push_back(number.substr(0, 8));
				}

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
			
		}
		if (plates.size() == 1 && data.size() == 1) {
			lock = true;
		}
		
		if (plates.size() == 0 && data.size() == 0) {
			lock = false;
		}

		system("cls");
		if (output.size() > 0) {
			cout << "+-----------------------------------------------+" << endl;
			//cout << "locked: " << lock << endl;
			
			for (const auto& elem : foundLocation)
			{
				std::cout << elem.first << " " << elem.second << endl;
			}

			cout << "+-----------------------------------------------+" << endl;
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