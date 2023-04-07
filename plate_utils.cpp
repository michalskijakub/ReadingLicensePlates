#include "plate_utils.h"


void endTesseractAPI(tesseract::TessBaseAPI** api) {
	if (api != NULL && *api != NULL) {
		(*api)->End();
		delete (*api);
		*api = NULL;
	}
}

int initApp(tesseract::TessBaseAPI** api, cv::CascadeClassifier& plateCascade, cv::VideoCapture& cap, std::string const& videoPath) {
	*api = new tesseract::TessBaseAPI();
	if ((*api)->Init("Resources\\Tesseract-OCR\\tessdata", "eng")) {
		std::cout << "Could not initialize tesseract." << std::endl;
		return 1;
	}

	plateCascade.load("Resources\\haarcascade_russian_plate_number.xml");
	if (plateCascade.empty()) {
		std::cout << "XML file not loaded" << std::endl;
		endTesseractAPI(api);
		return 1;
	}

	cap = cv::VideoCapture(videoPath);
	if (!cap.isOpened()) {
		std::cout << "Error opening video stream or file" << std::endl;
		endTesseractAPI(api);
		return 1;
	}
	return 0;
}

void displayStats(std::map<std::string, int> found_locations) {
	std::cout << "+-----------------------------------------------+" << std::endl;

	for (const auto& elem : found_locations){
		std::cout << elem.first << " " << elem.second << std::endl;
	}

	std::cout << "+-----------------------------------------------+" << std::endl;
}

void extractPlateNumber(char** input, std::string& plateNumber) {
	if (input == NULL || *input == NULL) return;

	for (size_t i = 0; *(*input + i) != '\0'; i++)
	{
		if (isalpha(*(*input + i)) || isdigit(*(*input + i))) {

			plateNumber.push_back(*(*input + i));
		}
	}
}

cv::Mat reduceImage(cv::Mat const& img, int K)
{
	int n = img.rows * img.cols;
	cv::Mat data = img.reshape(1, n);
	data.convertTo(data, CV_32F);

	std::vector<int> labels;
	cv::Mat1f colors;
	kmeans(data, K, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 10000, 0.0001), 5, cv::KMEANS_PP_CENTERS, colors);

	for (int i = 0; i < n; ++i) {
		data.at<float>(i, 0) = colors(labels[i], 0);
	}

	cv::Mat reduced = data.reshape(1, img.rows);
	reduced.convertTo(reduced, CV_8U);

	return reduced;
}

cv::Mat cleanPlate(cv::Mat const& img)
{
	cv::Mat resized_img;
	resize(img, resized_img, cv::Size(), 5.0, 5.0, cv::INTER_CUBIC);//5.0, 5.0

	cv::Mat gray;
	cv::cvtColor(resized_img, gray, cv::COLOR_BGR2GRAY);

	cv::Mat contrast;
	gray.convertTo(contrast, -1, 1.2, 0);

	cv::Mat eroded;
	cv::Mat kernel(cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

	cv::Mat denoised;
	fastNlMeansDenoising(contrast, denoised);

	cv::Mat final;
	final = denoised(cv::Rect(5, 40, denoised.size().width - 5, denoised.size().height - 60));

	return final;
}

std::string readCharactersFromPlate(cv::Mat const& plate, tesseract::TessBaseAPI** api) {

	cv::Mat final = cleanPlate(plate);

	imwrite("Resources\\cleanedPlate.png", final);

	Pix* image = pixRead("Resources\\cleanedPlate.png");\

	(*api)->SetImage(image);
	char* output = (*api)->GetUTF8Text();

	std::string plateNumber;
	extractPlateNumber(&output, plateNumber);
	delete[] output;

	return plateNumber;
}

void prepareFrame(cv::Mat& img, cv::CascadeClassifier& plateCascade, std::vector<cv::Rect>& plates, enum class mode_t mode) {
	cv::Rect roi;
	cv::Mat imgGray;
	if (mode == mode_t::SIDE) {
		roi = cv::Rect(0, 200, 1080, 1920 - 200);
	}
	else {
		roi = cv::Rect(0, 1050, 1080, 1920 - 1150);
	}
	
	img = img(roi);
	resize(img, img, cv::Size(), 0.7, 0.5);
	cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);
	plateCascade.detectMultiScale(imgGray, plates, 1.1, 4, 0, cv::Size(120, 30), cv::Size(240, 60));
}

void checkPlate(cv::Mat& img, std::vector<cv::Rect>& plates, tesseract::TessBaseAPI* api, std::string& number) {
	for(int i = 0; i < plates.size(); ++i){

		cv::Mat imgCrop = img(cv::Rect(plates[i].x - 20, plates[i].y, plates[i].width, plates[i].height));
		number = readCharactersFromPlate(imgCrop, &api);

		std::transform(number.begin(), number.end(), number.begin(), ::toupper);
		std::cout << "found: " << number << std::endl;
		cv::rectangle(img, plates[i].tl(), plates[i].br(), cv::Scalar(255, 0, 255), 3);
	}
}

int searchDB(std::vector<cv::Rect>& plates, std::map<std::string, std::vector<std::string>> tablice, std::string number, std::map<std::string, int>& foundLocation) {
	for (int i = 0; i < plates.size(); i++) {
		for (const auto& elem : tablice) {
			for (const auto& el : elem.second) {
				if (number.find(el) != -1) {
					int chars_left = number.length() - number.find(el) - el.length();
					if (chars_left >= 4) {
						number.at(number.find(el));
						foundLocation[elem.first]++;
						return 0;
						break;
					}

				}
			}
		}
	}
	return 1;
}
