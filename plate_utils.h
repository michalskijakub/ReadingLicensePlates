#if !defined _PLATE_UTILS_H_
#define _PLATE_UTILS_H

#include <map>
#include <vector>
#include <string>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <baseapi.h> //do tesseracct OCR
#include <allheaders.h>
#include "opencv.hpp"
#include <opencv2/imgproc.hpp>
#include <ctype.h>
#include <algorithm>
#include <opencv2/objdetect.hpp>

enum class mode_t {
	SIDE,
	BRIDGE
};

void endTesseractAPI(tesseract::TessBaseAPI** api);
int initApp(tesseract::TessBaseAPI** api, cv::CascadeClassifier& plateCascade, cv::VideoCapture& cap, std::string const& videoPath);
void displayStats(std::map<std::string, int> found_locations);
void extractPlateNumber(char** input, std::string& plateNumber);
cv::Mat reduceImage(cv::Mat const& img, int K);
cv::Mat cleanPlate(cv::Mat const& img);
std::string readCharactersFromPlate(cv::Mat const& plate, tesseract::TessBaseAPI** api);
void prepareFrame(cv::Mat& img , cv::CascadeClassifier& plateCascade, std::vector<cv::Rect>& plates, enum class mode_t mode);
void checkPlate(cv::Mat& img, std::vector<cv::Rect>& plates, tesseract::TessBaseAPI* api, std::string& number);
int searchDB(std::vector<cv::Rect>& plates, std::map<std::string, std::vector<std::string>> tablice, std::string number, std::map<std::string, int>& foundLocation);

#endif