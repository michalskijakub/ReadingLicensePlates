#include "plate_utils.h"

using namespace std;
using namespace cv;
using namespace tesseract;

map<string, vector<string>> tablice_wojewodzkie = {
		{"Wojewodztwo podlaskie", {"BI", "BS", "BL", "BAU", "BIA", "BBI", "BGR", "BHA", "BKL", "BMN", "BSE", "BSI", "BSK", "BSU", "BWM", "BZA", "BLM"}},
		{"Wojewodztwo kujawsko-pomorskie", {"CB", "CG", "CT", "CW", "CAL", "CBR", "CBY", "CCH", "CGD", "CGR", "CIN", "CLI", "CMG", "CNA", "CRA", "CRY", "CSE", "CSW", "CTR", "CTU", "CWA", "CWL", "CZN"}},
		{"Wojewodztwo dolnoslaskie", {"DJ", "DL", "DB", "DW", "DBL", "DDZ", "DGR", "DGL", "DJA", "DJE", "DKA", "DKL", "DLE", "DLB", "DLU", "DLW", "DMI", "DOL", "DOA", "DPL", "DSR", "DST", "DSW", "DTR", "DBA", "DWL", "DWR", "DZA", "DZG", "DZL" }},
		{"Wojewodztwo lodzkie", {"EL", "EPA", "EPI", "EP", "ES", "EBE", "EBR", "EKU", "EOP", "EPJ", "EPD", "ERA", "ERW", "ESI", "ESK", "ETM", "EWI", "EWE", "EZD", "EZG", "ELA", "ELE", "ELW", "ELC"}},
		{"Wojewodztwo mazowieckie", {"WO", "WP", "WR", "WS", "WA", "WPY", "WB", "WD", "WE", "WU", "WH", "WF", "WW", "WI", "WJ", "WK", "WN", "WT", "WX", "WY", "WBR", "WCI", "WCI", "WG", "WGS", "WGM", "WGR", "WKZ", "WZ", "WLI", "WMA", "WM", "WML", "WND", "WOR", "WOS", "WOT", "WPI", "WPR", "WPP", "WPS", "WPZ", "WPU", "WPL", "WPN", "WRA", "WSI", "WSE", "WSC", "WSK", "WSZ", "WZ", "WWE", "WV", "WWL", "WWY", "WZU", "WZW", "WZY", "WLS"}},
		{"Wojewodztwo lubuskie", {"FG", "FZ", "FGW", "FKR", "FMI", "FNW", "FSD", "FSU", "FSW", "FSL", "FWS", "FZG", "FZA", "FZI"}},
		{"Wojewodztwo pomorskie", {"GD", "GA", "GSP", "GS", "GBY", "GCH", "GCZ", "GDA", "GKA", "GKS", "GKW", "GLE", "GMB", "GND", "GPU", "GST", "GSZ", "GSL", "GTC", "GWE", "GWO"}},

};

int main() {
	TessBaseAPI* api;
	CascadeClassifier plateCascade;
	VideoCapture cap;

	//string mediaPath = "Resources\\video3.mov";
	string mediaPath = "Resources\\a.mov";

	enum class mode_t mode = mode_t::BRIDGE;

	if (initApp(&api, plateCascade, cap, mediaPath))
		return -1;
	
	map<string, int> foundLocation;
	Mat img, img_gray;
	vector<Rect> plates;
	bool lock = false;
	bool stats = false;
	bool found = false;
	static int FRAME_SLEEP = 28;
	int counter = FRAME_SLEEP;
	string number;

	while (cap.read(img))
	{
		prepareFrame(img, plateCascade, plates, mode);

		if (!lock) {
			checkPlate(img, plates, api, number);
			
			int result = searchDB(plates, tablice_wojewodzkie, number, foundLocation);

			if (!result) {
				found = true;
				stats = true;
				lock = true;
			}
		}//end if (!lock)

		if (found) {
			counter--;
		}
		if (counter == FRAME_SLEEP && found) {
			lock = true;
		}
		
		if (found && counter == 0) {
			lock = false;
			found = false;
			counter = FRAME_SLEEP;
		}

		system("cls");
		if (stats) {
			displayStats(foundLocation);
		}
		

		cv::imshow("Video", img);
		if (waitKey(50) == 13) break;
	}

	endTesseractAPI(&api);
	cap.release();
	cv::destroyAllWindows();
	return 0;
}