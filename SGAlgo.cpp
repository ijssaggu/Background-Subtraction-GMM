#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

int k = 3;
float alpha = 0.04;
float weightThresh = 0.5;
float inSigma = 6.0;
float bgTol = 30;

float ***muBlue;
float ***muGreen;
float ***muRed;
float ***w;
float ***sigma;

Mat frame;
Mat bg;
Mat fore;

double dHeight;
double dWidth;

string video;

VideoCapture cap;
VideoWriter writeFore;
VideoWriter writeBG;


void initializeVars();
bool setVideo();
bool algorithm();
void updateBackgroundForegroundModel(int y, int x, bool isFit);
bool match(int xVal, float mu, float sigma);
void updateWeights(int x, int y, int M);
void sortByWeights(int x, int y);
void updateMuSigma(int x, int y, int i, float rho, int blue, int green, int red);
bool fitGaussian(int x, int y, Vec3b pix);


int main(int argc, char* argv[])
{

	if(argc <= 1) {
		video = "C:/Users/Inderjot Singh/Downloads/umcp.avi";
	} else {
		video = argv[1];
	}

	if(argc >= 3) {
		k = (atoi(argv[2])>2)?atoi(argv[2]):k;
	}

	if(argc == 4) {
		alpha = (atof(argv[3]) > 0 && atof(argv[3]) < 1) ? atof(argv[3]):alpha;
	}

	if (!setVideo()) {
		cout << "Cannot open the video" << endl;
		return -1;
	}

	initializeVars();

	if(!algorithm()) {
		return -1;
	} else {
		return 0;
	}

}

void initializeVars() {

	dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

	muBlue = new float **[(int)dWidth];
	muGreen = new float **[(int)dWidth];
	muRed = new float **[(int)dWidth];
	w = new float **[(int)dWidth];
	sigma = new float **[(int)dWidth];

	for(int i = 0 ; i < (int)dWidth ; ++i) {
		muBlue[i] = new float *[(int)dHeight];
		muGreen[i] = new float *[(int)dHeight];
		muRed[i] = new float *[(int)dHeight];
		w[i] = new float *[(int)dHeight];
		sigma[i] = new float *[(int)dHeight];

		for(int j = 0 ; j<(int)dHeight ; ++j) {
			muBlue[i][j] = new float [k];
			muGreen[i][j] = new float [k];
			muRed[i][j] = new float [k];
			w[i][j] = new float [k];
			sigma[i][j] = new float [k];

			for(int l = 0 ; l < k ; ++l) {
				muBlue[i][j][l] = 0.0;
				muGreen[i][j][l] = 0.0;
				muRed[i][j][l] = 0.0;
				w[i][j][l] = (float)(1.0/(float)k);
				//w[i][j][l] = 0.0;
				sigma[i][j][l] = inSigma;
			}
		}
	}

	bg = Mat((int)dHeight, (int)dWidth, CV_8UC3, Scalar(255, 255, 255));
	fore = Mat((int)dHeight, (int)dWidth, CV_8UC1, Scalar(0));

	writeFore = VideoWriter("foreground.avi", CV_FOURCC('M','J','P','G'), cap.get(CAP_PROP_FPS), Size((int)dWidth, (int)dHeight), true);
	writeBG = VideoWriter("background.avi", CV_FOURCC('M','J','P','G'), cap.get(CAP_PROP_FPS), Size((int)dWidth, (int)dHeight), true);
}

bool setVideo() {
	cap.release();
	if(!video.compare("cam")) {
		cap = VideoCapture(0);
	}else {
		cap = VideoCapture(video);
	}


	if (!cap.isOpened()) {
		return false;
	}

	return true;
}

bool algorithm() {

	while (1) {

		bool bSuccess = cap.read(frame); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			cout << "Video stream terminated. Resetting video." << endl;
			if(setVideo()) {
				continue;
			} else {
				return false;
			}
		}

		imshow("Video", frame); //show the frame in "MyVideo" window

		bool isFit;

		for(int x = 0 ; x < (int)dWidth ; ++x) {
			for(int y = 0 ; y < (int)dHeight ; ++y) {
				Vec3b pix = frame.at<Vec3b>(y,x);

				isFit = fitGaussian(x,y,pix);

				updateBackgroundForegroundModel(y,x, isFit);
			}
		}

		medianBlur(fore, fore, 5);
		imshow("background", bg);
		writeBG.write(bg);
		imshow("foreground", fore);
		Mat temp;
		cvtColor(fore, temp, CV_GRAY2RGB);
		writeFore.write(temp);

		if (waitKey(1) == 27) // If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}
}

void updateBackgroundForegroundModel(int y, int x, bool isFit) {
	float wSum = 0;
	int i=0;
	float bVal = 0;
	float gVal = 0;
	float rVal = 0;


	do{
		bVal += w[x][y][i]*muBlue[x][y][i];
		gVal += w[x][y][i]*muGreen[x][y][i];
		rVal += w[x][y][i]*muRed[x][y][i];

		wSum += w[x][y][i];
		i++;
	} while (wSum < weightThresh);

	bVal /= wSum;
	gVal /= wSum;
	rVal /= wSum;

	bg.at<Vec3b>(y,x)[0] = bVal;
	bg.at<Vec3b>(y,x)[1] = gVal;
	bg.at<Vec3b>(y,x)[2] = rVal;

	if(fabs(frame.at<Vec3b>(y,x)[0] - bVal ) > bgTol || fabs( frame.at<Vec3b>(y,x)[1] - gVal) > bgTol || fabs(frame.at<Vec3b>(y,x)[2] - rVal) > bgTol || !isFit) {
		fore.at<uchar>(y,x) = 255;
	} else {
		fore.at<uchar>(y,x) = 0;
	}
}

bool match(int xVal, float mu, float sigma) {
	if(fabs(xVal-mu)<=2.5*sigma) {
		return true;
	} else {
		return false;
	}
}

void updateWeights(int x, int y, int M) {
	float sum = 0;
	for(int i = 0 ; i < k ; ++i) {
		if(i == M) {
			w[x][y][i] = (1.0-alpha)*w[x][y][i] + alpha;
		} else {
			w[x][y][i] = (1.0-alpha)*w[x][y][i];
		}
		sum += w[x][y][i];
	}
	for(int i = 0 ; i < k ; ++i) {
		w[x][y][i] /= sum;
	}
}

void sortByWeights(int x, int y) {
	//using n^2 sorting here

	for(int i = 1 ; i < k ; ++i) {
		for(int j = 0 ; j < k-i ; ++j) {
			if(w[x][y][j]/sigma[x][y][j] < w[x][y][j+1]/sigma[x][y][j+1]) {
				float temp = w[x][y][j];
				w[x][y][j] = w[x][y][j+1];
				w[x][y][j+1]= temp;

				temp = muBlue[x][y][j];
				muBlue[x][y][j] = muBlue[x][y][j+1];
				muBlue[x][y][j+1]= temp;

				temp = muGreen[x][y][j];
				muGreen[x][y][j] = muGreen[x][y][j+1];
				muGreen[x][y][j+1]= temp;

				temp = muRed[x][y][j];
				muRed[x][y][j] = muRed[x][y][j+1];
				muRed[x][y][j+1]= temp;

				temp = sigma[x][y][j];
				sigma[x][y][j] = sigma[x][y][j+1];
				sigma[x][y][j+1]= temp;
			}
		}
	}
}

void updateMuSigma(int x, int y, int i, float rho, int blue, int green, int red) {
	muBlue[x][y][i] = (1.0-rho)*muBlue[x][y][i] + rho*blue;
	muGreen[x][y][i] = (1.0-rho)*muGreen[x][y][i] + rho*green;
	muRed[x][y][i] = (1.0-rho)*muRed[x][y][i] + rho*red;

	sigma[x][y][i] = sqrt((1.0-rho)*pow(sigma[x][y][i], 2.0) + (rho)*(pow(((float)blue - muBlue[x][y][i]),2.0) + pow(((float)green - muGreen[x][y][i]),2.0) + pow(((float)red - muRed[x][y][i]),2.0)));

}

bool fitGaussian(int x, int y, Vec3b pix) {

	bool foundMatch = false;
	int foundNum = 0;

	uchar blue = pix.val[0];
	uchar green = pix.val[1];
	uchar red = pix.val[2];



	for(int i = 0 ; i < k ; ++i) {
		if(match(blue,muBlue[x][y][i],sigma[x][y][i]) && match(green,muGreen[x][y][i],sigma[x][y][i]) && match(red,muRed[x][y][i],sigma[x][y][i])) {
			foundNum = i;
			foundMatch = true;
			updateWeights(x, y, i);
			sortByWeights(x, y);
			float rho=alpha*(1.0/(pow (2.0*M_PI*sigma[x][y][i]*sigma[x][y][i], 1.5)))*exp(-0.5*(pow(((float)blue - muBlue[x][y][i]), 2.0) + pow(((float)green - muGreen[x][y][i]), 2.0) + pow(((float)red - muRed[x][y][i]), 2.0))/pow(sigma[x][y][i], 2.0));

			updateMuSigma(x, y, i, rho, (int)blue, (int)green, (int)red);

			break;
		}
	}

	if(!foundMatch) {
		w[x][y][k-1] = 0.33/((float)k);
		muBlue[x][y][k-1] = (float)blue;
		muGreen[x][y][k-1] = (float)green;
		muRed[x][y][k-1] = (float)red;
		sigma[x][y][k-1] = inSigma;

		updateWeights(x,y,k-1);
	}

	/*if(x == 100 && y == 100) {
		cout<<"blue pix val:"<<(int)(blue)<<endl;
		cout<<"found status:"<<foundMatch<<" on "<<foundNum<<endl<<endl;
		for (int i = 0; i < k; ++i)
		{
			cout<<"w: "<<i<<" "<<w[x][y][i]<<endl;
			cout<<"mu blue: "<<i<<" "<<muBlue[x][y][i]<<endl;
			cout<<"sigma: "<<i<<" "<<sigma[x][y][i]<<endl<<endl;
		}
	}*/

	return foundMatch;
}








