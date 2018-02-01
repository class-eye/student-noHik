#include <iostream>
#include <string>
#include <cstring> 
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cv.h"  
#include <caffe/caffe.hpp>
#include <thread>
#include "student/student.hpp"
#include "student/functions.hpp"
#include "incCn/HCNetSDK.h"  
#include "incCn/PlayM4.h" 
#include<tuple>
using namespace std;
using namespace cv;
using namespace caffe;
using namespace fs;


void initValue(int &n, int &max_student_num, vector<Class_Info>&class_info_all, vector<int>&student_valid, vector<vector<Student_Info>>&students_all){
	n = 0;
	max_student_num = 0;
	student_valid.clear();
	for (int i = 0; i < 70; i++){
		students_all[i].clear();
	}
	class_info_all.clear();
}

//-------------------------------------------------OpenCV------------------------------------------------------

int main(){

	vector<Class_Info>class_info_all;
	vector<int>student_valid;
	vector<vector<Student_Info>>students_all(70);
	vector<FaceInfo>standard_faces;
	int max_student_num = 55;
	std::tuple<vector<vector<Student_Info>>, vector<Class_Info>>student_info;
	int n1 = 0, n2 = 0;
	float score_all = 0;


	if (caffe::GPUAvailable()) {
		caffe::SetMode(caffe::GPU, 0);
	}
	Net net1("../models/pose_deploy.prototxt");
	net1.CopyTrainedLayersFrom("../models/pose_iter_440000.caffemodel");
	Net net2("../models/handsnet.prototxt");
	net2.CopyTrainedLayersFrom("../models/handsnet_iter_12000.caffemodel");
	Net net3("../models/face.prototxt");
	net3.CopyTrainedLayersFrom("../models/face.caffemodel");
	Net net4("../models/facefeature.prototxt");
	net4.CopyTrainedLayersFrom("../models/facefeature.caffemodel");
	jfda::JfdaDetector detector("../models/p.prototxt", "../models/p.caffemodel", "../models/r.prototxt", "../models/r.caffemodel", \
		"../models/o.prototxt", "../models/o.caffemodel", "../models/l.prototxt", "../models/l.caffemodel");
	detector.SetMaxImageSize(3000);
	detector.SetMinSize(20);
	detector.SetStageThresholds(0.5, 0.4, 0.55);

	//string videodir = "/home/data/jiangbo/xiaoxue/code1/54";
	string resultdir = "/home/lw/student_api_no_Hik/output";

	string imgdir = "/home/lw/student_api/inputimg_1080/";

	string output = "/home/lw/student_api_no_Hik/output2";
	vector<string>imagelist = fs::ListDir(imgdir, { "jpg" });

	if (!fs::IsExists(output)){
		fs::MakeDir(output);
	}

	//for (int i = 0; i < imagelist.size(); i++){
	//	string imagep = imgdir + imagelist[i];
	//	Mat image = imread(imagep);
	//	//if (i < 20){
	//	PoseInfo pose1;
	//	cout << "processing: " << i << endl;
	//	if (i >= 9){
	//		if (n1 != 1){
	//			int a = GetStandaredFeats(net1, pose1, image, i, n1, score_all, output, max_student_num, students_all, student_valid, class_info_all);
	//		}
	//		else{
	//			cout << "n1 Finish" << endl;
	//			
	//		}
	//	}
	//	if (i >= 9){
	//		if (n2 != max_student_num){
	//			int b = GetStandaredFeats1(net1, net3, net4, detector, standard_faces, pose1, image, i, n2, output, max_student_num);
	//		}
	//		else{ cout << "n2 Finish" << endl; }
	//	}
	//	/*if (i == 555){
	//		int b = GetStandaredFeats1(net1, net3, net4, detector, standard_faces, pose1, image, i, n2, output, max_student_num);
	//	}*/
	//	if (n1 == 1 && n2 == max_student_num){
	//		cout << "--------------------Finish----------------" << endl;
	//		break;
	//	}
	//	//}
	//	/*else{
	//		PoseInfo pose;
	//		student_info = student_detect(net1, net2, image, i, pose, output, students_all, student_valid, class_info_all);
	//		}*/
	//}

	for (int i = 640; i < imagelist.size(); i++){

		string imagep = imgdir + imagelist[i];
		Mat image = imread(imagep);
		//if (i < 20){
		PoseInfo pose1;
		cout << "processing: " << i << endl;
		if (n1 != 1){
			int a = GetStandaredFeats(net1, pose1, image, i, n1, score_all, output, max_student_num, students_all, student_valid, class_info_all);
		}
		else{
			cout << "n1 Finish" << endl;
			break;
		}
	}

	for (int i = 0; i < imagelist.size(); i++){
		string imagep = imgdir + imagelist[i];
		Mat image = imread(imagep);
		PoseInfo pose;
		student_info = student_detect(net1, net2, net3, net4, detector, image, i, pose, output, students_all, student_valid, class_info_all, standard_faces,max_student_num);
		/*vector<vector<Student_Info>>students_all = get<0>(student_info);
		vector<Class_Info>class_info_all = get<1>(student_info);*/
	}


	//-------------------------VIDEO---------------------------------------

	//string videopath = "../test_face.mp4";
	///*string output = "";
	//int max_student_num = 0;*/
	//VideoCapture capture(videopath);
	//if (!capture.isOpened())
	//{
	//	printf("video loading fail");
	//}
	//long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
	//cout << "all " << totalFrameNumber << " frame" << endl;
	////long frameToStart = 13500; 
	////capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
	//Mat frame;
	//int n = 0;
	////std::tuple<vector<vector<Student_Info>>, vector<Class_Info>>student_info;
	//while (true)
	//{
	//	if (!capture.read(frame)){
	//		break;
	//	}
	//	if (n % 25 == 0){
	//		string output_c = "/home/lw/student_api_no_Hik/inputimg/" + to_string(n) + ".jpg";
	//		imwrite(output_c, frame);
	//	}
	//	n++;
	//	/*if (n < 3*25){
	//		cv::resize(frame, frame, Size(0, 0), 2 / 3., 2 / 3.);
	//		PoseInfo pose1;
	//		int a = GetStandaredFeats(net1, pose1, frame, n, n1, score_all, output, max_student_num, students_all, student_valid, class_info_all);
	//		if (a == 3)break;
	//	}
	//	n++;*/
	//}
	//capture.release();
	////VideoCapture capture1(videopath);
	////if (!capture1.isOpened())
	////{
	////	printf("video loading fail");
	////}
	////while (true)
	////{
	////	
	////	if (!capture1.read(frame)){
	////		break;
	////	}
	////	
	////	//cv::resize(frame, frame, Size(0, 0), 2 / 3., 2 / 3.);
	////	PoseInfo pose;
	////	student_info = student_detect(net1, net2, frame, n, pose, output, students_all, student_valid, class_info_all);
	////	/*vector<vector<Student_Info>>students_all = get<0>(student_info);
	////	vector<Class_Info>class_info_all = get<1>(student_info);*/
	////	n++;
	////}

}

