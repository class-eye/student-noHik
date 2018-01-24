#ifndef _STUDENT_H_
#define _STUDENT_H_
#include <opencv2/core/core.hpp>
#include <caffe/caffe.hpp>
#include "pose.hpp"
#include "hand.hpp"
#include "face.hpp"
#include "jfda.hpp"
#include "Timer.hpp"
#include "fs.hpp"
#include "facefeature.hpp"
#include "incCn/HCNetSDK.h"  
#include "incCn/PlayM4.h" 
using namespace cv;
using namespace caffe;
struct Class_Info{
	bool all_bow_head=false;
	bool all_disscussion_2 = false;
	bool all_disscussion_4 = false;
	int cur_frame=0;
	
};
struct Student_Info{
	bool raising_hand=false;
	bool standing=false;
	bool disscussion= false;
	bool daze = false;
	bool bow_head = false;
	bool bow_head_each = false;

	bool turn_head = false;
	bool arm_vertical = false;
	bool whisper = false;
	bool turn_body = false;
	bool bow_head_tmp = false;
	Point2f loc;
	Point2f neck_loc;
	Rect body_bbox;
	Rect body_for_save;
	Rect face_bbox;
	//string output_body_dir;
	int away_from_seat = 0;
	int cur_frame1=0;
	int cur_size = 0;
	int energy = 0;
	int max_energy = 0;
	bool front=false;
	bool back = false;
	vector<int>miss_frame;
	//vector<Point2f>all_points;

	bool real_raise = false;
	bool front_face = false;
	float scores = 0.0;
	float face_scores = 0.0;
};
//vector<Student_Info> student_detect(Net &net1, Mat &image, int &n, PoseInfo &pose,string &output);
std::tuple<vector<vector<Student_Info>>, vector<Class_Info>>student_detect(Net &net1, Net &net2, Net &net3, Net &net4, jfda::JfdaDetector &detector, Mat &image, int &n, PoseInfo &pose, string &output, vector<vector<Student_Info>>&students_all, vector<int>&student_valid, vector<Class_Info> &class_info_all, vector<FaceInfo>&standard_faces);
int GetStandaredFeats(Net &net1, PoseInfo &pose, Mat &frame, int &n,int &n1, float &score_all, string &output, int &max_student_num, vector<vector<Student_Info>>&students_all, vector<int>&student_valid, vector<Class_Info>&class_info_all);

int GetStandaredFeats1(Net &net1, Net &net4,jfda::JfdaDetector &detector, vector<FaceInfo>&standard_faces, PoseInfo &pose, Mat &frame_1080, int &n, int &n1, string &output, int &max_student_num);
#endif