#include <fstream>
#include <iostream>
#include <thread>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "caffe/caffe.hpp"
#include "cv.h"  
#include "student/student.hpp"
#include "student/behavior.hpp"
#include "student/functions.hpp"
#include<cmath>
#include<tuple>

using namespace cv;
using namespace std;
using namespace caffe;
//vector<Class_Info>class_info_all;
//vector<int>student_valid;
//vector<vector<Student_Info>>students_all(70);
int standard_frame = 1;

int GetStandaredFeats(Net &net1, PoseInfo &pose, Mat &frame_1080, int &n,int &n1,float &score_all, string &output, int &max_student_num, vector<vector<Student_Info>>&students_all, vector<int>&student_valid, vector<Class_Info>&class_info_all){
	if (n%standard_frame == 0){
		Timer timer;
		Mat frame;
		cv::resize(frame_1080, frame, Size(0, 0), 2 / 3., 2 / 3.);
		pose_detect(net1, frame, pose);
		int stu_n = 0;
		int stu_real = 0;
		float score_sum = 0;
		for (int i = 0; i < pose.subset.size(); i++){
			float score = float(pose.subset[i][18]) / pose.subset[i][19];			
			if (pose.subset[i][19] >= 4 && score >= 0.4){
				stu_real++;
				if (pose.subset[i][1] == -1){
					for (int j = 0; j < 18; j++){
						if (pose.subset[i][j] != -1)cv::circle(frame, Point2f(pose.candicate[pose.subset[i][j]][0], pose.candicate[pose.subset[i][j]][1]), 5, cv::Scalar(0,255,255), -1);
					}
				}
				if (pose.subset[i][1] != -1){
					score_sum += pose.subset[i][18];
					stu_n++;
				}
			}
		}
		if (stu_n >= max_student_num){
			//max_student_num = stu_n;
			cout << stu_n << " / " << stu_real << endl;
			n1++;
			//if (score_sum > score_all){
				//score_all = score_sum;

				student_valid.clear();
				for (int i = 0; i < 70; i++){
					students_all[i].clear();
				}
				//int xuhao = 0;
				for (int i = 0; i < pose.subset.size(); i++){
					float score = float(pose.subset[i][18]) / pose.subset[i][19];
					if (pose.subset[i][19] >= 4 && score >= 0.4){
						if (pose.subset[i][1] != -1){
							float wid1 = 0, wid2 = 0, wid = 0;
							if (pose.subset[i][2] != -1 && pose.subset[i][5] != -1){
								wid1 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][2]][0]);
								wid2 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][5]][0]);
								wid = MAX(wid1, wid2);
								if (wid == 0)continue;
							}
							if (pose.subset[i][2] != -1 && pose.subset[i][5] == -1){
								wid1 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][2]][0]);
								wid2 = wid1;
								wid = wid1;
								if (wid == 0)continue;
							}
							if (pose.subset[i][2] == -1 && pose.subset[i][5] != -1){
								wid1 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][5]][0]);
								wid2 = wid1;
								wid = wid1;
								if (wid == 0)continue;
							}


							Rect standard_rect;
							standard_rect.x = pose.candicate[pose.subset[i][1]][0] - wid - 5;
							standard_rect.y = pose.candicate[pose.subset[i][1]][1] - 5;
							standard_rect.width = wid1 + wid2 + 10;
							standard_rect.height = wid1 + wid2;
							if (standard_rect.height < 5)standard_rect.height = 15;
							refine(standard_rect, frame);
							cv::rectangle(frame, standard_rect, Scalar(0, 0, 255), 2, 8, 0);

							Student_Info student_ori;
							student_ori.body_bbox = standard_rect;
							student_ori.neck_loc = Point2f(pose.candicate[pose.subset[i][1]][0], pose.candicate[pose.subset[i][1]][1]);

							if (pose.subset[i][0] != -1){
								student_ori.loc = Point2f(pose.candicate[pose.subset[i][0]][0], pose.candicate[pose.subset[i][0]][1]);
								student_ori.front = true;
							}
							else{
								student_ori.loc = student_ori.neck_loc;
								student_ori.front = false;
							}
							student_valid.push_back(i);
							students_all[i].push_back(student_ori);
							string b = output + "/" + "00.jpg";
							//cv::circle(frame, student_ori.loc, 3, cv::Scalar(0, 0, 255), -1);
							//if (xuhao == max_student_num - 1){
							//	cv::putText(frame, to_string(xuhao), student_ori.loc, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
							//}
							//else{
							cv::putText(frame, to_string(i), student_ori.loc, FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
							//}
							//xuhao++;
							cv::imwrite(b, frame);
						}
					}
				}
			//}
		}
		return n1;
	}
}

int GetStandaredFeats1(Net &net1, Net &net4, jfda::JfdaDetector &detector ,vector<FaceInfo>&standard_faces, PoseInfo &pose, Mat &frame_1080, int &n, int &n1, string &output, int &max_student_num){
	if (n%standard_frame == 0){
		Timer timer;
		Mat frame;
		cv::resize(frame_1080, frame, Size(0, 0), 2 / 3., 2 / 3.);
		pose_detect(net1, frame, pose);
		int stu_n = 0;
		int stu_real = 0;
		float score_sum = 0;
		for (int i = 0; i < pose.subset.size(); i++){
			float score = float(pose.subset[i][18]) / pose.subset[i][19];
			if (pose.subset[i][19] >= 4 && score >= 0.4){
				stu_real++;
	
				if (pose.subset[i][1] != -1 && pose.subset[i][0] != -1){
					stu_n++;
				}
			}
		}
		if (stu_n >= max_student_num){
			//max_student_num = stu_n;
			cout << stu_n << " / " << stu_real << endl;
			n1++;
			
			for (int i = 0; i < pose.subset.size(); i++){
				float score = float(pose.subset[i][18]) / pose.subset[i][19];
				if (pose.subset[i][19] >= 4 && score >= 0.4){
					if (pose.subset[i][1] != -1 && pose.subset[i][0] != -1){
						float wid1 = 0, wid2 = 0, wid = 0;
						if (pose.subset[i][2] != -1 && pose.subset[i][5] != -1){
							wid1 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][2]][0]);
							wid2 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][5]][0]);
							wid = MAX(wid1, wid2);
							if (wid == 0)continue;
						}
						if (pose.subset[i][2] != -1 && pose.subset[i][5] == -1){
							wid1 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][2]][0]);
							wid2 = wid1;
							wid = wid1;
							if (wid == 0)continue;
						}
						if (pose.subset[i][2] == -1 && pose.subset[i][5] != -1){
							wid1 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][5]][0]);
							wid2 = wid1;
							wid = wid1;
							if (wid == 0)continue;
						}

						Rect face_box;
						face_box.x = pose.candicate[pose.subset[i][0]][0] - 1.6*wid / 2;
						face_box.y = pose.candicate[pose.subset[i][0]][1] - 1.8*wid *1.2 / 2;
						face_box.width = wid*1.8;
						face_box.height = wid*1.8;
						refine(face_box, frame);
						Rect face_bbox_1080(face_box.x * 3 / 2, face_box.y * 3 / 2, face_box.width * 3 / 2., face_box.height * 3 / 2);
						Mat faceimg = frame_1080(face_bbox_1080);			
						vector<FaceInfoInternal>facem;
						vector<FaceInfo> faces = detector.Detect(faceimg, facem);
						if (faces.size() != 0){
							FaceInfo faceinfo = faces[0];
							faceinfo.feature = Extract(net4, faceimg, faceinfo);
							string output3 = "/home/lw/student_api_no_Hik/output_face/" + to_string(i);
							faceinfo.path = output3;
							standard_faces.push_back(faceinfo);

							if (!fs::IsExists(output3)){
								fs::MakeDir(output3);
							}
							string b = output3 + "/" + "standard.jpg";
							//resize(faceimg, faceimg, Size(96, 112));
							cv::imwrite(b, faceimg);
						}
					}
				}
			}
			//}
		}
		return n1;
	}
}




std::tuple<vector<vector<Student_Info>>, vector<Class_Info>>student_detect(Net &net1, Net &net2, Net &net3, Net &net4, jfda::JfdaDetector &detector, Mat &image_1080, int &n, PoseInfo &pose, string &output, vector<vector<Student_Info>>&students_all, vector<int>&student_valid, vector<Class_Info> &class_info_all, vector<FaceInfo>&standard_faces){
	/*vector<Student_Info>student_detect(Net &net1, Mat &image, int &n, PoseInfo &pose,string &output)*/
	Timer timer;
	
	if (n % standard_frame == 0){
		Mat image;
		cv::resize(image_1080, image, Size(0, 0), 2 / 3., 2 / 3.);
		timer.Tic();
		pose_detect(net1, image, pose);
		/*timer.Toc();
		cout << "pose detect cost " << timer.Elasped() / 1000.0 << " s" << endl;
		timer.Tic();*/
		int color[18][3] = { { 255, 0, 0 }, { 255, 85, 0 }, { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 255, 0, 0 }, { 0, 255, 0 }, { 0, 0, 255 }, { 0, 255, 170 }, { 0, 255, 255 }, { 0, 170, 255 }, { 0, 255, 170 }, { 0, 255, 255 }, { 0, 170, 255 }, { 170, 0, 255 }, { 170, 0, 255 }, { 255, 0, 170 }, { 255, 0, 170 } };
		int x[18];
		int y[18];
		int num_turn_body = 0;

		/*int area1 = 0, area2 = 0, area3 = 0;
		vector<Point2f>area1_ = { { 283, 432 }, { 848, 432 }, { 1080, 709 }, { 0, 706 } };
		vector<Point2f>area2_ = { { 423, 347 }, { 769, 347 }, { 842,432 }, { 288, 432 } };
		vector<Point2f>area3_ = { { 489, 308 }, { 730, 308 }, { 772, 347 }, { 427, 347 } };*/
		

		for (int i = 0; i < pose.subset.size(); i++){
			Student_Info student_info;
			student_info.cur_frame1 = n;
			
			int v = 0;
			float score = float(pose.subset[i][18]) / pose.subset[i][19];
			if (pose.subset[i][19] >= 4 && score >= 0.4){
				for (int j = 0; j < 8; j++){
					if (pose.subset[i][j] == -1){
						x[j] = 0;
						y[j] = 0;
						v = 1;
						if (j == 0 || j == 4 || j == 7){
							v = 0;
						}
					}
					else{
						x[j] = pose.candicate[pose.subset[i][j]][0];
						y[j] = pose.candicate[pose.subset[i][j]][1];
					}
					//student_info.all_points.push_back(Point2f(x[j], y[j]));
				}

				//-------------------------ͳ����������---------------------------------

				/*int dis = 0;
				if (y[1] != 0){
					if (y[3] != 0){
						dis = abs(y[3] - y[1]);
					}
					else if (y[6] != 0){
						dis = abs(y[6] - y[1]);
					}
					else {
						if (y[1] < image.size().height / 2)dis = 20;
						else dis = 30;
					}
				}
				Point2f cur(x[1],y[1]+dis);
				if (PtInAnyRect2(cur, area1_[0], area1_[1], area1_[2], area1_[3])){
					circle(image, cur, 3, Scalar(255, 0, 0), -1);
					area1++;
				}
				if (PtInAnyRect2(cur, area2_[0], area2_[1], area2_[2], area2_[3])){
					circle(image, cur, 3, Scalar(0, 255, 0), -1);
					area2++;
				}

				if (PtInAnyRect2(cur, area3_[0], area3_[1], area3_[2], area3_[3])){
					circle(image, cur, 3, Scalar(0, 255, 255), -1);
					area3++;
				}*/

				//--------------------------�ж�������ͷ-----------------------------------

				if (pose.subset[i][1] != -1 && pose.subset[i][0] != -1){
					float wid1 = 0, wid2 = 0, wid = 0;
					if (pose.subset[i][2] != -1 && pose.subset[i][5] != -1){
						wid1 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][2]][0]);
						wid2 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][5]][0]);
						wid = MAX(wid1, wid2);
						if (wid == 0)continue;
					}
					if (pose.subset[i][2] != -1 && pose.subset[i][5] == -1){
						wid1 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][2]][0]);
						wid2 = wid1;
						wid = wid1;
						if (wid == 0)continue;
					}
					if (pose.subset[i][2] == -1 && pose.subset[i][5] != -1){
						wid1 = abs(pose.candicate[pose.subset[i][1]][0] - pose.candicate[pose.subset[i][5]][0]);
						wid2 = wid1;
						wid = wid1;
						if (wid == 0)continue;
					}

					Rect face_box;
					face_box.x = pose.candicate[pose.subset[i][0]][0] - 1.6*wid / 2;
					face_box.y = pose.candicate[pose.subset[i][0]][1] - 1.8*wid *1.2 / 2;
					face_box.width = wid*1.8;
					face_box.height = wid*1.8;
					refine(face_box, image);
					student_info.face_bbox = face_box;

					/*Rect face_bbox_1080(face_box.x * 3 / 2, face_box.y * 3 / 2, face_box.width * 3 / 2., face_box.height * 3 / 2);						
					std::tuple<bool, float>front_face_or_not = is_front_face(net3, image_1080, face_box);
					student_info.front_face = get<0>(front_face_or_not);
					string sco = to_string(get<1>(front_face_or_not));
					if (student_info.front_face == true){
						putText(image, sco, Point(face_box.x, face_box.y), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 255));
						cv::rectangle(image, face_box, Scalar(0, 255, 0), 2, 8, 0);
					}
					else {
						cv::rectangle(image, face_box, Scalar(255, 0, 0), 2, 8, 0);
						putText(image, sco, Point(face_box.x, face_box.y), FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 255));
					}*/

				}

				//----------------------------------�жϷ������Ķ���-----------------------------------------------

				detect_discontinuous_behavior(net2, image, pose, student_info, i, v, x, y, num_turn_body);

				//--------------------use IOU to classify--------------------------------------------------------------------------------

				//----------------obtain a rect range for i person in a new frame---------------------
				if (/*pose.subset[i][0] != -1&&*/pose.subset[i][1] != -1 && pose.subset[i][2] != -1 && pose.subset[i][5] != -1){

					float wid1 = abs(x[1] - x[2]);
					float wid2 = abs(x[1] - x[5]);
					float wid = MAX(wid1, wid2);
					if (wid == 0)continue;

					Rect rect_for_save;
					rect_for_save.x = x[1] - wid - 5;
					rect_for_save.y = y[1] - (wid1 + wid2 - 5);
					rect_for_save.width = wid1 + wid2 + 10;
					rect_for_save.height = 2 * (wid1 + wid2 - 5);
					if (rect_for_save.height < 5)rect_for_save.height = 15;
					//cv::rectangle(image, rect_for_save, Scalar(0, 255, 0), 2, 8, 0);
					refine(rect_for_save, image);
					student_info.body_for_save = rect_for_save;
					int thr = 0;
					if (y[1] < image.size().height / 3)thr = 30;
					else thr = 40;
					Rect cur_rect;
					if (student_info.arm_vertical){
						cur_rect.x = x[1] - wid - 5;
						cur_rect.y = y[1];
						cur_rect.width = wid1 + wid2 + 10;
						cur_rect.height = wid1 + wid2 - 15 + thr;
					}
					else{
						cur_rect.x = x[1] - wid;
						cur_rect.y = y[1];
						cur_rect.width = wid1 + wid2 + 10;
						cur_rect.height = wid1 + wid2;
						if (cur_rect.height < 5)cur_rect.height = 15;
					}
					refine(cur_rect, image);

					//cv::rectangle(image, cur_rect, Scalar(0, 255, 0), 2, 8, 0);
					student_info.body_bbox = cur_rect;
					if (cur_rect.y < image.size().height*0.3 && cur_rect.height>80){
						cur_rect = Rect(0, 0, 1, 1);
					}
					student_info.neck_loc = Point2f(x[1], y[1]);
					if (pose.subset[i][0] != -1){
						student_info.loc = Point2f(x[0], y[0]);
						student_info.front = true;
					}
					else{
						student_info.loc = student_info.neck_loc;
						student_info.front = false;
					}

					//---------------- IOU recognization in a rect range --------------------------------------
					
					std::multimap<float, int, greater<float>>IOU_map;
					for (int j = 0; j < student_valid.size(); j++){
						float cur_IOU = Compute_IOU(students_all[student_valid[j]][0].body_bbox, cur_rect);
						IOU_map.insert(make_pair(cur_IOU, student_valid[j]));
					}
					if (IOU_map.begin()->first > 0){
						int thre = (student_info.loc.y < image.size().height / 2) ? 100 : 150;
						int size1 = students_all[IOU_map.begin()->second].size();
						float dis = abs(students_all[IOU_map.begin()->second][size1 - 1].loc.y - student_info.loc.y);
						if (dis < thre){
							students_all[IOU_map.begin()->second].push_back(student_info);
						}
					}
					else{
						//cv::rectangle(image,student_info.body_bbox,Scalar(0,255,0),1,8,0);
						students_all[69].push_back(student_info);
					}
				}

				/*for (int j = 0; j < 8; j++){
					if (!(x[j] || y[j])){
					continue;
					}
					else{
					cv::circle(image, Point2f(x[j], y[j]), 3, cv::Scalar(color[j][0], color[j][1], color[j][2]), -1);
					}
					}*/

			} //if (pose.subset[i][19] >= 3 && score >= 0.4) end
		}//for (int i = 0; i < pose.subset.size(); i++) end

		//----------------------------------------------------------------------------------------------------------------
		/*for (int i = 0; i < 4; i++){
			if (i < 3){
				line(image, area1_[i], area1_[i + 1], cv::Scalar(255, 0, 0), 2);
				line(image, area2_[i], area2_[i + 1], cv::Scalar(0, 255, 0), 2);
				line(image, area3_[i], area3_[i + 1], cv::Scalar(0, 255, 255), 2);
			}
			else{
				line(image, area1_[3], area1_[0], cv::Scalar(255, 0, 0), 2);
				line(image, area2_[3], area2_[0], cv::Scalar(0, 255, 0), 2);
				line(image, area3_[3], area3_[0], cv::Scalar(0, 255, 255), 2);
			}
		}	
		cv::putText(image, to_string(area1), Point((area1_[0].x + area1_[1].x) / 2, (area1_[0].y + area1_[3].y) / 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255),2);
		cv::putText(image, to_string(area2), Point((area2_[0].x + area2_[1].x) / 2, (area2_[0].y + area2_[3].y) / 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255),2);
		cv::putText(image, to_string(area3), Point((area3_[0].x + area3_[1].x) / 2, (area3_[0].y + area3_[3].y) / 2), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255),2);
*/
		//-------------------------����ʶ��--------------------------------------------
		face_recog(net4, detector, students_all, student_valid, n, image_1080, standard_faces);


		//----------------------������Ϊ----------------------------------

		Analys_Behavior(students_all, student_valid, class_info_all, image_1080, n, num_turn_body);
		if (n % (10) == 0){
			writeJson(student_valid, students_all, class_info_all, output, n);
		}

		


		//drawGrid(image,student_valid,students_all);
		string output1;	
		output1 = output + "/" + to_string(n) + ".jpg";
		//cv::resize(image, image, Size(0, 0), 1 / 2., 1 / 2.);
		cv::imwrite(output1, image);
		timer.Toc();
		cout << "the " << n << " frame cost " << timer.Elasped() / 1000.0 << " s" << endl;
	} //if (n % standard_frame == 0) end
	return std::make_tuple(students_all, class_info_all);
}
