#include <caffe/caffe.hpp>
#define USE_OPENCV
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
//for multithreading
//#include"multithreading.h"
//for tracking
#include <sstream>
#include "fdssttracker.hpp"
#include <opencv2/objdetect/objdetect.hpp> //to include HOGDescriptor
//for thread
#include <unistd.h>
#include <cstdlib>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <future>
//for socket
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
//for face detection
#include "network.h"
#include "mtcnn.h"
#include <time.h>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;
using namespace std;
using namespace cv;
/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;
typedef std::pair<Mat, int> Result_all;

static const int kItemRepositorySize = 50;

#define IMG_WIDTH 1280
#define IMG_HEIGHT 720
#define BUFFERSIZE IMG_WIDTH*IMG_HEIGHT*3
typedef int Send_data;

class SocketMatTransmissionServer
{
public:
	SocketMatTransmissionServer(void);
	~SocketMatTransmissionServer(void);
	int sockConn;
private:
	unsigned char data_buf[BUFFERSIZE];

public:

	int socketConnect(int PORT);
	int receive_image(cv::Mat& image);
	int receive_data(uchar* buf, int size);
	int send_data(Send_data* buf, int size);
	void socketDisconnect(void);
};

SocketMatTransmissionServer::SocketMatTransmissionServer(void)
{
}


SocketMatTransmissionServer::~SocketMatTransmissionServer(void)
{
}


int SocketMatTransmissionServer::socketConnect(int PORT)
{
	int server_sockfd = socket(AF_INET, SOCK_STREAM, 0);

	struct sockaddr_in server_sockaddr;
	server_sockaddr.sin_family = AF_INET;
	server_sockaddr.sin_port = htons(PORT);
	server_sockaddr.sin_addr.s_addr = htonl(INADDR_ANY);

	if (bind(server_sockfd, (struct sockaddr *)&server_sockaddr, sizeof(server_sockaddr)) == -1)
	{
		perror("bind");
		return -1;
	}

	if (listen(server_sockfd, 5) == -1)
	{
		perror("listen");
		return -1;
	}

	struct sockaddr_in client_addr;
	socklen_t length = sizeof(client_addr);

	sockConn = accept(server_sockfd, (struct sockaddr*)&client_addr, &length);
	if (sockConn < 0)
	{
		perror("connect");
		return -1;
	}
	else
	{
		printf("connect successful!\n");
		return 1;
	}

	close(server_sockfd);
}


void SocketMatTransmissionServer::socketDisconnect(void)
{
	close(sockConn);
}

int SocketMatTransmissionServer::receive_data(uchar* buf, int size) {
	int count = 0;
	while (count < size) {
		int len = recv(sockConn, buf + count, size - count, 0);
		if (len <= 0) return -1;
		count += len;
	}
	return count;
}

int SocketMatTransmissionServer::send_data(Send_data* buf, int size) {
	int len = send(sockConn, buf, size, 0);
	return len;
}

int SocketMatTransmissionServer::receive_image(cv::Mat& image)
{
	memset(data_buf, 0, BUFFERSIZE);
	int rtn_size = 0;

	// head
	if ((rtn_size = receive_data(data_buf, sizeof(int))) < 0) {
		return -1;
	}
	int needRecv = ((int*)data_buf)[0];

	// data
	if ((rtn_size = receive_data(data_buf, needRecv)) < 0) {
		return -1;
	}

	vector<uchar> data_vec(data_buf, data_buf + needRecv);
	imdecode(cv::Mat(data_vec), CV_LOAD_IMAGE_COLOR, &image);
	return rtn_size;
}





class Classifier {
public:
	Classifier(const string& model_file,
		const string& trained_file,
		const string& mean_file,
		const string& label_file);

	std::vector<Prediction> Classify(const cv::Mat& img, int N = 1);

private:
	void SetMean(const string& mean_file);

	std::vector<float> Predict(const cv::Mat& img);

	void WrapInputLayer(std::vector<cv::Mat>* input_channels);

	void Preprocess(const cv::Mat& img,
		std::vector<cv::Mat>* input_channels);

private:
	boost::shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	std::vector<string> labels_;
};

Classifier::Classifier(const string& model_file,
	const string& trained_file,
	const string& mean_file,
	const string& label_file) {
	//#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
	//#else
	 // Caffe::set_mode(Caffe::GPU);
	//#endif

	  /* Load the network. */
	net_.reset(new Net<float>(model_file, TEST));
	net_->CopyTrainedLayersFrom(trained_file);

	CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
	CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

	Blob<float>* input_layer = net_->input_blobs()[0];
	num_channels_ = input_layer->channels();
	CHECK(num_channels_ == 3 || num_channels_ == 1)
		<< "Input layer should have 1 or 3 channels.";
	input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

	/* Load the binaryproto mean file. */
	if (mean_file != "Null")
	{
		SetMean(mean_file);
	}
	else
	{
		cv::Scalar channel_mean = cv::Scalar(103.94, 116.78, 123.68);
		mean_ = cv::Mat(input_geometry_, CV_32FC3, channel_mean);
	}


	/* Load labels. */
	std::ifstream labels(label_file.c_str());
	CHECK(labels) << "Unable to open labels file " << label_file;
	string line;
	while (std::getline(labels, line))
		labels_.push_back(string(line));

	Blob<float>* output_layer = net_->output_blobs()[0];
	CHECK_EQ(labels_.size(), output_layer->channels())
		<< "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], i));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
	std::vector<float> output = Predict(img);

	N = std::min<int>(labels_.size(), N);
	std::vector<int> maxN = Argmax(output, N);
	std::vector<Prediction> predictions;
	for (int i = 0; i < N; ++i) {
		int idx = maxN[i];
		predictions.push_back(std::make_pair(labels_[idx], output[idx]));
	}

	return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
	BlobProto blob_proto;
	ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

	/* Convert from BlobProto to Blob<float> */
	Blob<float> mean_blob;
	mean_blob.FromProto(blob_proto);
	CHECK_EQ(mean_blob.channels(), num_channels_)
		<< "Number of channels of mean file doesn't match input layer.";

	/* The format of the mean file is planar 32-bit float BGR or grayscale. */
	std::vector<cv::Mat> channels;
	float* data = mean_blob.mutable_cpu_data();
	for (int i = 0; i < num_channels_; ++i) {
		/* Extract an individual channel. */
		cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
		channels.push_back(channel);
		data += mean_blob.height() * mean_blob.width();
	}

	/* Merge the separate channels into a single image. */
	cv::Mat mean;
	cv::merge(channels, mean);

	/* Compute the global mean pixel value and create a mean image
	 * filled with this value. */
	cv::Scalar channel_mean = cv::mean(mean);
	mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
	Blob<float>* input_layer = net_->input_blobs()[0];
	input_layer->Reshape(1, num_channels_,
		input_geometry_.height, input_geometry_.width);
	/* Forward dimension change to all layers. */
	net_->Reshape();

	std::vector<cv::Mat> input_channels;
	WrapInputLayer(&input_channels);

	Preprocess(img, &input_channels);

	net_->Forward();

	/* Copy the output layer to a std::vector */
	Blob<float>* output_layer = net_->output_blobs()[0];
	const float* begin = output_layer->cpu_data();
	const float* end = begin + output_layer->channels();
	return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
	Blob<float>* input_layer = net_->input_blobs()[0];

	int width = input_layer->width();
	int height = input_layer->height();
	float* input_data = input_layer->mutable_cpu_data();
	for (int i = 0; i < input_layer->channels(); ++i) {
		cv::Mat channel(height, width, CV_32FC1, input_data);
		input_channels->push_back(channel);
		input_data += width * height;
	}
}

void Classifier::Preprocess(const cv::Mat& img,
	std::vector<cv::Mat>* input_channels) {
	/* Convert the input image to the input image format of the network. */
	cv::Mat sample;
	if (img.channels() == 3 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
	else if (img.channels() == 4 && num_channels_ == 1)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
	else if (img.channels() == 4 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
	else if (img.channels() == 1 && num_channels_ == 3)
		cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
	else
		sample = img;

	cv::Mat sample_resized;
	if (sample.size() != input_geometry_)
		cv::resize(sample, sample_resized, input_geometry_);
	else
		sample_resized = sample;

	cv::Mat sample_float;
	if (num_channels_ == 3)
		sample_resized.convertTo(sample_float, CV_32FC3);
	else
		sample_resized.convertTo(sample_float, CV_32FC1);

	cv::Mat sample_normalized;
	cv::subtract(sample_float, mean_, sample_normalized);

	/* This operation will write the separate BGR planes directly to the
	 * input layer of the network because it is wrapped by the cv::Mat
	 * objects in input_channels. */
	cv::split(sample_normalized, *input_channels);

	CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
		== net_->input_blobs()[0]->cpu_data())
		<< "Input channels are not wrapping the input layer of the network.";
}
template <class T>
struct ItemRepository {
	T item_buffer[kItemRepositorySize];
	//pair<Mat,int> item_buffer[kItemRepositorySize]; // 产品缓冲区, 配合 read_position 和 write_position 模型环形队列.
	size_t read_position; // 消费者读取产品位置.
	size_t write_position; // 生产者写入产品位置.
	std::mutex mtx; // 互斥量,保护产品缓冲区
	std::condition_variable repo_not_full; // 条件变量, 指示产品缓冲区不为满.
	std::condition_variable repo_not_empty; // 条件变量, 指示产品缓冲区不为空.
};
ItemRepository<pair<Mat, int>> gItemRepository; // 产品库全局变量, 生产者和消费者操作该变量.
ItemRepository<int> gItemRepository_to_client;
//typedef struct ItemRepository ItemRepository;
template <class T>
void ProduceItem(ItemRepository<T> *ir, T item)
{
	std::unique_lock<std::mutex> lock(ir->mtx);
	while (((ir->write_position + 1) % kItemRepositorySize)
		== ir->read_position) { // item buffer is full, just wait here.
	//	std::cout << "Producer is waiting for an empty slot...\n";
		(ir->repo_not_full).wait(lock); // 生产者等待"产品库缓冲区不为满"这一条件发生.
	}

	(ir->item_buffer)[ir->write_position] = item; // 写入产品.
	(ir->write_position)++; // 写入位置后移.

	if (ir->write_position == kItemRepositorySize) // 写入位置若是在队列最后则重新设置为初始位置.
		ir->write_position = 0;

	(ir->repo_not_empty).notify_all(); // 通知消费者产品库不为空.
	lock.unlock(); // 解锁.
}
template <class T>
T ConsumeItem(ItemRepository<T> *ir)
{
	T data;
	std::unique_lock<std::mutex> lock(ir->mtx);
	// item buffer is empty, just wait here.
	while (ir->write_position == ir->read_position) {
		//	std::cout << "Consumer is waiting for items...\n";
		(ir->repo_not_empty).wait(lock); // 消费者等待"产品库缓冲区不为空"这一条件发生.
	}

	data = (ir->item_buffer)[ir->read_position]; // 读取某一产品
	(ir->read_position)++; // 读取位置后移

	if (ir->read_position >= kItemRepositorySize) // 读取位置若移到最后，则重新置位.
		ir->read_position = 0;

	(ir->repo_not_full).notify_all(); // 通知消费者产品库不为满.
	lock.unlock(); // 解锁.

	return data; // 返回产品.
}

void ProducerTask(SocketMatTransmissionServer& socketMat) // 生产者任务
{
	pair<Mat, int> picture_num;
	//Mat stream_pic;
	long long  count_pic_num = 0;
	Mat image(IMG_HEIGHT, IMG_WIDTH, CV_8UC3, cv::Scalar(0));
	//VideoCapture capture;
	//capture.open("/home/shunya/Downloads/1/1.avi");
	while (true)
	{
		//capture >> stream_pic;
		//CHECK(!stream_pic.empty()) << "Unable to decode image ";
		if (socketMat.receive_image(image) > 0)
		{

			if(count_pic_num>50)
                                count_pic_num-=50;
			picture_num = make_pair(image, ++count_pic_num);
			//	cout << "produce the " << picture_num.second << " ^th picture" << endl;
			ProduceItem<pair<Mat, int>>(&gItemRepository, picture_num); // 循环生产 kItemsToProduce 

		}
	}
}

int Max(const int temp[])
{
	int a = 0;
	int b = 0;
	for (int i = 0; i < 10; ++i)
	{
		if (a <= temp[i])
		{
			a = temp[i];
			b = i;
		}
	}
	return b;
}
void ConsumerTask(Classifier& classifier) // 消费者任务
{
	vector<Rect> found;
	bool is_first = true;
	cv::Rect showRect;
	Mat frame_to_gray;
	Mat frame;
	Mat frame_rectangle;
	int p_int;
	Rect r;
	int count_pic_num = 0;
	pair<Mat, int> pic_num;
	Prediction p;
	vector<int> result_all;
	vector<int>::iterator iter;
	// Create Tracker object
	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool LAB = false;
	FDSSTTracker Tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
	int flag_temp =0;
	string name__;
	while (true)
	{
		pic_num =ConsumeItem<pair<Mat, int>>(&gItemRepository);
		frame = pic_num.first;
		count_pic_num = pic_num.second;
		cout<<"having gotten frame "<<count_pic_num<<" from queue"<<endl;
		if (is_first)
		{
			mtcnn find(frame.rows, frame.cols);
			r = find.findFace(frame);
			if (r.width == 0 || r.height == 0)
			{
				cout<<"facedetection failed !!!"<<endl;
				char picture_name = count_pic_num + '0';
				name__ = picture_name ;
				name__ += ".jpg";
				imwrite(name__,frame);
				continue;
			}
			cv::Rect_<float> initRect = r;
			//  cv::Mat img;
			cv::cvtColor(frame, frame_to_gray, cv::COLOR_RGB2GRAY);
			Tracker.init(initRect, frame_to_gray);
			cout << "face dection has worked !! " << endl;
			is_first = false;
		}
		else if(count_pic_num < 35)
		{
			cv::cvtColor(frame, frame_to_gray, cv::COLOR_RGB2GRAY);
			showRect = Tracker.update(frame_to_gray);
			//is rec outof range
			if (showRect.x >= 150)
			{
				showRect.x -= 150;
			}
			else
			{
				showRect.x = 0;
			}
			if (showRect.x + showRect.width + 300 > frame.cols)
			{
				showRect.width = frame.cols - showRect.x;
			}
			else
			{
				showRect.width += 300;
			}
			if (showRect.y >= 60)
			{
				showRect.y -= 60;
			}
			else
			{
				showRect.y = 0;
			}
			if (showRect.y + showRect.height + 150 > frame.rows)
			{
				showRect.height = frame.rows - showRect.y;
			}
			else
			{
				showRect.height += 150;
			}

		//	rectangle(frame, showRect, Scalar(0, 255, 0), 6);
		  // 	imshow("picture", frame);
		  // 	waitKey(1);
			frame_rectangle = frame(Rect(showRect.x, showRect.y, showRect.width, showRect.height));
			vector<Prediction> predictions = classifier.Classify(frame_rectangle);
			p = predictions[0];
			p_int = atoi(p.first.c_str());
			cout << "the " << count_pic_num << " ^th picture is recognition !" << endl;
			if (result_all.size() != 15 && flag_temp ==0)
			{ 
				result_all.push_back(p_int);
			}
			else
			{
				if(result_all.size() != 0)
                              {
				flag_temp++;
				int temp[10] = { 0 };
				for (iter = result_all.begin(); iter != result_all.end(); ++iter)
				{
					temp[(*iter)] += 1;

				}
				int max_flag = Max(temp);
				cout << "command control result is : " << max_flag << endl;
				ProduceItem<int>(&gItemRepository_to_client, max_flag);
				result_all.clear();
			     }
			}
		}
		else
		{
                  cout<<"  the "<<count_pic_num << " ^th frame is missed for too much frame being facedetection ! "<<endl;
		}
		if(count_pic_num == 50)
		{
                        is_first = true;
			flag_temp =0;
			cout << "50 number frames have been handled !! wait for next 50 frames and re_face_detection " << endl;
		}
	}
}
		template <class T>
		void InitItemRepository(ItemRepository<T> *ir)
		{
			ir->write_position = 0;
			ir->read_position = 0;
		}

		void Send_Task(SocketMatTransmissionServer& socket_to_send)
		{
			int temp_data;
			int result;
			while (true)
			{
				temp_data = ConsumeItem<int>(&gItemRepository_to_client);
				result = socket_to_send.send_data(&temp_data, sizeof(temp_data));
				if (result < 0)
					cout << "server send error !!! " << endl;
				else
				{
					cout << "send to client successfully !!! " << endl;
				}
			}

		}
		int main(int argc, char** argv) {
			if (argc != 5) {
				std::cerr << "Usage: " << argv[0]
					<< " deploy.prototxt network.caffemodel"
					<< " mean.binaryproto labels.txt img.jpg" << std::endl;
				return 1;
			}

			::google::InitGoogleLogging(argv[0]);

			string model_file = argv[1];
			string trained_file = argv[2];
			string mean_file = argv[3];
			string label_file = argv[4];
			Classifier classifier(model_file, trained_file, mean_file, label_file);

			SocketMatTransmissionServer socketMat;
			if (socketMat.socketConnect(6666) < 0)
			{
				return 0;
			}

			InitItemRepository<pair<Mat, int>>(&gItemRepository);
			InitItemRepository<int>(&gItemRepository_to_client);

			std::thread producer(ProducerTask, std::ref(socketMat));
			producer.detach();
			std::thread consumer(ConsumerTask, std::ref(classifier));
			consumer.detach();
			std::thread sender(Send_Task, std::ref(socketMat));
			sender.join();
			socketMat.socketDisconnect();
		}
#else
int main(int argc, char** argv) {
	LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
#endif  // USE_OPENCV
