#include "flight_control_sample.hpp"
#include "camera_gimbal_sample.hpp"
#include "dji_vehicle.hpp"
#include <pthread.h>
#include <stdio.h>
#include <setjmp.h>
#include <malloc.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <errno.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <time.h> //add for clock
#include <signal.h> //add for ctrl+C
#include<math.h>
#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <getopt.h>
#include <ctype.h>
#include <utility>
#include <vector>
#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<string>
#include<sstream>
#include <cstdlib>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <future>
extern "C" {
#include "djicam.h"
}
#define FRAME_SIZE              (1280*720*3/2)  /*format NV12*/
#define BLOCK_MODE               1
#define IMG_WIDTH 1280	
#define IMG_HEIGHT 720
#define BUFFER_SIZE IMG_WIDTH*IMG_HEIGHT*3
#define PI 3.1416
using namespace DJI::OSDK;
using namespace DJI::OSDK::Telemetry;
using namespace std;
using namespace cv;
static unsigned char buffer_NV12[FRAME_SIZE + 8] = { 0 };
static unsigned int nframe = 0;
static int mode = 0;
static const int kItemRepositorySize = 30;


float cur_x = 0;
float cur_y = 0;
float cur_z = 0;
float cur_yaw = 0;
float Angle(float x, float y, float z)
{
	float param = (y*0.75) / (x + 4.0);
	float angle = -atan(param) * 180 / PI;
	return angle;

}
float Pos(float x, float y, float z)
{
	float param = (z*0.3) / (x + 4.0);
	float angle = atan(param) * 180 / PI;
	return angle;

}

class SocketMatTransmissionClient
{
public:
	SocketMatTransmissionClient(void);
	~SocketMatTransmissionClient(void);

private:
	int sockClient;
	//unsigned char buff[BUFFER_SIZE];

public:

	int socketConnect(const char* IP, int PORT);
	int send_image(cv::Mat image);
	void socketDisconnect(void);
	int recv_data(int& result_int);
};
SocketMatTransmissionClient::SocketMatTransmissionClient(void)
{
}


SocketMatTransmissionClient::~SocketMatTransmissionClient(void)
{
}


int SocketMatTransmissionClient::socketConnect(const char* IP, int PORT)
{
	struct sockaddr_in    servaddr;

	if ((sockClient = socket(AF_INET, SOCK_STREAM, 0)) < 0)
	{
		printf("create socket error: %s(errno: %d)\n", strerror(errno), errno);
		return -1;
	}

	memset(&servaddr, 0, sizeof(servaddr));
	servaddr.sin_family = AF_INET;
	servaddr.sin_port = htons(PORT);
	if (inet_pton(AF_INET, IP, &servaddr.sin_addr) <= 0)
	{
		printf("inet_pton error for %s\n", IP);
		return -1;
	}

	if (connect(sockClient, (struct sockaddr*)&servaddr, sizeof(servaddr)) < 0)
	{
		printf("connect error: %s(errno: %d)\n", strerror(errno), errno);
		return -1;
	}
	else
	{
		printf("connect successful!\n");
	}
}


void SocketMatTransmissionClient::socketDisconnect(void)
{
	close(sockClient);
}

int SocketMatTransmissionClient::send_image(cv::Mat image)
{
	if (image.empty())
	{
		printf("empty image\n\n");
		return -1;
	}

	if (image.cols != IMG_WIDTH || image.rows != IMG_HEIGHT || image.type() != CV_8UC3)
	{
		printf("the image must satisfy : cols == IMG_WIDTH（%d）  rows == IMG_HEIGHT（%d） type == CV_8UC3\n\n", IMG_WIDTH, IMG_HEIGHT);
		return -1;
	}

	vector<uchar> data_encode;
	imencode(".jpg", image, data_encode);
	int data_size = data_encode.size();
	//	printf("send size: %d\n", data_size);
	if (send(sockClient, &data_size, sizeof(int), 0) < 0) {
		printf("send error\n");
		return -1;
	}
	if (send(sockClient, &data_encode[0], data_encode.size(), 0) < 0)
	{
		printf("send image error: %s(errno: %d)\n", strerror(errno), errno);
		return -1;
	}
	//printf("%d %d %d %d\n", data_encode[0], data_encode[1], data_encode[2], data_encode[data_encode.size() - 1]);
}

int SocketMatTransmissionClient::recv_data(int& result_int)
{
	int flag;
	flag = recv(sockClient, &result_int, sizeof(result_int), 0);
	if (flag < 0)
		cout << "client recv erro !!! " << endl;
	return flag;
}
static void print_usage(const char *prog)
{
	printf("Usage: sudo %s [-dgt]\n", prog);
	puts("  -d --display    display vedio stream\n"
		"  -g --getbuffer  get NV12 format buffer\n"
		"  -t --transfer   transfer vedio datas to RC\n"
		"  Note: -d and -g cannot be set at the same time\n");
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
// 产品库全局变量, 生产者和消费者操作该变量.
ItemRepository<int> gItemRepository_int;
template <class T>
void ProduceItem(ItemRepository<T> *ir, T item)
{
	std::unique_lock<std::mutex> lock(ir->mtx);
	while (((ir->write_position + 1) % kItemRepositorySize)
		== ir->read_position) { // item buffer is full, just wait here.
		//std::cout << "Producer is waiting for an empty slot...\n";
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
		//std::cout << "Consumer is waiting for items...\n";
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

template <class T>
void InitItemRepository(ItemRepository<T> *ir)
{
	ir->write_position = 0;
	ir->read_position = 0;
}

static void parse_opts(int argc, char **argv)
{
	/*	int c;
	   static const struct option lopts[] = {
		   { "display",   0,0,'d' },
		   { "getbuffer", 0,0,'g' },
		   { "transfer",  0,0,'t' },
		   { NULL,        0,0, 0 },
	   };

	   while ((c = getopt_long(argc, argv, "dgt", lopts, NULL)) != -1)
	   {

		   switch (c)
		   {
		   case 'd':
			   mode |= DISPLAY_MODE;
			   break;
		   case 'g':
			   mode |= GETBUFFER_MODE;
			   break;
		   case 't':
			   mode |= TRANSFER_MODE;
			   break;
		   default:
			   print_usage(argv[0]);
			   exit(0);
		   }

	   } */
	mode |= GETBUFFER_MODE;
	mode |= TRANSFER_MODE;

}


static void *get_images_loop(void *data)
{
	char key_name;
	int ret;
	cv::Mat picBGR;
	cv::Mat picRGB;
	Mat picNV12;
	SocketMatTransmissionClient socketMat = *(SocketMatTransmissionClient*)data;
	long count__ = 1;
	long count_wait = 0;
	bool flag_once = true;
	while (!manifold_cam_exit()) /*Ctrl+c to break out*/
	{
		if (mode & GETBUFFER_MODE)
		{
#if BLOCK_MODE
			ret = manifold_cam_read(buffer_NV12, &nframe, CAM_BLOCK); /*blocking read*/
			if (ret < 0)
			{
				printf("manifold_cam_read error \n");
				break;
			}
			if (count_wait < 18)
			{
				sleep(1);
				count_wait++;
			}
			else
			{

				if (flag_once && (count__ == 1))
				{

					cout << "start to send picture ? y for yes " << endl;
					cin >> key_name;
					flag_once = false;
					continue;

				}
				else
				{
					if (key_name == 'y')
					{
						cout << "begin to send picture !! " << endl;
						if (count__ != 51)
						{

							picNV12 = Mat(720 * 3 / 2, 1280, CV_8UC1, buffer_NV12);
							cvtColor(picNV12, picBGR, CV_YUV2RGB_NV12, 3);
							cvtColor(picBGR, picRGB, CV_BGR2RGB);
							socketMat.send_image(picRGB);
							++count__;
						}
						else
						{
							cout << "50 number frames have been sended !!" << endl;
							flag_once = true;
							count__ = 1;
						}
					}
				}
			}
#else
			ret = manifold_cam_read(buffer, &nframe, CAM_NON_BLOCK); /*non_blocking read*/
			if (ret < 0)
			{
				printf("manifold_cam_read error \n");
				break;
			}
#endif
		}

		usleep(1000);
	}

	socketMat.socketDisconnect();
	cout << "get_images_loop thread exit! \n";
	cout << "socket is disconnect !" << endl;


}

float NEDFromGps(Vehicle* vehicle, Telemetry::Vector3f& deltaNed,
	void* target, void* origin)
{
	Telemetry::GPSFused*       subscriptionTarget;
	Telemetry::GPSFused*       subscriptionOrigin;
	Telemetry::GlobalPosition* broadcastTarget;
	Telemetry::GlobalPosition* broadcastOrigin;
	double                     deltaLon;
	double                     deltaLat;

	if (!vehicle->isM100() && !vehicle->isLegacyM600())
	{
		subscriptionTarget = (Telemetry::GPSFused*)target;
		subscriptionOrigin = (Telemetry::GPSFused*)origin;
		deltaLon = subscriptionTarget->longitude - subscriptionOrigin->longitude;
		deltaLat = subscriptionTarget->latitude - subscriptionOrigin->latitude;
		deltaNed.x = deltaLat * C_EARTH;
		deltaNed.y = deltaLon * C_EARTH * cos(subscriptionTarget->latitude);
		deltaNed.z = subscriptionTarget->altitude - subscriptionOrigin->altitude;
	}
	else
	{
		broadcastTarget = (Telemetry::GlobalPosition*)target;
		broadcastOrigin = (Telemetry::GlobalPosition*)origin;
		deltaLon = broadcastTarget->longitude - broadcastOrigin->longitude;
		deltaLat = broadcastTarget->latitude - broadcastOrigin->latitude;
		deltaNed.x = deltaLat * C_EARTH;
		deltaNed.y = deltaLon * C_EARTH * cos(broadcastTarget->latitude);
		deltaNed.z = broadcastTarget->altitude - broadcastOrigin->altitude;
		return sqrt(deltaNed.x*deltaNed.x + deltaNed.y*deltaNed.y + deltaNed.z*deltaNed.z);
	}
}

void Flight_Camera_Control(Vehicle* vehicle)
{
	int command_int;
	// Global position retrieved via broadcast
	Telemetry::GlobalPosition currentBroadcastGP;
	Telemetry::GlobalPosition originBroadcastGP;

	// Convert position offset from first position to local coordinates
	Telemetry::Vector3f localOffset;

	currentBroadcastGP = vehicle->broadcast->getGlobalPosition();
	originBroadcastGP = currentBroadcastGP;
	float offset_by_UAV = NEDFromGps(vehicle, localOffset, static_cast<void*>(&currentBroadcastGP), static_cast<void*>(&originBroadcastGP));
	//camera
	if (!vehicle->gimbal)
	{
		DERROR("Gimbal object does not exist.\n");
		//return false;
	}

	GimbalContainer              gimbal;
	RotationAngle                initialAngle;
	RotationAngle                currentAngle;
	sleep(1);
	// Get Gimbal initial values
	initialAngle.roll = vehicle->broadcast->getGimbal().roll;
	initialAngle.pitch = vehicle->broadcast->getGimbal().pitch;
	initialAngle.yaw = vehicle->broadcast->getGimbal().yaw;
	std::cout << "Initial Gimbal rotation angle: [" << initialAngle.roll << ", "
		<< initialAngle.pitch << ", " << initialAngle.yaw << "]\n\n";
	gimbal = GimbalContainer(0, 0, 0, 20, 1, initialAngle);
	doSetGimbalAngle(vehicle, &gimbal);
	cout << "start to flight_camera control !!! " << endl;

	while (fabs(offset_by_UAV) < 40)
	{
		command_int = ConsumeItem<int>(&gItemRepository_int);
		cout << "receive connmand : " << command_int << endl;
		switch (command_int)
		{
		case 0:
			cout << "command is None ,just do nothing !!" << endl;
			usleep(1000);
			break;
		case 1:
			cout << "command is Up(1) ,now fly upward !!" << endl;
			++cur_z;
			moveByPositionOffset(vehicle, 0, 0, 1, cur_yaw);
			cur_yaw = Angle(cur_x, cur_y, cur_z);
			moveByPositionOffset(vehicle, 0, 0, 0, cur_yaw);
			gimbal.pitch = -Pos(cur_x, cur_y, cur_z) * 10;
			doSetGimbalAngle(vehicle, &gimbal);
			usleep(1000);
			break;
		case 2:
			cout << "command is down(2) ,now fly downward !!" << endl;
			--cur_z;
			moveByPositionOffset(vehicle, 0, 0, -1, cur_yaw);
			cur_yaw = Angle(cur_x, cur_y, cur_z);
			moveByPositionOffset(vehicle, 0, 0, 0, cur_yaw);
			gimbal.pitch = -Pos(cur_x, cur_y, cur_z) * 10;
			doSetGimbalAngle(vehicle, &gimbal);
			usleep(1000);
			break;
		case 3:
			cout << "command is left(3) ,now fly left !!" << endl;
			++cur_y;
			moveByPositionOffset(vehicle, 0, 1, 0, cur_yaw);
			cur_yaw = Angle(cur_x, cur_y, cur_z);
			moveByPositionOffset(vehicle, 0, 0, 0, cur_yaw);
			gimbal.pitch = -Pos(cur_x, cur_y, cur_z) * 10;
			doSetGimbalAngle(vehicle, &gimbal);
			usleep(1000);
			break;
		case 4:
			cout << "command is right(4) ,now fly right !!" << endl;
			--cur_y;
			moveByPositionOffset(vehicle, 0, -1, 0, cur_yaw);
			cur_yaw = Angle(cur_x, cur_y, cur_z);
			moveByPositionOffset(vehicle, 0, 0, 0, cur_yaw);
			gimbal.pitch = -Pos(cur_x, cur_y, cur_z) * 10;
			doSetGimbalAngle(vehicle, &gimbal);
			usleep(1000);
			break;
		case 5:
			cout << "command is forward(5) ,now fly forward !!" << endl;
			++cur_x;
			moveByPositionOffset(vehicle, 1, 0, 0, cur_yaw);
			cur_yaw = Angle(cur_x, cur_y, cur_z);
			moveByPositionOffset(vehicle, 0, 0, 0, cur_yaw);
			gimbal.pitch = -Pos(cur_x, cur_y, cur_z) * 10;
			doSetGimbalAngle(vehicle, &gimbal);
			usleep(1000);
			break;
		case 6:
			cout << "command is back(6) ,now fly back !!" << endl;
			--cur_x;
			moveByPositionOffset(vehicle, -1, 0, 0, cur_yaw);
			cur_yaw = Angle(cur_x, cur_y, cur_z);
			moveByPositionOffset(vehicle, 0, 0, 0, cur_yaw);
			gimbal.pitch = -Pos(cur_x, cur_y, cur_z) * 10;
			doSetGimbalAngle(vehicle, &gimbal);
			usleep(1000);
			break;
		case 7:
			cout << "command is stop(7) ,now do nothing !!" << endl;
			usleep(1000);
			break;
		}
		if (!vehicle->isM100() && !vehicle->isLegacyM600())
		{
			currentAngle.roll = vehicle->subscribe->getValue<TOPIC_GIMBAL_ANGLES>().y;
			currentAngle.pitch = vehicle->subscribe->getValue<TOPIC_GIMBAL_ANGLES>().x;
			currentAngle.yaw = vehicle->subscribe->getValue<TOPIC_GIMBAL_ANGLES>().z;
		}
		else
		{
			currentAngle.roll = vehicle->broadcast->getGimbal().roll;
			currentAngle.pitch = vehicle->broadcast->getGimbal().pitch;
			currentAngle.yaw = vehicle->broadcast->getGimbal().yaw;
		}

		displayResult(&currentAngle);
		currentBroadcastGP = vehicle->broadcast->getGlobalPosition();
		offset_by_UAV = NEDFromGps(vehicle, localOffset, static_cast<void*>(&currentBroadcastGP), static_cast<void*>(&originBroadcastGP));
	}
}

void RecvTask(SocketMatTransmissionClient& socketMat)
{
	int flag = 1;
	int result_int;
	while (flag)
	{
		flag = socketMat.recv_data(result_int);
		ProduceItem<int>(&gItemRepository_int, result_int);

	}

}


int main(int argc, char **argv)
{
	// Initialize variables
	int functionTimeout = 1;

	// Setup OSDK.
	LinuxSetup linuxEnvironment(argc, argv);
	Vehicle*   vehicle = linuxEnvironment.getVehicle();
	if (vehicle == NULL)
	{
		std::cout << "Vehicle not initialized, exiting.\n";
		return -1;
	}

	// Obtain Control Authority
	vehicle->obtainCtrlAuthority(functionTimeout);

	int ret;
	pthread_attr_t attr;
	struct sched_param schedparam;
	pthread_t read_thread;

	SocketMatTransmissionClient socketMat;
	if (socketMat.socketConnect("192.168.1.129", 6666) < 0)
	{
		return -1;
	}
	InitItemRepository<int>(&gItemRepository_int);

	if (0 != geteuid())
	{
		printf("Please run ./test as root!\n");
		print_usage(argv[0]);
		return -1;
	}
	parse_opts(argc, argv); /*get parameters*/
	if (0 == mode || 3 == mode || 7 == mode) /*invalid mode*/
	{
		print_usage(argv[0]);
		return -1;
	}
	ret = manifold_cam_init(mode);
	if (-1 == ret)
	{
		printf("manifold init error \n");
		return -1;
	}


	/*
	* if the cpu usage is high, the scheduling policy of the read thread
	* is recommended setting to FIFO, and also, the priority of the thread should be high enough.
	*/
	pthread_attr_init(&attr);
	pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED);
	pthread_attr_setschedpolicy((pthread_attr_t *)&attr, SCHED_FIFO);
	schedparam.sched_priority = 90;
	pthread_attr_setschedparam(&attr, &schedparam);
	pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);

	if (pthread_create(&read_thread, &attr, get_images_loop, &socketMat) != 0)
	{
		perror("usbRead_thread create");
		assert(0);
	}

	if (pthread_attr_destroy(&attr) != 0)
	{
		perror("pthread_attr_destroy error");
	}
	monitoredTakeoff(vehicle);

	std::thread producer(RecvTask, std::ref(socketMat));
	std::thread consumer_flight_camera_contral(Flight_Camera_Control, vehicle);



	pthread_join(read_thread, NULL);/*wait for read_thread exit*/

	producer.join();
	consumer_flight_camera_contral.join();

	monitoredLanding(vehicle);

	while (!manifold_cam_exit())
	{
		sleep(1);
	}

	return 0;
}
