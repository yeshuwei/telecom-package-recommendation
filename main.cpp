#include <unistd.h>
#include <sys/socket.h>
#include <iostream>
#include "stdarg.h"
#include "slave/slave_adapter.h"
#include "robot_type.h"
#include "tcpsocket.h"
#include "deque"

#include <cmath>
#include <map>
#include <time.h>
#include <thread>
#include "server/server.cpp"


using namespace Eigen;
using namespace std;

#define PI 3.1415926

UR* g_UR = nullptr;
General_6S* g_General_6s = nullptr;
Slave_Adapter *g_slave_adapter = nullptr;



extern tcpsocket* g_tcp;
extern void start_ecm(int argc, char* argv[]);
extern void loop_display();

// 编码器分辨率
const double encoder_resolution = 131072;

// 机械臂物理方向，逆时针为正，顺时针为负
//const int joint_direction[] = {1, 1, -1, -1, 1, 1};
const int joint_direction[] = {1, 1, 1, 1, 1, 1};

typedef void (*FnPtr)(void);
typedef string (*StrFnPtr)(void);
typedef string (*StrFnStrPtr)(string);

string joint_tcp_action(string control_str);

// 机械臂是否在动作
bool robot_is_moving = false;

double joint_add(int axis, double initial, double deg)
{
    return initial + deg * joint_direction[axis];
}



void tcp_server(void)
{
    RbServer server(8000, '>', joint_tcp_action);
}

int cycle_run()
{
	g_General_6s->cycle_run();
	return 0;
}


void app_exit(void)
{
    exit(0);
}

void joint_move()
{
    if (not robot_is_moving)
    {
        // 机器人同时只可执行一次动作
        robot_is_moving = true;
    }

    char joint_info[255];
    cout << "逆时针旋转角度为正，顺时针旋转角度为负，输入角度格式为：轴0-5;角度" << endl;
    cin >> joint_info;
    char joint_axis_str[2];
    char joint_deg_str[255];
    // 读取关节轴
    joint_axis_str[0] = joint_info[0];
    joint_axis_str[1] = '\0';
    // 读取关节角度
    for (int i = 0; i < 255; i++)
    {
        cout << joint_info << endl;
        if ('\0' == joint_info[i + 2])
        {
            joint_deg_str[i] = '\0';
            break;
        }
        joint_deg_str[i] = joint_info[i + 2];
    }
    int axis = atoi(joint_axis_str);
    double deg = atof(joint_deg_str);
    cout << "逆时针旋转角度为正，顺时针旋转角度为负，输入角度格式为：轴;角度" << endl;
    cout << "轴:" << axis << endl;
    cout << "转角:" << deg << endl;
    cout << "是否确定？yes|no" << endl;
    cin >> joint_info;
    if (0 != strcmp(joint_info, "yes"))
    {
        cout << "取消转动操作，程序退出";
        robot_is_moving = false;
        return;
    }

    VectorXd target_point_joint(6);      //目标位置,角度制
    VectorXd origin_point_joint_test(6); //初始位置,角度制
    VectorXd vel_current_joint_test(6);  //当前速度,角度制
    VectorXd acc_current_joint_test(6);  //当前加速度,角度制
    double pos_cur_ang[6];               //当前位置角度值,角度制

    for (int i = 0; i < 6; i++)
    {
        pos_cur_ang[i] = g_General_6s->get_actual_position(i); //获取当前位置角度值
        origin_point_joint_test(i) = pos_cur_ang[i];           //当前位置作为起始位置
        cout << "joint" << i << "--" << pos_cur_ang[i] << endl;
    }

    // 机器人标准初始姿态位置(0位置)
    for (int i = 0; i < 6; i++)
    {
        if (axis == i)
        {
            target_point_joint(i) = joint_add(i, pos_cur_ang[i], deg);
        }
        else
        {
            target_point_joint(i) = pos_cur_ang[i];
        }
    }

    vel_current_joint_test << 0, 0, 0, 0, 0, 0; //设置当前速度
    acc_current_joint_test << 0, 0, 0, 0, 0, 0; //设置当前加速度
    double Ts_joint_test = 0.001;               //设置运动周期
    double velPerc_joint_test = 2;              //设置速度百分比
    double accPerc_joint_test = 2;              //设置加速度百分比
    double decPerc_joint_test = 2;              //设置减速度百分比
    double jerkPerc_joint_test = 2;             //设置雅可比速度百分比
    std::deque<double> trajectory_joint;

    //计算关节插补
    g_General_6s->move_joint_interp(target_point_joint,
                                    origin_point_joint_test, vel_current_joint_test, acc_current_joint_test, Ts_joint_test, velPerc_joint_test,
                                    accPerc_joint_test, decPerc_joint_test, jerkPerc_joint_test, trajectory_joint);


    if (!g_General_6s->get_power_on_status())
    {
        //判断使能状态
        g_General_6s->power_on(); //开启使能
    }
    sleep(2);
    //插补轨迹写入运动队列
    g_General_6s->set_angle_deque(trajectory_joint); //设置运动轨迹

    double cur_angle_double[6];
    while (g_General_6s->get_power_on_status()) //循环检测使能状态
    {
		for(int i=0;i<6;i++)
		{
			cur_angle_double[i] = g_General_6s->get_actual_position(i);  ///获取当前位置角度值
		}
		printf("joint_move_test %lf %lf %lf %lf %lf %lf \n",cur_angle_double[0],cur_angle_double[1],cur_angle_double[2],cur_angle_double[3],cur_angle_double[4],cur_angle_double[5]);
        if (g_General_6s->get_angle_deque().empty() && g_General_6s->get_power_on_status())
        {
            g_General_6s->power_off(); //关闭使能
        }                              //判断运动状态
        sleep(1);
    }
    cout << "joint zero end,current joint reg :" << endl;
    cout << "current joint reg :" << endl;
    for (int i = 0; i < 6; i++)
    {
        cout << "joint" << i << "--" << g_General_6s->get_actual_position(i) << endl;
    }

    robot_is_moving = false;
}

/* 函数功能：运动到零点 */
// 需补充代码
void joint_zero_test()
{
    if (not robot_is_moving)
    {
        // 机器人同时只可执行一次动作
        robot_is_moving = true;
    }
    else
    {
        return;
    }

    VectorXd target_point_joint_test(6);  ///目标位置,角度制
	VectorXd origin_point_joint_test(6);  ///初始位置,角度制
	VectorXd vel_current_joint_test(6);   ///当前速度,角度制
	VectorXd acc_current_joint_test(6);   ///当前加速度,角度制
	
	/***************根据class2作业任务，将空缺代码复制补充***************/
    double pos_cur_ang_zero[6];
    for (int i = 0; i < 6; i++)
    {
        pos_cur_ang_zero[i] = g_General_6s->get_actual_position(i); 
        origin_point_joint_test(i) = pos_cur_ang_zero[i]; 
        cout << "joint" << i << "--" << pos_cur_ang_zero[i] << " deg" << endl;
    }

    for (int i = 0; i < 6; i++)
    {
        target_point_joint_test(i) = 0.0; // 零点位置
    }

    /******************请在以上空间复制补充代码******************/

	vel_current_joint_test<<0,0,0,0,0,0;  ///设置当前速度
	acc_current_joint_test<<0,0,0,0,0,0;  ///设置当前加速度
	double Ts_joint_test = 0.001;  ///设置运动周期
	double velPerc_joint_test = 1;  ///设置速度百分比
	double accPerc_joint_test = 1;  ///设置加速度百分比
	double decPerc_joint_test = 1;  ///设置减速度百分比
	double jerkPerc_joint_test = 1;  ///设置雅可比速度百分比
	std::deque<double> trajectory_joint_test;
	///计算关节插补

	g_General_6s->move_joint_interp(target_point_joint_test,
			origin_point_joint_test, vel_current_joint_test, acc_current_joint_test, Ts_joint_test, velPerc_joint_test,
			accPerc_joint_test, decPerc_joint_test, jerkPerc_joint_test,trajectory_joint_test);

	if(!g_General_6s->get_power_on_status())  ///判断使能状态
		g_General_6s->power_on();  ///开启使能

	sleep(2);

	//插补轨迹写入运动队列
	g_General_6s->set_angle_deque(trajectory_joint_test);  ///设置运动轨迹


 
    while (g_General_6s->get_power_on_status() && !g_General_6s->get_angle_deque().empty()) //循环检测使能状态
    {
        if (g_General_6s->get_angle_deque().empty() && g_General_6s->get_power_on_status())
        {
            g_General_6s->power_off(); //关闭使能
        }                              //判断运动状态
        sleep(1);
    }



    /*************请补充打印当前角度的代码**************/

    cout << "joint zero end,current joint reg :" << endl;
    for (int i = 0; i < 6; i++)
    {
        cout << "joint" << i << "--" << g_General_6s->get_actual_position(i) << endl;
    }

    /**************请在以上空间复制补充代码*************/

    robot_is_moving = false;
}


/* 函数功能：关节角度（单位：度）求取末端笛卡尔坐标（单位：毫米） */
// 需补充代码
void joint_to_cartesian(const VectorXd& posACS,VectorXd& XYZ_position, MatrixXd& transMatrix)
{
	VectorXd pos(3);

    XYZ_position = pos;
   
	XYZ_position[0] = 0;
	XYZ_position[1] = 0;
	XYZ_position[2] = 0;
	
	/***********请根据class2作业任务，将空缺代码复制补充***********/
    g_General_6s->calc_forward_kin(posACS, transMatrix);
    XYZ_position(0) = transMatrix(0, 3);
    XYZ_position(1) = transMatrix(1, 3);
    XYZ_position(2) = transMatrix(2, 3); 
    /***********请在以上空间补充代码***********/

}


/* 函数功能：六轴运动 */
// 需补充代码
void joints_move_test()
{
	if (not robot_is_moving)
    {
        // 机器人同时只可执行一次动作
        robot_is_moving = true;
    }
    else
    {
        return;
    }
	
	VectorXd target_point_joint_test(6);  ///目标位置,角度制
	VectorXd origin_point_joint_test(6);  ///初始位置,角度制


	/***********请根据class2作业任务，将空缺代码复制补充***********/
    double input_deg;
    std::cout << "please enter degree" << std::endl;
    std::cin >> input_deg;
    if (!cin || input_deg < -20 || input_deg > 20) 
    {
        cout << "test end" << endl;
        return; 
    }
    double pos_cur_ang[6];
    for (int i = 0; i < 6; i++)
    {
        pos_cur_ang[i] = g_General_6s->get_actual_position(i);
        origin_point_joint_test(i) = pos_cur_ang[i]; 
    }
    for (int i = 0; i < 6; i++)
    {
        target_point_joint_test(i) = pos_cur_ang[i] + input_deg;
    }











    /***********请在以上空间补充代码***********/



	VectorXd vel_current_joint_test(6);   ///当前速度,角度制
	VectorXd acc_current_joint_test(6);   ///当前加速度,角度制
	vel_current_joint_test<<0,0,0,0,0,0;  ///设置当前速度
	acc_current_joint_test<<0,0,0,0,0,0;  ///设置当前加速度
	double Ts_joint_test = 0.001;  ///设置运动周期
	double velPerc_joint_test = 1;  ///设置速度百分比
	double accPerc_joint_test = 1;  ///设置加速度百分比
	double decPerc_joint_test = 1;  ///设置减速度百分比
	double jerkPerc_joint_test = 1;  ///设置雅可比速度百分比
	std::deque<double> trajectory_joint_test;
	///计算关节插补
	g_General_6s->move_joint_interp(target_point_joint_test,
			origin_point_joint_test, vel_current_joint_test, acc_current_joint_test, Ts_joint_test, velPerc_joint_test,
			accPerc_joint_test, decPerc_joint_test, jerkPerc_joint_test,trajectory_joint_test);


    if (!g_General_6s->get_power_on_status())
    {
        //判断使能状态
        g_General_6s->power_on(); //开启使能
    }
    sleep(2);
    //插补轨迹写入运动队列
    g_General_6s->set_angle_deque(trajectory_joint_test); //设置运动轨迹

    double cur_angle_double[6];
    while (g_General_6s->get_power_on_status() && !g_General_6s->get_angle_deque().empty()) //循环检测使能状态
    {
		for(int i=0;i<6;i++)
		{
			cur_angle_double[i] = g_General_6s->get_actual_position(i);  ///获取当前位置角度值
		}
		printf("joint_move_test %lf %lf %lf %lf %lf %lf \n",cur_angle_double[0],cur_angle_double[1],cur_angle_double[2],cur_angle_double[3],cur_angle_double[4],cur_angle_double[5]);
        if (g_General_6s->get_angle_deque().empty() && g_General_6s->get_power_on_status())
        {
            g_General_6s->power_off(); //关闭使能
        }                              //判断运动状态
        sleep(1);
    }


	/***********请在以下空间补充代码***********/

    joint_zero_test();
    /***********请在以上空间补充代码***********/
    robot_is_moving = false;

}



void joint_current()
{
    cout << "当前角度 deg:" << endl;
    for (int i = 0; i < 6; i++)
    {
        cout << g_General_6s->get_actual_position(i) << "  ";
    }
	cout << endl;
	cout << "当前扭矩 Nm:" << endl;
	for (int i = 0; i < 6; i++)
    {
        cout << g_General_6s->get_actual_torque(i) << "  ";
    }
	cout << endl;
}


void power_on(void)
{

    // 上电
    if (!g_General_6s->get_power_on_status())
    {                             //判断使能状态
        g_General_6s->power_on(); //开启使能
        sleep(2);
    }
}

void power_off(void)
{
    // 下电
    g_General_6s->power_off(); //关闭使能
}

void test_io(void)
{
    if (not robot_is_moving)
    {
        // 机器人同时只可执行一次动作
        robot_is_moving = true;
    }
    else
    {
        return;
    }

    cout << "test_io" << endl;

    Pdo_value pdo_value;
    string i_str;

    // while (true)
    // {
    //     cout << "con str" << endl;
    //     cin >> i_str;
    //     if (strcmp("1", i_str.c_str()) == 0)
    //     {
    //         pdo_value.int_value = 0x200;
    //         g_General_6s->set_pdo_value(0x7011, 6, pdo_value);
    //     }
    //     else if (strcmp("0", i_str.c_str()) == 0)
    //     {
    //         pdo_value.int_value = 0x0;
    //         g_General_6s->set_pdo_value(0x7011, 6, pdo_value);
    //     }
    //     else
    //     {
    //         cout << "break" << endl;
    //         break;
    //     }
    // }



	Pdo_value io_input_value;
	io_input_value.ushort_value = 128;
	printf("******************************\n");
	// for(int i = 0; i < g_General_6s->axis_sum;i++)
	// {
		g_slave_adapter->set_pdo_value(6, 0x7000, io_input_value);
	// }

    robot_is_moving = false;
}


string power_on(string c_r)
{

    // 上电
    if (!g_General_6s->get_power_on_status())
    {                             //判断使能状态
        g_General_6s->power_on(); //开启使能
        sleep(2);
    }
    return string("OK");
}

string power_off(string c_r)
{
    // // 下电
    g_General_6s->power_off(); //关闭使能
    return string("OFF");
}



void joint_cmd_action()
{
    cout << "开启线程,执行任务tcp 任务" << endl;
    thread thread_obj(tcp_server);

    char control_str[100];

    map<string, FnPtr> func_map = {
        // 程序退出
        {"exit", app_exit},
		// 上电
        {"on", power_on},
		// 下电
        {"off", power_off},
        // 输出当前关节角度信息
        {"current", joint_current},
        // 单关节移动
        {"joint_move", joint_move},
		// 机械臂归零，使用五次多项式插值
        {"zero", joint_zero_test},
		// 多关节移动
        {"moves_test", joints_move_test}, 
        
    };

    while (true)
    {
        cout << "请输入控制字：" << endl;
        cin >> control_str;
        if (func_map.count(control_str) == 0)
        {
            cout << "目标控制字不存在，请输入有效的控制字：" << endl;
            continue;
        }
        cout << "执行任务：" << control_str << endl;
        func_map[control_str]();
    }
}






string joint_tcp_action(string control_str)
{
    int split_pos = control_str.find('&');
    string control_command = control_str.substr(0, split_pos);
    string control_info = control_str.substr(split_pos + 1);
    map<string, StrFnStrPtr> func_map = {
        // {"current", get_joints_deg_str},
        // {"goto_position", goto_position},
        // {"get_robot_state", get_robot_state},
        // {"close_grasp", close_grasp},
        // {"open_grasp", open_grasp},
        {"power_on", power_on},
        {"power_off", power_off},
		// {"joint_move",joint_move_test},
    };

    if (func_map.count(control_command) == 0)
    {
        cout << "目标控制字不存在" << control_str << endl;
        return string("目标控制字不存在");
    }
    // cout << "执行任务：" << control_str << endl;
    return func_map[control_command](control_info);
}



void test_general_6s_func()
{
	printf("general_6s function\n");
	///设置DH参数,若机器人轴数小于6，则不需要设置多余轴号的参数
	DH_param dh_example;
	dh_example.a[0] = 0;
	dh_example.a[1] = 295;
	dh_example.a[2] = 37;
	dh_example.a[3] = 0;
	dh_example.a[4] = 0;
	dh_example.a[5] = 0;
	dh_example.alpha[0] = M_PI*90/180;
	dh_example.alpha[1] = M_PI*0/180;
	dh_example.alpha[2] = M_PI*90/180;
	dh_example.alpha[3] = M_PI*90/180;
	dh_example.alpha[4] = M_PI*-90/180;
	dh_example.alpha[5] = M_PI*0/180;
	dh_example.d[0] = 367.5;
	dh_example.d[1] = 0;
	dh_example.d[2] = 0;
	dh_example.d[3] = 295.5;
	dh_example.d[4] = 0;
	dh_example.d[5] = 78.0;
	dh_example.theta[0] = M_PI*0/180;
	dh_example.theta[1] = M_PI*90/180;
	dh_example.theta[2] = M_PI*0/180;
	dh_example.theta[3] = M_PI*0/180;
	dh_example.theta[4] = M_PI*90/180;
	dh_example.theta[5] = M_PI*0/180;
	g_General_6s->set_DH_param(dh_example);

	///设置笛卡尔参数,若机器人轴数小于6，则不需要设置多余轴号的参数
	Decare_Para decare;
	decare.maxacc = 3;
	decare.maxdec = -3;
	decare.maxjerk = 10000;
	decare.maxvel = 1000;
	g_General_6s->set_decare_param(decare);

	///设置电机参数,若机器人轴数小于6，则不需要设置多余轴号的参数
	Motor_Param motor_pa;
	motor_pa.encoder.reducRatio[0] = 81;
	motor_pa.encoder.reducRatio[1] = 101;
	motor_pa.encoder.reducRatio[2] = 63.462;
	motor_pa.encoder.reducRatio[3] = 68.966;
	motor_pa.encoder.reducRatio[4] = 66;
	motor_pa.encoder.reducRatio[5] = 40.625;
	motor_pa.encoder.singleTurnEncoder[0] = 578.089599609375;
	motor_pa.encoder.singleTurnEncoder[1] = -758.652648925781;
	motor_pa.encoder.singleTurnEncoder[2] = 432.81738281250;
	motor_pa.encoder.singleTurnEncoder[3] = 420.455017089844;
	motor_pa.encoder.singleTurnEncoder[4] = 0.035705566406;
	motor_pa.encoder.singleTurnEncoder[5] = 127.878112792969;
	motor_pa.encoder.direction[0] = 1;
	motor_pa.encoder.direction[1] = 1;
	motor_pa.encoder.direction[2] = -1;
	motor_pa.encoder.direction[3] = -1;
	motor_pa.encoder.direction[4] = 1;
	motor_pa.encoder.direction[5] = 1;
	
	for(int i=0;i<6;i++)
	{
		motor_pa.encoder.deviation[i] = 0;
		motor_pa.encoder.encoderResolution[i] = 17;
		motor_pa.RatedVel_rpm[i] = 3000;
		motor_pa.maxAcc[i] = 3;
		motor_pa.maxDecel[i] = -3;
		motor_pa.maxRotSpeed[i] = 2;
		motor_pa.RatedVel[i] = motor_pa.RatedVel_rpm[i] * 6 / motor_pa.encoder.reducRatio[i];
		motor_pa.DeRatedVel[i] = -motor_pa.RatedVel[i];
	}
	g_General_6s->set_motor_param(motor_pa);
	///正解测试
	VectorXd pos_acs(6);

	pos_acs<<0.5,0,0,0,0,0;  ///输入的关节角度,弧度制

	MatrixXd trans_matrix;  ///存储正解矩阵

	g_General_6s->calc_forward_kin(pos_acs,trans_matrix);  ///正解函数
	printf("calcForwardKin transMatrix:\n");  ///正解结果打印在屏幕上
	for(int i=0;i<4;i++)
	{
		for(int j=0;j<4;j++)
		{
			printf("%lf ",trans_matrix(i,j));
		}
		printf("\n");
	}
	printf("\n");
	///逆解测试
	VectorXd pos_result(6);   ///存储逆解结果
	g_General_6s->calc_inverse_kin(trans_matrix,pos_acs,pos_result);  ///逆解函数
	printf("calc_inverse_kin axis pos:\n");  ///逆解结果打印在屏幕上
	for(int i=0;i<6;i++)
		printf("%lf ",pos_result[i]);
	printf("\n");

	int actual_torq[6] = {11111};
	for(int i = 0;i < 6;i++)
	{
		actual_torq[i] = g_General_6s->get_actual_torque(i);
	}
	printf("actual_torq %i %i %i %i %i %i\n",actual_torq[0],actual_torq[1],actual_torq[2],actual_torq[3],actual_torq[4],actual_torq[5]);

	unsigned char setsdovalue = 8;
	unsigned char getsdovalue = 0;
	printf("getsdovalue111 %i\n",getsdovalue);

//	g_slave_adapter->set_sdo_value( 0, 0x6060, 0, (unsigned char *)&setsdovalue, sizeof(EC_T_BYTE));
//	g_slave_adapter->get_sdo_value( 0, 0x6060, 0, (unsigned char *)&getsdovalue, sizeof(unsigned char));
//	printf("getsdovalue222 %i\n",getsdovalue);


    joint_cmd_action();

	//关节运动测试
	// joint_move_test();
//
	///回零点
	// move_to_zero_pos();
//
//	///测试直线插补
//	line_move_test();
}



void* readdata(void* args)
{
	g_tcp = new tcpsocket();
	printf("readdata\n");
	g_tcp->initSocket();
	return 0;
}

void createPthread()
{
	printf("createPthread\n");
    // 定义线程的 id 变量，多个变量使用数组
	pthread_t tid_read;

	//参数依次是：创建的线程id，线程参数，调用的函数，传入的函数参数
	int ret = pthread_create(&tid_read, NULL, readdata, NULL);
	if (ret != 0)
	{
	}
    //等各个线程退出后，进程才结束，否则进程强制结束了，线程可能还没反应过来；
   // pthread_exit(NULL);
}

int CALLBACK()
{
	static int iState = 1;
	printf("iState =======\n");
//	if(iState < 100)
//	{
////		ctrlwdSend(iState);
//		printf("iState +++++++++++++\n");
//		iState++;
//	}
//	else
//	{
//		for(int i = 0; i < g_General_6s->axis_sum; i++)
//		{
//			if(target_position != nullptr)
//			{
//		        iPos[0] += 300;
//		        *target_position = iPos[0];
//			}
//		}
//	}
	return 0;
}

int start_controller()
{
	///初始化robot指针
	g_General_6s = new General_6S(); ///通用六轴模型

//	g_UR = new UR(); ///UR模型
	printf("g_General_6s_ptr %p\n",g_General_6s);

	g_General_6s->slave_num.resize(g_General_6s->axis_sum);
	for(int i = 0;i < g_General_6s->axis_sum;i++)
		g_General_6s->slave_num[i] = i;  ///设置机器人各轴对应的从站序号, 从站序号从0开始

	EC_PF_EC_START_AppWorkpd_CALLBACK p = cycle_run;
	registerCustomeAppWorkpd(p);  ///注册循环执行的函数，每个通讯周期调用一次

	// Pdo_value get_value;
	// get_value.uint_value = 0;
	// sleep(10);
	// printf("222222222222222222222222\n");
	// for(int i = 0; i < g_General_6s->axis_sum;i++)
	// {
	// 	get_value = g_slave_adapter->get_pdo_value(g_General_6s->slave_num[i], 0x603f);
	// 	printf("报错码 : %i \n",get_value.uint_value);
	// }
	sleep(5);


	/*		2021-11-26新增		*/
	//pdo测试
	Pdo_value set_value;
	set_value.uchar_value = 8;
	for(int i = 0; i < g_General_6s->axis_sum;i++)
	{
		g_slave_adapter->set_pdo_value(g_General_6s->slave_num[i], 0x6060, set_value);
	}



	Pdo_value get_value;
	get_value.uchar_value = 5;
	for(int i = 0; i < g_General_6s->axis_sum;i++)
	{
		get_value = g_slave_adapter->get_pdo_value(g_General_6s->slave_num[i], 0x6060);
		printf("%i ",get_value.uchar_value);
		get_value.int_value = 5;
	}
	printf("\n");
	/*		2021-11-26新增结束		*/

	///功能测试函数
	test_general_6s_func(); ///通用六轴模型示例程序
//	test_UR_func(); //////UR模型示例程序

	return 0;
}
/*		2021-11-26新增		*/
void add_pdo_object()
{
	Slave_Adapter::add_pdo(0x6040, pdo_object_type::ushort_); ///控制字
	Slave_Adapter::add_pdo(0x6041, pdo_object_type::ushort_); ///状态字
	Slave_Adapter::add_pdo(0x6060, pdo_object_type::uchar_); ///操作模式
	Slave_Adapter::add_pdo(0x6064, pdo_object_type::int_); ///实际位置
	Slave_Adapter::add_pdo(0x606c, pdo_object_type::int_); ///实际速度
	Slave_Adapter::add_pdo(0x6071, pdo_object_type::short_); ///目标转矩
	Slave_Adapter::add_pdo(0x6077, pdo_object_type::short_); ///实际转矩
	Slave_Adapter::add_pdo(0x607a, pdo_object_type::int_); ///目标位置
	Slave_Adapter::add_pdo(0x603f, pdo_object_type::uint_); ///目标位置

	Slave_Adapter::add_pdo(0x6000, pdo_object_type::ushort_); ///目标位置
	Slave_Adapter::add_pdo(0x7000, pdo_object_type::ushort_); ///目标位置
}

/*		2023-09-26新增		*/

void myprintf(unsigned char c1, const char *s1, const char *s2, const char *s3, const long n, const char *format, ...)
{
	char dest[1024 * 16*16];
	va_list argptr;
	va_start(argptr, format);
	vsprintf(dest, format, argptr);
	va_end(argptr);
	printf(dest);
}

int main()
{
    int nArgc = 1;
    char *argv[8];
    int CycleTime = 1000;

    argv[0]=(char *)(&CycleTime);

	/*		2021-11-26新增		*/
	add_pdo_object();

    enableRealtimeEnvironment();

    // EC_PF_EC_START_CustomeLog_CALLBACK p2 = myprintf;
    // registerCustomeAppLog(p2);

    startEcMaster(nArgc,argv); //igh 主站启动函数
	g_slave_adapter = new Slave_Adapter(); ///创建从站适配器指针
	// createPthread();
	///启动控制器程序
	start_controller();

	//multi_motor_servo_run_test(); //一拖多伺服测试程序

	// test_general_6s_func();

    loop_display();
    return 0;
}
