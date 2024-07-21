#include <fstream>
#include <iostream>
#include <cmath>
#include "../include/common.hpp"
#include "../include/json.hpp"
#include "controlcenter.cpp"
#include "FuzzyPID.cpp"
using namespace std;
using namespace cv;

/**
 * @brief 运动控制器
 *
 */
class Motion
{
private:
    int countShift = 0; // 变速计数器
    std::vector<double> errors;
    //FuzzyPID fuzzy;

public:
    /**
     * @brief 初始化：加载配置文件
     *
     */
    Motion()
    {
        string jsonPath = "../src/config/config.json";
        std::ifstream config_is(jsonPath);//打开json文件
        if (!config_is.good())
        {
            std::cout << "Error: Params file path:[" << jsonPath
                      << "] not find .\n";
            exit(-1);
        }

        nlohmann::json js_value;
        config_is >> js_value;

        try
        {
            params = js_value.get<Params>();
        }
        catch (const nlohmann::detail::exception &e)
        {
            std::cerr << "Json Params Parse failed :" << e.what() << '\n';
            exit(-1);
        }

        speed = params.speedLow;
        cout <<"speedhigh: " << params.speedHigh;
        //FuzzyPID pid;
        cout << "EFF_P :";
        for (int i = 0; i < params.EFF_P.size(); i++)
        {
            cout << params.EFF_P[i] << " ";
        }
        cout << endl;

        cout << "DFF_P :";
        for (int i = 0; i < params.DFF_P.size(); i++)
        {
            cout << params.DFF_P[i] << " ";
        }
        cout << endl;

        cout << "UFF_P :";
        for (int i = 0; i < params.UFF_P.size(); i++)
        {
            cout << params.UFF_P[i] << " ";
        }
        cout << endl;

        cout << "EFF_D :";
        for (int i = 0; i < params.EFF_D.size(); i++) {
            cout << params.EFF_D[i] << " ";
        }
        cout << endl;

        cout << "DFF_D :";
        for (int i = 0; i < params.DFF_D.size(); i++) {
            cout << params.DFF_D[i] << " ";
        }
        cout << endl;

        cout << "UFF_D :";
        for (int i = 0; i < params.UFF_D.size(); i++) {
            cout << params.UFF_D[i] << " ";
        }
        cout << endl;
    };

    /**
     * @brief 控制器核心参数
     *
     */
    struct Params
    {
        float up_speed=0;
        float down_speed=0;
        float speedLow = 0.8;                              // 智能车最低速
        float speedHigh = 0.8;                             // 智能车最高速
        float speedBridge = 0.6;                           // 坡道速度
        float speedDown = 0.5;                             // 特殊区域降速速度
        float runP1 = 0;                                 // 一阶比例系数：直线控制量
        float runP2 = 0;
        float runP3 = 0.0;                                 // 三阶比例系数：弯道控制量
        float runP4=0;
        float runP=0.0;
        float prospect=0.0;
        float turnD1 = 0;
        float turnD2= 0;
        float turnD3 = 0;                                 // 一阶微分系数：转弯控制量
        float turnD=0;
        vector<float> EFF_P = { -56, -20, -10, 0, 10, 20, 56 }; // Error范围
        /*输入量D语言值特征点*/
        vector<float> DFF_P = { -2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5 }; // 误差变化率范围
        /*输出量U语言值特征点(根据赛道类型选择不同的输出值)*/
        vector<float> UFF_P = { 1.7, 1.8, 1.9, 2.0, 2.1, 2.8, 3 };

        vector<float> EFF_D = { -56, -20, -10, 0, 10, 25, 56 };
        /*输入量D语言值特征点*/
        vector<float> DFF_D = { -2.5, -1.5, -0.5, 0, 0.5, 1.5, 2 };
        /*输出量U语言值特征点(根据赛道类型选择不同的输出值)*/
        vector<float> UFF_D = { 2.3, 4.2, 8.2, 7, 6, 5, 4 };
        int servoOffset = 0;    //舵机偏移量
        int outRate = 1;    //控制台输出倍率
        bool debug = true;                                // 调试模式使能
        bool saveImg = false;                              // 存图使能
        uint16_t rowCutUp = 10;                            // 图像顶部切行
        uint16_t rowCutBottom = 10;                        // 图像低部切行
        bool bridge = true;                                // 坡道区使能
        bool danger = true;                                // 危险区使能
        bool rescue = true;                                // 救援区使能
        bool racing = true;                                // 追逐区使能
        bool parking = true;                               // 停车区使能
        bool debug_fps=true;                               //帧率测试使能
        bool debug_uart=true;                              //串口发送使能
        bool ring = true;                                  // 环岛使能
        bool cross = true;                                 // 十字道路使能
        float score = 0.5;                                 // AI检测置信度
        string model = "../res/model/yolov3_mobilenet_v1"; // 模型路径
        string video = "../res/samples/sample.mp4";          // 视频路径
        NLOHMANN_DEFINE_TYPE_INTRUSIVE(Params,up_speed,down_speed, speedLow, speedHigh, speedBridge, speedDown, runP1, runP2, runP3,runP4,
                                       runP,prospect,turnD1,turnD2,turnD3, turnD, debug, saveImg, rowCutUp, rowCutBottom, bridge, danger,
                                       rescue, racing, parking, debug_fps,debug_uart,ring, cross, score, model, video, EFF_P, DFF_P, UFF_P, EFF_D,
                                        DFF_D, UFF_D,servoOffset,
            outRate); // 添加构造函数
    };

    Params params;                   // 读取控制参数
    uint16_t servoPwm = PWMSERVOMID; // 发送给舵机的PWM
    float speed = 0.3;               // 发送给电机的速度
    int pwm_Diff;
    int errp_ppre = 0;  
    /**
     * @brief 姿态PD控制器
     *
     * @param controlCenter 智能车控制中心
     */
    void poseCtrl(int controlCenter)
    {
        float error = controlCenter - COLSIMAGE / 2; // 图像控制中心转换偏差
                //std::cout<<error<<std::endl;
        static int errorLast = 0;                    // 记录前一次的偏差    
        if (abs(error - errorLast) > COLSIMAGE / 10)
        {
            //std::cout<<"进去了"<<error<<"\t"<<errorLast<<std::endl;
            error = error > errorLast ? errorLast + COLSIMAGE / 10 : errorLast - COLSIMAGE / 10;
        }
        // 每收集10个误差样本后进行一次PID参数的自适应调
        // if(speed==params.speedLow){
        //     errors.push_back(std::abs(error));
        //     if (errors.size() >= 10) {
        //         adaptPIDGains();
        //     }
        // }
        // std::cout<<"\tp"<<params.runP1;
        // std::cout<<"\tp2"<<params.runP2;
        // std::cout<<"\td"<<params.turnD1<<std::endl;
        if(speed==params.speedDown){
            params.runP = abs(error) * params.runP2;
            int pwmDiff = (error * params.runP) + (error - errorLast) * params.turnD3;
            if(abs(pwmDiff)>300){
                if(pwmDiff>0)
                    pwmDiff=300;
                else
                    pwmDiff=-300;
            }
             servoPwm = (uint16_t)(PWMSERVOMID + pwmDiff);
        }
        else if(speed==params.speedHigh){ 
            params.runP = params.runP1;
            int pwmDiff = (error * params.runP) + (error - errorLast) * params.turnD1;
            if(abs(pwmDiff)>320){
                speed=params.speedLow-0.3;
                if(pwmDiff>0)
                    pwmDiff=320;
                else
                    pwmDiff=-320;
            }
            servoPwm = (uint16_t)(PWMSERVOMID + pwmDiff); 
        }
        else{
            params.runP = abs(error) *  params.runP4 + params.runP3;
            int pwmDiff = (error * params.runP) + (error - errorLast) * params.turnD2;
            if(abs(pwmDiff)>320){
                speed=params.speedLow-0.3;
                if(pwmDiff>0)
                    pwmDiff=320;
                else
                    pwmDiff=-320;
            }
            servoPwm = (uint16_t)(PWMSERVOMID + pwmDiff); 
        }
        //std::cout<<servoPwm<<std::endl;
        errp_ppre=errorLast;
        errorLast = error;
    }

    void poseCtrl1(int controlCenter, double frametime) {
        float error = controlCenter - COLSIMAGE / 2; // 图像控制中心转换偏差
        static int errorLast = 0;               // 记录前一次的偏差
        if (abs(error - errorLast) > COLSIMAGE / 10) {
            error = error > errorLast ? errorLast + COLSIMAGE / 10
                : errorLast - COLSIMAGE / 10;
        }
        float error_change_rate = ((error - errorLast) / frametime) * 35;
        float kp = Fuzzy_P(error, error_change_rate);
        float kd = Fuzzy_D(error, error_change_rate);
        float pwmDiff = kp * error + kd * error_change_rate;
        int derror = error - errorLast;
        errorLast = error;
        printf("error: %d derror: %d\n", error, derror);
        //cout << error << ' ' << derror << endl;
        //cout << frametime << " ms" << " " << error_change_rate << endl;
        servoPwm = (uint16_t)((PWMSERVOMID + pwmDiff) + params.servoOffset); // PWM转换
        //cout << (servoPwm - 1500) * params.outRate << endl;
    }


    float Fuzzy_P(float E, float EC) {
        /*输入量P语言值特征点*/
        // 6.14注 eff dff uff需要做成json便于调试
        //  1.15 2  2.2 2.4 2.6 2.8 3 4}
        //  float UFF[7] = {2.0, 2.1, 2.2, 2.4, 2.8, 2.8, 3};
        int rule[7][7] = {
            {6, 5, 4, 3, 2, 1, 0}, // 规则表，不建议修改
            {5, 4, 3, 2, 1, 0, 1}, 
            {4, 3, 2, 1, 0, 1, 2}, 
            {3, 2, 1, 0, 1, 2, 3},
            {2, 1, 0, 1, 2, 3, 4}, 
            {1, 0, 1, 2, 3, 4, 5}, 
            {0, 1, 2, 3, 4, 5, 6},
        };

        float U = 0; /*偏差,偏差微分以及输出值的精确量*/
        float PF[2] = { 0 }, DF[2] = { 0 },
            UF[4] = { 0 }; // 偏差,偏差微分以及输出值的隶属度

        int Pn = 0, Dn = 0, Un[4] = { 0 };
        float t1 = 0, t2 = 0, t3 = 0, t4 = 0, temp1 = 0, temp2 = 0;
        /*隶属度的确定*/
        /*根据PD的指定语言值获得有效隶属度*/
        if (E > params.EFF_P[0] && E < params.EFF_P[6]) {
            if (E <= params.EFF_P[1]) {
                Pn = -2;
                PF[0] = (params.EFF_P[1] - E) / (params.EFF_P[1] - params.EFF_P[0]);
            }
            else if (E <= params.EFF_P[2]) {
                Pn = -1;
                PF[0] = (params.EFF_P[2] - E) / (params.EFF_P[2] - params.EFF_P[1]);
            }
            else if (E <= params.EFF_P[3]) {
                Pn = 0;
                PF[0] = (params.EFF_P[3] - E) / (params.EFF_P[3] - params.EFF_P[2]);
            }
            else if (E <= params.EFF_P[4]) {
                Pn = 1;
                PF[0] = (params.EFF_P[4] - E) / (params.EFF_P[4] - params.EFF_P[3]);
            }
            else if (E <= params.EFF_P[5]) {
                Pn = 2;
                PF[0] = (params.EFF_P[5] - E) / (params.EFF_P[5] - params.EFF_P[4]);
            }
            else if (E <= params.EFF_P[6]) {
                Pn = 3;
                PF[0] = (params.EFF_P[6] - E) / (params.EFF_P[6] - params.EFF_P[5]);
            }
        }

        else if (E <= params.EFF_P[0]) {
            Pn = -2;
            PF[0] = 1;
        }
        else if (E >= params.EFF_P[6]) {
            Pn = 3;
            PF[0] = 0;
        }

        PF[1] = 1 - PF[0];

        // 判断D的隶属度
        if (EC > params.DFF_P[0] && EC < params.DFF_P[6]) {
            if (EC <= params.DFF_P[1]) {
                Dn = -2;
                DF[0] = (params.DFF_P[1] - EC) / (params.DFF_P[1] - params.DFF_P[0]);
            }
            else if (EC <= params.DFF_P[2]) {
                Dn = -1;
                DF[0] = (params.DFF_P[2] - EC) / (params.DFF_P[2] - params.DFF_P[1]);
            }
            else if (EC <= params.DFF_P[3]) {
                Dn = 0;
                DF[0] = (params.DFF_P[3] - EC) / (params.DFF_P[3] - params.DFF_P[2]);
            }
            else if (EC <= params.DFF_P[4]) {
                Dn = 1;
                DF[0] = (params.DFF_P[4] - EC) / (params.DFF_P[4] - params.DFF_P[3]);
            }
            else if (EC <= params.DFF_P[5]) {
                Dn = 2;
                DF[0] = (params.DFF_P[5] - EC) / (params.DFF_P[5] - params.DFF_P[4]);
            }
            else if (EC <= params.DFF_P[6]) {
                Dn = 3;
                DF[0] = (params.DFF_P[6] - EC) / (params.DFF_P[6] - params.DFF_P[5]);
            }
        }
        // 不在给定的区间内
        else if (EC <= params.DFF_P[0]) {
            Dn = -2;
            DF[0] = 1;
        }
        else if (EC >= params.DFF_P[6]) {
            Dn = 3;
            DF[0] = 0;
        }

        DF[1] = 1 - DF[0];

        /*使用误差范围优化后的规则表rule[7][7]*/
        /*输出值使用13个隶属函数,中心值由UFF[7]指定*/
        /*一般都是四个规则有效*/
        Un[0] = rule[Pn + 2][Dn + 2];
        Un[1] = rule[Pn + 3][Dn + 2];
        Un[2] = rule[Pn + 2][Dn + 3];
        Un[3] = rule[Pn + 3][Dn + 3];

        UF[0] = PF[0] * DF[0];
        UF[1] = PF[0] * DF[1];
        UF[2] = PF[1] * DF[0];
        UF[3] = PF[1] * DF[1];
        t1 = UF[0] * params.UFF_P[Un[0]];
        t2 = UF[1] * params.UFF_P[Un[1]];
        t3 = UF[2] * params.UFF_P[Un[2]];
        t4 = UF[3] * params.UFF_P[Un[3]];
        temp1 = t1 + t2 + t3 + t4;
        temp2 = UF[0] + UF[1] + UF[2] + UF[3]; // 模糊量输出
        U = temp1 / temp2;
        return U;
    }

    float Fuzzy_D(float E, float EC) {
        /*输入量P语言值特征点*/
        // float UFF[7] = {0.6, 0.9, 1.2, 8, 7, 6, 5};

        int rule[7][7] = {
            {0, 1, 2, 3, 4, 5, 6}, {1, 2, 3, 4, 5, 6, 5}, {2, 3, 4, 5, 6, 5, 4},
            {3, 4, 5, 6, 5, 4, 3}, {4, 5, 6, 5, 4, 3, 2}, {5, 6, 5, 4, 3, 2, 1},
            {6, 5, 4, 3, 2, 1, 0},
        };

        float U = 0; /*偏差,偏差微分以及输出值的精确量*/
        float PF[2] = { 0 }, DF[2] = { 0 }, UF[4] = { 0 };
        /*偏差,偏差微分以及输出值的隶属度*/
        int Pn = 0, Dn = 0, Un[4] = { 0 };
        float t1 = 0, t2 = 0, t3 = 0, t4 = 0, temp1 = 0, temp2 = 0;
        /*隶属度的确定*/
        /*根据PD的指定语言值获得有效隶属度*/
        if (E > params.EFF_D[0] && E < params.EFF_D[6]) {
            if (E <= params.EFF_D[1]) {
                Pn = -2;
                PF[0] = (params.EFF_D[1] - E) / (params.EFF_D[1] - params.EFF_D[0]);
            }
            else if (E <= params.EFF_D[2]) {
                Pn = -1;
                PF[0] = (params.EFF_D[2] - E) / (params.EFF_D[2] - params.EFF_D[1]);
            }
            else if (E <= params.EFF_D[3]) {
                Pn = 0;
                PF[0] = (params.EFF_D[3] - E) / (params.EFF_D[3] - params.EFF_D[2]);
            }
            else if (E <= params.EFF_D[4]) {
                Pn = 1;
                PF[0] = (params.EFF_D[4] - E) / (params.EFF_D[4] - params.EFF_D[3]);
            }
            else if (E <= params.EFF_D[5]) {
                Pn = 2;
                PF[0] = (params.EFF_D[5] - E) / (params.EFF_D[5] - params.EFF_D[4]);
            }
            else if (E <= params.EFF_D[6]) {
                Pn = 3;
                PF[0] = (params.EFF_D[6] - E) / (params.EFF_D[6] - params.EFF_D[5]);
            }
        }

        else if (E <= params.EFF_D[0]) {
            Pn = -2;
            PF[0] = 1;
        }
        else if (E >= params.EFF_D[6]) {
            Pn = 3;
            PF[0] = 0;
        }

        PF[1] = 1 - PF[0];

        // 判断D的隶属度
        if (EC > params.DFF_D[0] && EC < params.DFF_D[6]) {
            if (EC <= params.DFF_D[1]) {
                Dn = -2;
                DF[0] = (params.DFF_D[1] - EC) / (params.DFF_D[1] - params.DFF_D[0]);
            }
            else if (EC <= params.DFF_D[2]) {
                Dn = -1;
                DF[0] = (params.DFF_D[2] - EC) / (params.DFF_D[2] - params.DFF_D[1]);
            }
            else if (EC <= params.DFF_D[3]) {
                Dn = 0;
                DF[0] = (params.DFF_D[3] - EC) / (params.DFF_D[3] - params.DFF_D[2]);
            }
            else if (EC <= params.DFF_D[4]) {
                Dn = 1;
                DF[0] = (params.DFF_D[4] - EC) / (params.DFF_D[4] - params.DFF_D[3]);
            }
            else if (EC <= params.DFF_D[5]) {
                Dn = 2;
                DF[0] = (params.DFF_D[5] - EC) / (params.DFF_D[5] - params.DFF_D[4]);
            }
            else if (EC <= params.DFF_D[6]) {
                Dn = 3;
                DF[0] = (params.DFF_D[6] - EC) / (params.DFF_D[6] - params.DFF_D[5]);
            }
        }
        // 不在给定的区间内
        else if (EC <= params.DFF_D[0]) {
            Dn = -2;
            DF[0] = 1;
        }
        else if (EC >= params.DFF_D[6]) {
            Dn = 3;
            DF[0] = 0;
        }

        DF[1] = 1 - DF[0];

        /*使用误差范围优化后的规则表rule[7][7]*/
        /*输出值使用13个隶属函数,中心值由UFF[7]指定*/
        /*一般都是四个规则有效*/
        Un[0] = rule[Pn + 2][Dn + 2];
        Un[1] = rule[Pn + 3][Dn + 2];
        Un[2] = rule[Pn + 2][Dn + 3];
        Un[3] = rule[Pn + 3][Dn + 3];

        Un[0] = rule[Pn + 2][Dn + 2];
        Un[1] = rule[Pn + 3][Dn + 2];
        Un[2] = rule[Pn + 2][Dn + 3];
        Un[3] = rule[Pn + 3][Dn + 3];

        UF[0] = PF[0] * DF[0];
        UF[1] = PF[0] * DF[1];
        UF[2] = PF[1] * DF[0];
        UF[3] = PF[1] * DF[1];
        t1 = UF[0] * params.UFF_D[Un[0]];
        t2 = UF[1] * params.UFF_D[Un[1]];
        t3 = UF[2] * params.UFF_D[Un[2]];
        t4 = UF[3] * params.UFF_D[Un[3]];
        temp1 = t1 + t2 + t3 + t4;
        temp2 = UF[0] + UF[1] + UF[2] + UF[3]; // 模糊量输出
        U = temp1 / temp2;
        return U;
    }

    /**
     * @brief 变加速控制
     *
     * @param enable 加速使能
     * @param control
     */
    void speedCtrl(bool enable, bool slowDown, ControlCenter control)
    {
        // 控制率
        uint8_t controlLow = 0;   // 速度控制下限
        uint8_t controlMid = 5;   // 控制率
        uint8_t controlHigh = 10; // 速度控制上限

        if (slowDown)
        {
            countShift = controlLow;
            speed = params.speedDown;
        }
        else if (enable) // 加速使能
        {
            if (control.centerEdge.size() < 10)
            {
                speed = params.speedLow;
                countShift = controlLow;
                return;
            }
            if (control.centerEdge[control.centerEdge.size() - 1].x > ROWSIMAGE / 2)
            {
                speed = params.speedLow;
                countShift = controlLow;
                return;
            }
            if (abs(control.sigmaCenter) <150.0)
            {
                countShift++;
                if (countShift > controlHigh)
                    countShift = controlHigh;
            }
            else if (abs(control.sigmaCenter)>2000.0)
            {
                speed = params.speedLow-0.4;
                return;
            }
            else
            {
                countShift--;
                if (countShift < controlLow)
                    countShift = controlLow;
            }

            if (countShift > controlMid)
                speed = params.speedHigh;
            else
                speed = params.speedLow;
        }
        else
        {
            countShift = controlLow;
            speed = params.speedLow;
        }
    }
    void adaptPIDGains() {
        double errorSum = std::accumulate(errors.begin(), errors.end(), 0.0);
        double averageError = errorSum / errors.size();
        double maxError = *std::max_element(errors.begin(), errors.end());
        double minError = *std::min_element(errors.begin(), errors.end());

        // 调整 params.runP1
        if (averageError > 50) {
            params.runP1 *= 1.05;  // 如果平均误差较大，增加 runP1
        } else if (averageError < 20) {
            params.runP1 *= 0.95;  // 如果平均误差较小，减少 runP1
        }

        // 调整 params.runP2
        if (maxError > 50) {
            params.runP2 *= 1.1;  // 如果最大误差过大，增加 runP2
        } else if (minError < 20) {
            params.runP2 *= 0.9;  // 如果最小误差过小，减少 runP2
        }

        // 调整 params.turnD1
        if (averageError > 40) {
            params.turnD1 *= 1.025;  // 如果平均误差较大，增加 turnD1
        } else if (averageError < 10) {
            params.turnD1 *= 0.975;  // 如果平均误差较小，减少 turnD1
        }

        errors.clear();  // 清空误差列表以便重新收集数据
    }
    float speed_control(float target_speed,float new_speed){
        if (target_speed >= 2.5) target_speed = 2.5;
        if(target_speed>=params.speedHigh){
            if (new_speed == 0) {  
                float real_speed = 0.3;
                return real_speed;
            }
            else {
                float real_speed = new_speed * params.up_speed;
                if (real_speed > params.speedHigh)
                    real_speed = params.speedHigh;
                    return real_speed;
            }
            
        }
        else{
            
            return target_speed;
        }
    }
};
