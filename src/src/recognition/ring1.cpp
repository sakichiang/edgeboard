#pragma once
/**
 ********************************************************************************************************
 *                                               ʾ������
 *                                             EXAMPLE  CODE
 *
 *                      (c) Copyright 2024; SaiShu.Lcc.; Leo; https://bjsstech.com
 *                                   ��Ȩ����[SASU-��������Ƽ����޹�˾]
 *
 *            The code is for internal use only, not for commercial transactions(��Դѧϰ,��������).
 *            The code ADAPTS the corresponding hardware circuit board(��������ٶ�Edgeboard-�����������°�),
 *            The specific details consult the professional(��ӭ��ϵ����,������������������ע��ؿ�Դ����).
 *********************************************************************************************************
 * @file ring.cpp
 * @author Leo
 * @brief ����ʶ�𣨻���track����ʶ���
 * @version 0.1
 * @date 2022-02-28
 *
 * @copyright Copyright (c) 2022
 *
 * @note  ����ʶ���裨ringStep����
 *          1������ʶ�𣨳�ʼ����
 *          2���뻷����
 *          3�����д���
 *          4����������
 *          5����������
 */

#include <fstream>
#include <iostream>
#include <cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "../../include/common.hpp"
#include "tracking.cpp"

using namespace cv;
using namespace std;

class Ring
{
public:
    uint16_t counterShield = 0; // ����������μ����������γ�������

    /**
     * @brief ����ʶ���ʼ��|��λ
     *
     */
    void reset(void)
    {
        RingType ringType = RingType::RingLeft; // ��������
        RingStep ringStep = RingStep::None;     // ��������׶�
        int rowRepairLine = 0;                  // ���ڻ����ߵĵ㣨�кţ�
        int colRepairLine = 0;                  // ���ڻ����ߵĵ㣨�кţ�
        counterSpurroad = 0;                    // ��·������
        counterShield = 0;
    }
    /**
     * @brief ����ʶ�����о��滮
     *
     * @param track ��������ʶ����
     * @param imagePath ����·��ͼ��
     */
    bool process(Tracking& track, Mat& imagePath, Motion& motion)
    {
        if (counterShield < 100) // ȷ���ڻ�����ⱻ�������һ��ʱ���ڲ����ٴδ���
        {
            counterShield++;
            return false;
        }

        bool ringEnable = false;                                 // �л���־
        RingType ringTypeTemp = RingType::RingNone;              // �������ͣ���ʱ����
        int rowBreakpointLeft = 0;                               // ��Ե�յ���ʼ�У���
        int rowBreakpointRight = 0;                              // ��Ե�յ���ʼ�У��ң�
        int colEnterRing = 0;                                    // �뻷�㣨ͼ������ţ�
        int rowRepairRingside = track.widthBlock.size() - 1;     // ��һ�࣬������㣨�кţ�
        int rowRepairStraightside = track.widthBlock.size() - 1; // ֱ���࣬������㣨�кţ�
        int rowYendStraightside = track.widthBlock.size() - 1;   // ֱ���࣬�ӳ������յ㣨�кţ�
        _index = 0;
        _ringPoint = POINT(0, 0);

        // �㻷�ò��ߵĺ�ѡ��
        rowRepairLine = max(rowRepairLine - 5, 0);
        if (ringStep == RingStep::Entering && !track.spurroad.empty())
        {
            if (ringType == RingType::RingLeft && track.pointsEdgeLeft.size() > 20)
            {
                for (int j = max(rowRepairLine - 30, 10);
                    j < track.pointsEdgeLeft.size() - 10 && j < rowRepairLine + 30 &&
                    track.pointsEdgeLeft[j].x >= track.spurroad[0].x;
                    j++)
                {
                    if (track.pointsEdgeLeft[j].y > track.pointsEdgeLeft[j - 10].y &&
                        track.pointsEdgeLeft[j].y > track.pointsEdgeLeft[j + 10].y)
                    {
                        rowRepairLine = j;
                        break;
                    }
                }
            }
            else if (ringType == RingType::RingRight && track.pointsEdgeRight.size() > 20)
            {
                for (int j = max(rowRepairLine - 30, 10);
                    j < track.pointsEdgeRight.size() - 10 && j < rowRepairLine + 30 &&
                    track.pointsEdgeRight[j].x >= track.spurroad[0].x;
                    j++)
                {
                    if (track.pointsEdgeRight[j].y < track.pointsEdgeRight[j - 10].y &&
                        track.pointsEdgeRight[j].y < track.pointsEdgeRight[j + 10].y)
                    {
                        rowRepairLine = j;
                        break;
                    }
                }
            }
        }

        // �����������ұ�Ե����ͼ����ص���ߴ�
        for (int ii = 0; ii < track.pointsEdgeLeft.size(); ++ii)
        {
            rowBreakpointLeft = track.pointsEdgeLeft[ii].x;
            if (track.pointsEdgeLeft[ii].y > 2)
                break;
        }
        for (int ii = 0; ii < track.pointsEdgeRight.size(); ++ii)
        {
            rowBreakpointRight = track.pointsEdgeRight[ii].x;
            if (track.pointsEdgeRight[ii].y < COLSIMAGE - 3)
                break;
        }

        // �л�
        int countWide = 0; // ������ڱ����������



        //�°�����·��Ľǵ�����Ѱ��
        //for (int i = 1; i < track.spurroad[track.spurroad.size()-1].x+4; ++i)
        // ԭ������ɨ��
        for (int i = 1; i < track.widthBlock.size(); ++i)
        {
            if (track.widthBlock[i].y > track.widthBlock[i - 1].y && track.widthBlock[i].y > COLSIMAGE * 0.6 && track.widthBlock[i].x > 30 &&
                ((track.stdevLeft > 120 && track.stdevRight < 50) || ringStep == RingStep::Entering)) // ����ͻȻ����·������
            {
                ++countWide;
            }
            else
            {
                countWide = 0;
            }
            // [1] �뻷�ж�
            if ((ringStep == RingStep::None || ringStep == RingStep::Entering) && countWide >= 3 && !track.spurroad.empty())
            {
                if (ringTypeTemp == RingType::RingNone) // ���������ж�
                {
                    int tmp_flag = 0;
                    for (int j = 0; j < track.spurroad.size(); j++)
                    {
                        if (track.spurroad[j].x < track.pointsEdgeLeft[i - 5].x)
                        {
                            tmp_flag = 1;
                        }
                    }
                    if (tmp_flag == 0)
                    {
                        countWide = 0;
                        continue;
                    }
                    if (track.pointsEdgeLeft[i].y < track.pointsEdgeLeft[i - 5].y)
                    {
                        ringTypeTemp = RingType::RingLeft;            // �������ͣ����뻷
                        colEnterRing = track.pointsEdgeLeft[i - 5].y; // �뻷���к�
                        _ringPoint.x = track.pointsEdgeLeft[i - 5].x;
                        _ringPoint.y = track.pointsEdgeLeft[i - 5].y;

                        rowRepairLine = i;                         // ���ڻ����ߵ��к�
                        colRepairLine = track.pointsEdgeLeft[i].x; // ���ڻ����ߵ��к�
                    }
                    else if (track.pointsEdgeRight[i].y > track.pointsEdgeRight[i - 5].y)
                    {
                        ringTypeTemp = RingType::RingRight;            // �������ͣ����뻷
                        colEnterRing = track.pointsEdgeRight[i - 5].y; // �뻷���к�
                        rowRepairLine = i;                             // ���ڻ����ߵ��к�
                        colRepairLine = track.pointsEdgeRight[i].x;    // ���ڻ����ߵ��к�
                    }
                }

                // ��Բ���
                if ((ringTypeTemp == RingType::RingLeft && colEnterRing - track.pointsEdgeLeft[i].y >= 3) ||
                    (ringTypeTemp == RingType::RingRight && track.pointsEdgeRight[i].y - colEnterRing >= 3))
                {
                    ringEnable = true;
                    ringStep = RingStep::Entering;
                    ringType = ringTypeTemp;
                    if (rowRepairStraightside == track.widthBlock.size() - 1)
                    {
                        rowRepairStraightside = i - countWide;
                    }
                }
                else
                {
                    countWide = 0;
                }
            }
            /*if(ringStep == RingStep::Entering && ringEnable == false){
                ringEnable = true;
                rowRepairStraightside = rowRepairLine;
            }*/

            if (ringEnable == true && ringStep == RingStep::Entering)
            {
                if (ringTypeTemp == RingType::RingLeft)
                {
                    if (track.pointsEdgeLeft[i].y <= 2 && i != track.widthBlock.size() - 1)
                    {
                        if (rowRepairRingside == track.widthBlock.size() - 1)
                        {
                            rowRepairRingside = i;
                        }
                        rowYendStraightside = track.pointsEdgeLeft[i].x;
                    }
                    else if (rowRepairRingside != track.widthBlock.size() - 1)
                    {

                        int x = track.pointsEdgeLeft[rowRepairStraightside].x +
                            (rowYendStraightside -
                                track.pointsEdgeRight[rowRepairStraightside].x) *
                            5 / 4;
                        int y = (track.pointsEdgeLeft[rowRepairStraightside].y +
                            track.pointsEdgeRight[rowRepairStraightside].y) /
                            2;

                        POINT startPoint = track.pointsEdgeRight[rowRepairStraightside]; // ���ߣ����
                        POINT midPoint(x, y);                                            // ���ߣ��е�
                        POINT endPoint(rowYendStraightside, 0);                          // ���ߣ��յ�

                        vector<POINT> input = { startPoint, midPoint, endPoint };
                        vector<POINT> b_modify = Bezier(0.01, input);
                        track.pointsEdgeLeft.resize(rowRepairRingside);
                        track.pointsEdgeRight.resize(rowRepairStraightside);
                        for (int kk = 0; kk < b_modify.size(); ++kk)
                        {
                            track.pointsEdgeRight.emplace_back(b_modify[kk]);
                        }
                        break;
                    }
                }
                //�һ�
                else {
                    if (track.pointsEdgeRight[i].y <= 2 &&
                        i != track.widthBlock.size() - 1) {
                        if (rowRepairRingside == track.widthBlock.size() - 1) {
                            rowRepairRingside = i;
                        }
                        rowYendStraightside = track.pointsEdgeRight[i].x;
                    }
                    else if (rowRepairRingside != track.widthBlock.size() - 1) {
                        int x = track.pointsEdgeRight[rowRepairStraightside].x +
                            (rowYendStraightside -
                                track.pointsEdgeLeft[rowRepairStraightside].x) *
                            5 / 4;
                        int y = (track.pointsEdgeRight[rowRepairStraightside].y +
                            track.pointsEdgeLeft[rowRepairStraightside].y) /
                            2;

                        POINT startPoint =
                            track.pointsEdgeLeft
                            [rowRepairStraightside]; // ���ߣ����
                        POINT midPoint(x, y);            // ���ߣ��е�
                        POINT endPoint(rowYendStraightside,
                            COLSIMAGE - 1); // ���ߣ��յ�

                        vector<POINT> input = { startPoint, midPoint, endPoint };
                        vector<POINT> b_modify = Bezier(0.01, input);
                        track.pointsEdgeLeft.resize(rowRepairRingside);
                        track.pointsEdgeRight.resize(rowRepairStraightside);
                        for (int kk = 0; kk < b_modify.size(); ++kk) {
                            track.pointsEdgeLeft.emplace_back(b_modify[kk]);
                        }
                        break;
                    }
                }

            }
        }

        int tmp_ttttt = 0;
        if (ringEnable == false && ringStep == RingStep::Entering)
        {
            // ����û�г�������û�зֲ�
            if (!track.spurroad.empty() && rowRepairLine < track.pointsEdgeRight.size() - 1 && rowBreakpointRight > ROWSIMAGE / 2)
            {

                rowRepairStraightside = rowRepairLine;

                if (ringType == RingType::RingLeft)
                {
                    tmp_ttttt = 1;
                    for (int i = rowRepairLine; i < track.pointsEdgeLeft.size() - 1; i++)
                    {
                        if (track.pointsEdgeLeft[i].y <= 2 && i != track.widthBlock.size() - 1)
                        {
                            rowRepairRingside = i;
                            break;
                            // rowYendStraightside = track.pointsEdgeLeft[i].x;
                        }
                    }

                    for (int i = rowRepairRingside; i < track.pointsEdgeLeft.size() - 1; i++)
                    {
                        if (track.pointsEdgeLeft[i].y <= 2 && i != track.widthBlock.size() - 1)
                        {
                            rowYendStraightside = track.pointsEdgeLeft[i].x;
                        }
                        else if (rowRepairRingside != track.widthBlock.size() - 1)
                        {
                            int x = track.pointsEdgeLeft[rowRepairStraightside].x + (rowYendStraightside - track.pointsEdgeRight[rowRepairStraightside].x) * 5 / 4;
                            int y = (track.pointsEdgeLeft[rowRepairStraightside].y + track.pointsEdgeRight[rowRepairStraightside].y) / 2;

                            POINT startPoint = track.pointsEdgeRight[rowRepairStraightside]; // ���ߣ����
                            POINT midPoint(x, y);                                            // ���ߣ��е�
                            POINT endPoint(rowYendStraightside, 0);                          // ���ߣ��յ�

                            // for (int i = 0; i < track.spurroad.size(); i++)
                            // {
                            //     if (track.spurroad[i].y < startPoint.y && track.spurroad[i].x < startPoint.x)
                            //         endPoint = track.spurroad[i];
                            //     break;
                            // }

                            vector<POINT> input = { startPoint, midPoint, endPoint };
                            vector<POINT> b_modify = Bezier(0.02, input);
                            track.pointsEdgeLeft.resize(rowRepairRingside);
                            track.pointsEdgeRight.resize(rowRepairStraightside);

                            for (int kk = 0; kk < b_modify.size(); ++kk)
                            {
                                track.pointsEdgeRight.emplace_back(b_modify[kk]);
                            }
                            break;
                        }
                    }
                }
            }
            // ����û�г������зֲ�
            else
            {
                if (ringType == RingType::RingLeft && track.pointsEdgeRight.size() > 1)
                {
                    tmp_ttttt = 2;
                    int x_end = track.pointsEdgeRight[track.pointsEdgeRight.size() - 1].x;
                    for (int kkk = track.pointsEdgeRight[track.pointsEdgeRight.size() - 1].x; kkk < track.pointsEdgeRight[track.pointsEdgeRight.size() - 1].x + 50; kkk++)
                    {
                        if (imagePath.at<Vec3b>(kkk, 0)[2] > 0)
                        {
                            x_end = kkk;
                            break;
                        }
                    }

                    POINT startPoint(ROWSIMAGE - 10, COLSIMAGE - 1); // ���ߣ����
                    POINT endPoint(x_end, 0);                        // ���ߣ��յ�

                    // for (int i = 0; i < track.spurroad.size(); i++)
                    // {
                    //     if (track.spurroad[i].y < startPoint.y && track.spurroad[i].x < startPoint.x)
                    //         endPoint = track.spurroad[i];
                    //     break;
                    // }
                    POINT midPoint = POINT((startPoint.x + endPoint.x) * 0.5, (startPoint.y + endPoint.y) * 0.5); // ���ߣ��е�
                    vector<POINT> input = { startPoint, midPoint, endPoint };
                    vector<POINT> b_modify = Bezier(0.02, input);
                    track.pointsEdgeRight.resize(0);
                    track.pointsEdgeLeft.resize(0);
                    for (int kk = 0; kk < b_modify.size(); ++kk)
                    {
                        track.pointsEdgeRight.emplace_back(b_modify[kk]);
                    }
                }
            }
        }
        // ����
        if (ringStep == RingStep::Entering && track.spurroad.empty() && counterSpurroad >= 3)
        {
            ringStep = RingStep::Inside;
        }
        // ��������
        if (ringStep == RingStep::Inside)
        {
            if (ringType == RingType::RingLeft)
            {
                int rowBreakRight = 0; // �ұ�Ե������������(�к�)
                for (int i = 0; i < track.pointsEdgeRight.size(); i += 3)
                {
                    if (track.pointsEdgeRight[i].y <= track.pointsEdgeRight[rowBreakRight].y)
                    {
                        rowBreakRight = i;
                        continue;
                    }
                    if (i > rowBreakRight && track.pointsEdgeRight[i].y - track.pointsEdgeRight[rowBreakRight].y > 5)
                    {
                        rowBreakpointRight = rowBreakRight;
                        break; // Ѱ�ҵ������ڣ���������
                    }
                }
                track.pointsEdgeLeft.resize(0); // ���߿���
                int acute_angle_flag = 0;
                if (!track.pointsEdgeRight.empty() && track.pointsEdgeRight[rowBreakRight].y < COLSIMAGE / 4)
                {
                    track.pointsEdgeRight.resize(rowBreakRight); // ǰ80�в���Ҫ����
                }
                else if (track.pointsEdgeRight.size() - rowBreakRight > 20)
                {
                    float slopeTop = 0;    // б�ʣ�������ϰ벿��
                    float slopeButtom = 0; // б�ʣ�������°벿��
                    if (track.pointsEdgeRight[rowBreakRight].x != track.pointsEdgeRight[0].x)
                    {
                        slopeButtom = (track.pointsEdgeRight[rowBreakRight].y - track.pointsEdgeRight[0].y) * 100 /
                            (track.pointsEdgeRight[rowBreakRight].x - track.pointsEdgeRight[0].x);
                    }
                    if (track.pointsEdgeRight[rowBreakRight].x != track.pointsEdgeRight[rowBreakRight + 20].x)
                    {
                        slopeTop = (track.pointsEdgeRight[rowBreakRight + 20].y - track.pointsEdgeRight[rowBreakRight].y) *
                            100 / (track.pointsEdgeRight[rowBreakRight + 20].x - track.pointsEdgeRight[rowBreakRight].x);
                    }

                    if (slopeButtom * slopeTop <= 0)
                    {
                        rowBreakpointLeft = track.pointsEdgeRight[track.validRowsLeft].x;
                        POINT p_end(rowBreakpointLeft, 0); // �����յ�Ϊ�����Ч�ж���
                        POINT p_mid((track.pointsEdgeRight[rowBreakRight].x + rowBreakpointLeft) * 3 / 8, track.pointsEdgeRight[rowBreakRight].y / 2);
                        vector<POINT> input = { track.pointsEdgeRight[rowBreakRight], p_mid, p_end };
                        vector<POINT> b_modify = Bezier(0.01, input);
                        track.pointsEdgeRight.resize(rowBreakRight);
                        for (int kk = 0; kk < b_modify.size(); ++kk)
                        {
                            track.pointsEdgeRight.emplace_back(b_modify[kk]);
                        }
                    }
                }
                else if (track.pointsEdgeRight.size() - rowBreakRight <= 20)
                {
                    _index = 2;
                    POINT p_end(rowBreakpointLeft, 0);
                    POINT p_start(max(rowBreakpointRight, ROWSIMAGE - 80), COLSIMAGE);
                    POINT p_mid((ROWSIMAGE - 50 + rowBreakpointLeft) / 4, COLSIMAGE / 2);
                    vector<POINT> input = { p_start, p_mid, p_end };
                    vector<POINT> b_modify = Bezier(0.01, input);
                    track.pointsEdgeRight.resize(0);
                    for (int kk = 0; kk < b_modify.size(); ++kk)
                    {
                        track.pointsEdgeRight.emplace_back(b_modify[kk]);
                    }
                }
            }
            else
            {
                ;
            }
            if (max(rowBreakpointLeft, rowBreakpointRight) < ROWSIMAGE / 2)
            {
                ringStep = RingStep::Exiting;
            }
        }
        // �������
        else if (ringStep == RingStep::Exiting)
        {
            if (ringType == RingType::RingLeft && rowBreakpointLeft < ROWSIMAGE / 2)
            {
                POINT p_end(rowBreakpointLeft, 0);
                POINT p_start(ROWSIMAGE - 50, COLSIMAGE - 1);
                POINT p_mid((ROWSIMAGE - 50 + rowBreakpointLeft) * 3 / 8, COLSIMAGE / 2);
                vector<POINT> input = { p_start, p_mid, p_end };
                vector<POINT> b_modify = Bezier(0.01, input);
                track.pointsEdgeRight.resize(0);
                track.pointsEdgeLeft.resize(0);
                for (int kk = 0; kk < b_modify.size(); ++kk)
                {
                    track.pointsEdgeRight.emplace_back(b_modify[kk]);
                }
                if (rowBreakpointRight > ROWSIMAGE / 2)
                {
                    ringStep = RingStep::Finish;
                }
            }
        }

        // //����߽��edge��
        // vector<POINT> v_temp, v_temp2;
        // for (int jj = 0; jj < track.pointsEdgeLeft.size(); ++jj)
        // {
        //     if (track.pointsEdgeLeft[jj].y > 2)
        //     {
        //         v_temp.push_back(track.pointsEdgeLeft[jj]);
        //     }
        //     else
        //     {
        //         if (jj > track.pointsEdgeLeft.size() * 9 / 10)
        //         {
        //             break;
        //         }
        //     }

        //     if (track.pointsEdgeLeft[jj].y > COLSIMAGE * 9 / 10 && jj < track.pointsEdgeLeft.size() - 5)
        //     {
        //         break;
        //     }
        // }
        // track.pointsEdgeLeft = v_temp;
        // if (track.pointsEdgeLeft.size() < 5)
        // {
        //     track.pointsEdgeLeft.resize(0);
        // }

        // for (int jj = 0; jj < track.pointsEdgeRight.size(); ++jj)
        // {
        //     if (track.pointsEdgeRight[jj].y < COLSIMAGE - 3)
        //     {
        //         v_temp2.push_back(track.pointsEdgeRight[jj]);
        //     }
        //     else
        //     {
        //         if (jj > track.pointsEdgeRight.size() * 9 / 10)
        //         {
        //             break;
        //         }
        //     }
        //     if (track.pointsEdgeRight[jj].y < COLSIMAGE / 10 && jj < track.pointsEdgeRight.size() - 5)
        //     {
        //         break;
        //     }
        // }
        // track.pointsEdgeRight = v_temp2;
        // if (track.pointsEdgeRight.size() < 5)
        // {
        //     track.pointsEdgeRight.resize(0);
        // }

        // �������л�����ѭ��
        if (ringStep == RingStep::Finish)
        {
            if (track.pointsEdgeLeft.size() > 30 && track.pointsEdgeRight.size() > 30 &&
                abs(track.pointsEdgeRight.size() - track.pointsEdgeLeft.size() < track.pointsEdgeRight.size() / 4) &&
                track.spurroad.empty())
            {
                ringStep = RingStep::None;
                reset();
                motion.params.ring = false;

            }
        }

        if (track.spurroad.empty())
            counterSpurroad++;
        else
            counterSpurroad = 0;

        //--------------------------------��ʱ����----------------------------------
        _ringStep = ringStep;
        _ringEnable = ringEnable;
        _tmp_ttttt = tmp_ttttt;

        // ����ʶ����
        if (ringStep == RingStep::None)
            return false;
        else
            return true;
    }

    /**
     * @brief ���ƻ���ʶ��ͼ��
     *
     * @param ringImage ��Ҫ������ʾ��ͼ��
     */
    void drawImage(Tracking track, Mat& ringImage)
    {
        for (int i = 0; i < track.pointsEdgeLeft.size(); i++)
        {
            circle(ringImage, Point(track.pointsEdgeLeft[i].y, track.pointsEdgeLeft[i].x), 2,
                Scalar(0, 255, 0), -1); // ��ɫ��
        }
        for (int i = 0; i < track.pointsEdgeRight.size(); i++)
        {
            circle(ringImage, Point(track.pointsEdgeRight[i].y, track.pointsEdgeRight[i].x), 2,
                Scalar(0, 255, 255), -1); // ��ɫ��
        }

        for (int i = 0; i < track.spurroad.size(); i++)
        {
            circle(ringImage, Point(track.spurroad[i].y, track.spurroad[i].x), 5,
                Scalar(0, 0, 255), -1); // ��ɫ��
        }

        putText(ringImage, to_string(_ringStep) + " " + to_string(_ringEnable) + " " + to_string(_tmp_ttttt),
            Point(COLSIMAGE - 80, ROWSIMAGE - 20), cv::FONT_HERSHEY_TRIPLEX, 0.3, cv::Scalar(0, 0, 255), 1, CV_AA);

        putText(ringImage, to_string(_index), Point(80, ROWSIMAGE - 20), cv::FONT_HERSHEY_TRIPLEX, 0.3, cv::Scalar(0, 0, 255), 1, CV_AA);

        putText(ringImage, to_string(track.validRowsRight) + " | " + to_string(track.stdevRight),
            Point(COLSIMAGE - 100, ROWSIMAGE - 50),
            FONT_HERSHEY_TRIPLEX, 0.3, Scalar(0, 0, 255), 1, CV_AA);
        putText(ringImage, to_string(track.validRowsLeft) + " | " + to_string(track.stdevLeft),
            Point(30, ROWSIMAGE - 50), FONT_HERSHEY_TRIPLEX, 0.3, Scalar(0, 0, 255), 1, CV_AA);

        putText(ringImage, "[7] RING - ENABLE", Point(COLSIMAGE / 2 - 30, 10), cv::FONT_HERSHEY_TRIPLEX, 0.3, cv::Scalar(0, 255, 0), 1, CV_AA);
        circle(ringImage, Point(_ringPoint.y, _ringPoint.x), 4, Scalar(255, 0, 0), -1); // ��ɫ��
    }

private:
    uint16_t counterSpurroad = 0; // ��·������
    // ��ʱ�����ò���
    int _ringStep;
    int _ringEnable;
    int _tmp_ttttt;
    int _index = 0;
    POINT _ringPoint = POINT(0, 0);

    /**
     * @brief ��������
     *
     */
    enum RingType
    {
        RingNone = 0, // δ֪����
        RingLeft,     // ���뻷��
        RingRight     // ���뻷��
    };

    /**
     * @brief �������в���/�׶�
     *
     */
    enum RingStep
    {
        None = 0, // δ֪����
        Entering, // �뻷
        Inside,   // ����
        Exiting,  // ����
        Finish    // ���������
    };

    RingType ringType = RingType::RingLeft; // ��������
    RingStep ringStep = RingStep::None;     // ��������׶�
    int rowRepairLine = 0;                  // ���ڻ����ߵĵ㣨�кţ�
    int colRepairLine = 0;                  // ���ڻ����ߵĵ㣨�кţ�
};
