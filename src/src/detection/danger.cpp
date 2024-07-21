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
 * @file danger.cpp
 * @author Leo
 * @brief Σ����AIʶ����·���滮
 * @version 0.1
 * @date 2024-01-09
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <fstream>
#include <iostream>
#include <cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "../../include/common.hpp"
#include "../../include/detection.hpp" // Aiģ��Ԥ��

using namespace std;
using namespace cv;

/**
 * @brief Σ����AIʶ����·���滮��
 *
 */
class Danger
{

public:
    /**
     * @brief Σ����AIʶ����·���滮����
     *
     * @param track ����ʶ����
     * @param predict AI�����
     * @return true
     * @return false
     */
    bool process(Tracking& track, vector<PredictResult> predict)
    {
        enable = false; // �������ʹ�ܱ�־
        if (track.pointsEdgeLeft.size() < ROWSIMAGE / 2 || track.pointsEdgeRight.size() < ROWSIMAGE / 2)
            return enable;

        vector<PredictResult> resultsObs; // ׶ͰAI�������
        for (int i = 0; i < predict.size(); i++)
        {
            if ((predict[i].type == LABEL_CONE || predict[i].type == LABEL_BLOCK) && (predict[i].y + predict[i].height) > ROWSIMAGE * 0.1) // AI��־�������
                resultsObs.push_back(predict[i]);
        }

        if (resultsObs.size() <= 0)
            return enable;

        // ѡȡ���������׶Ͱ
        int areaMax = 0; // �����
        int index = 0;   // Ŀ�����
        for (int i = 0; i < resultsObs.size(); i++)
        {
            int area = resultsObs[i].width * resultsObs[i].height;
            if (area >= areaMax)
            {
                index = i;
                areaMax = area;
            }
        }
        resultObs = resultsObs[index];
        enable = true; // �������ʹ�ܱ�־

        // �ϰ��﷽���ж�����/�ң�
        int row = track.pointsEdgeLeft.size() - (resultsObs[index].y + resultsObs[index].height - track.rowCutUp);
        cout << row << endl;
        if (row < 0) // ����滮·��
            return enable;

        int disLeft = resultsObs[index].x + resultsObs[index].width - track.pointsEdgeLeft[row].y;
        int disRight = track.pointsEdgeRight[row].y - resultsObs[index].x;
        if (resultsObs[index].x + resultsObs[index].width > track.pointsEdgeLeft[row].y &&
            track.pointsEdgeRight[row].y > resultsObs[index].x &&
            disLeft <= disRight) //[1] �ϰ��￿��
        {
            if (resultsObs[index].type == LABEL_BLOCK) // ��ɫ·�����⴦��
            {
                curtailTracking(track, false); // �����Ż������ߣ�˫��������������
            }
            else
            {
                vector<POINT> points(4); // ���ױ���������
                points[0] = { track.pointsEdgeLeft[0].x,
                             resultsObs[index].x +  resultsObs[index].width };
                points[1] = { resultsObs[index].y + resultsObs[index].height, resultsObs[index].x + 1.3 * resultsObs[index].width };
                points[2] = { (resultsObs[index].y + resultsObs[index].height + resultsObs[index].y) / 2, resultsObs[index].x + 1.5 * resultsObs[index].width };
                if (resultsObs[index].y > track.pointsEdgeLeft[track.pointsEdgeLeft.size() - 1].x)
                    points[3] = track.pointsEdgeLeft[track.pointsEdgeLeft.size() - 1];
                else
                    points[3] = { resultsObs[index].y, resultsObs[index].x + resultsObs[index].width };

                track.pointsEdgeLeft.clear(); // ɾ������·��
                vector<POINT> repair = Bezier(0.01, points);  // ���¹滮������
                for (int i = 0; i < repair.size(); i++)
                    track.pointsEdgeLeft.push_back(repair[i]);
            }
        }
        else if (resultsObs[index].x + resultsObs[index].width > track.pointsEdgeLeft[row].y &&
            track.pointsEdgeRight[row].y > resultsObs[index].x &&
            disLeft > disRight) //[2] �ϰ��￿��
        {
            if (resultsObs[index].type == LABEL_BLOCK) // ��ɫ·�����⴦��
            {
                curtailTracking(track, true); // �����Ż������ߣ�˫��������������
            }
            else
            {
                vector<POINT> points(4); // ���ױ���������
                points[0] = {track.pointsEdgeRight[0].x, resultsObs[index].x -  resultsObs[index].width };
                points[1] = { resultsObs[index].y + resultsObs[index].height, resultsObs[index].x - 1.3*resultsObs[index].width };
                points[2] = { (resultsObs[index].y + resultsObs[index].height + resultsObs[index].y) / 2, 1.5*resultsObs[index].x - resultsObs[index].width };
                if (resultsObs[index].y > track.pointsEdgeRight[track.pointsEdgeRight.size() - 1].x)
                    points[3] = track.pointsEdgeRight[track.pointsEdgeRight.size() - 1];
                else
                    points[3] = { resultsObs[index].y, resultsObs[index].x };

                track.pointsEdgeRight.resize((size_t)row / 2); // ɾ������·��
                vector<POINT> repair = Bezier(0.01, points);   // ���¹滮������
                for (int i = 0; i < repair.size(); i++)
                    track.pointsEdgeRight.push_back(repair[i]);
            }
        }

        return enable;
    }

    /**
     * @brief ͼ����ƽ�����ʶ����
     *
     * @param img ��Ҫ������ʾ��ͼ��
     */
    void drawImage(Mat& img)
    {
        if (enable)
        {
            putText(img, "[2] DANGER - ENABLE", Point(COLSIMAGE / 2 - 30, 10), cv::FONT_HERSHEY_TRIPLEX, 0.3, cv::Scalar(0, 255, 0), 1, CV_AA);
            cv::Rect rect(resultObs.x, resultObs.y, resultObs.width, resultObs.height);
            cv::rectangle(img, rect, cv::Scalar(0, 0, 255), 1);
        }
    }

private:
    bool enable = false;     // �������ʹ�ܱ�־
    PredictResult resultObs; // ����Ŀ��׶Ͱ

    /**
     * @brief �����Ż������ߣ�˫��������������
     *
     * @param track
     * @param left
     */
    void curtailTracking(Tracking& track, bool left)
    {
        if (left) // ���������
        {
            if (track.pointsEdgeRight.size() > track.pointsEdgeLeft.size())
                track.pointsEdgeRight.resize(track.pointsEdgeLeft.size());

            for (int i = 0; i < track.pointsEdgeRight.size(); i++)
            {
                track.pointsEdgeRight[i].y = (track.pointsEdgeRight[i].y + track.pointsEdgeLeft[i].y) / 2;
            }
        }
        else // ���Ҳ�����
        {
            if (track.pointsEdgeRight.size() < track.pointsEdgeLeft.size())
                track.pointsEdgeLeft.resize(track.pointsEdgeRight.size());

            for (int i = 0; i < track.pointsEdgeLeft.size(); i++)
            {
                track.pointsEdgeLeft[i].y = (track.pointsEdgeRight[i].y + track.pointsEdgeLeft[i].y) / 2;
            }
        }
    }
};
