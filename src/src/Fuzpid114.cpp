#include <iostream>
#include <cmath>
#include <vector>
#include <numeric> // �������ڼ����ۻ��͵ĺ���

class FuzzyPID1 {
private:
    //ceshi
public:
    int e_membership_values[7] = { -90,-60,-30,0,30,60,90 }; //����e������ֵ
    int ec_membership_values[7] = { -12,-8,-4,0,4,8,12 };//����de/dt������ֵ
    float kp_menbership_values[7] = { -0.12,-0.08,-0.04,0,0.04,0.08,0.012 };//�������kp������ֵ
    float kd_menbership_values[7] = { -0.12,-0.08,-0.04,0,0.04,0.08,0.012 };  //�������kd������ֵ
    const int  num_area = 8;
    float e_gradmembership[2];      //����e��������
    float ec_gradmembership[2];     //����de/dt��������
    int e_grad_index[2];            //����e�������ڹ���������
    int ec_grad_index[2];           //����de/dt�������ڹ���������
    float qerro;                    //����e��Ӧ�����е�ֵ
    float qerro_c;                  //����de/dt��Ӧ�����е�ֵ
    float kp = 1.2;
    float kd = 1;
    float qdetail_kp;               //����kp��Ӧ�����е�ֵ
    float qdetail_kd;               //����kd��Ӧ�����е�ֵ
    float detail_kp;                //�������kp
    float detail_kd;                //�������kd
    int NB = -24, NM = -16, NS = -8, ZO = 0, PS = 8, PM = 16, PB = 24; //��������ֵ
    float KpgradSums[7] = { 0,0,0,0,0,0,0 };   //�������kp�ܵ�������
    float KdgradSums[7] = { 0,0,0,0,0,0,0 };   //�������kd�ܵ�������
    int  Kp_rule_list[7][7] = { {PB,PB,PM,PM,PS,ZO,ZO},     //kp�����
                                {PB,PB,PM,PS,PS,ZO,NS},
                                {PM,PM,PM,PS,ZO,NS,NS},
                                {PM,PM,PS,ZO,NS,NM,NM},
                                {PS,PS,ZO,NS,NS,NM,NM},
                                {PS,ZO,NS,NM,NM,NM,NB},
                                {ZO,ZO,NM,NM,NM,NB,NB} };

    int  Kd_rule_list[7][7] = { {PS,NS,NB,NB,NB,NM,PS},    //kd�����
                                {PS,NS,NB,NM,NM,NS,ZO},
                                {ZO,NS,NM,NM,NS,NS,ZO},
                                {ZO,NS,NS,NS,NS,NS,ZO},
                                {ZO,ZO,ZO,ZO,ZO,ZO,ZO},
                                {PB,NS,PS,PS,PS,PS,PB},
                                {PB,PM,PM,PM,PS,PS,PB} };
    float Inverse_quantization(float maximum, float minimum, float qvalues)
    {
        //��������ǰ�qvalues��ֵ�԰ٷֱȵ���ʽ(qvalues + 240)/480ӳ�䵽maximun��minimum�ϣ����ǵ��øú��������ȫ��ע����
        float x = (maximum - minimum) * (qvalues + 240) / 480 + minimum;
        return x;
    }

    void Get_grad_membership(int error, int error_c)
    {
        //std::cout<<erro<<"\t"<<erro_c<<std::endl;
        float erro = (float)error;
        float erro_c = (float)error_c;
        if (erro > e_membership_values[0] && erro < e_membership_values[6])
            //���ƫ����������֮��
        {
            for (int i = 0; i < num_area - 2; i++)
            {
                if (erro >= e_membership_values[i] && erro <= e_membership_values[i + 1])
                {
                    e_gradmembership[0] = -(erro - e_membership_values[i + 1]) / (e_membership_values[i + 1] - e_membership_values[i]);
                    e_gradmembership[1] = 1 + (erro - e_membership_values[i + 1]) / (e_membership_values[i + 1] - e_membership_values[i]);
                    e_grad_index[0] = i;
                    e_grad_index[1] = i + 1;
                    break;
                }
            }
        }
        else
        {//������ֵ�Ļ���ֱ�����ó������ޣ���ʹ��e_grad_index[1] = -1����ע
            if (erro <= e_membership_values[0])
            {
                e_gradmembership[0] = 1;
                e_gradmembership[1] = 0;
                e_grad_index[0] = 0;
                e_grad_index[1] = -1;
            }
            else if (erro >= e_membership_values[6])
            {
                e_gradmembership[0] = 1;
                e_gradmembership[1] = 0;
                e_grad_index[0] = 6;
                e_grad_index[1] = -1;
            }
        }

        if (erro_c > ec_membership_values[0] && erro_c < ec_membership_values[6])
        {
            for (int i = 0; i < num_area - 2; i++)
            {
                if (erro_c >= ec_membership_values[i] && erro_c <= ec_membership_values[i + 1])
                {
                    ec_gradmembership[0] = -(erro_c - ec_membership_values[i + 1]) / (ec_membership_values[i + 1] - ec_membership_values[i]);
                    ec_gradmembership[1] = 1 + (erro_c - ec_membership_values[i + 1]) / (ec_membership_values[i + 1] - ec_membership_values[i]);
                    ec_grad_index[0] = i;
                    ec_grad_index[1] = i + 1;
                    break;
                }
            }
        }
        else
        {
            if (erro_c <= ec_membership_values[0])
            {
                ec_gradmembership[0] = 1;
                ec_gradmembership[1] = 0;
                ec_grad_index[0] = 0;
                ec_grad_index[1] = -1;
            }
            else if (erro_c >= ec_membership_values[6])
            {
                ec_gradmembership[0] = 1;
                ec_gradmembership[1] = 0;
                ec_grad_index[0] = 6;
                ec_grad_index[1] = -1;
            }
        }

    }
    void GetSumGrad()
    {
        for (int i = 0; i <= num_area - 1; i++)
        {
            KpgradSums[i] = 0;
            KdgradSums[i] = 0;

        }
        for (int i = 0; i < 2; i++)
        {
            if (e_grad_index[i] == -1)
            {
                continue;
            }
            for (int j = 0; j < 2; j++)
            {
                if (ec_grad_index[j] != -1)
                {//��δ���ʹ��֮ǰ���������������һ�����ڼ�������p��d�����飬������������ֵ��Ӿ������յ�p��d
                    //���Ǻ���ֵ�һ�������᲻�ϸ��ǵ�����������ĸ��ھ�����ֵͬ�Ľڵ�
                    //Ҳ����˵���ĸ��ڵ��нڵ���ֵ��ͬ���ᰴ�մ��ϵ��£������ҵ�˳���������һ��
                    //ʮ�������ŷֵ����
                    int indexKp = Kp_rule_list[e_grad_index[i]][ec_grad_index[j]] / 8 + 3;
                    int indexKd = Kd_rule_list[e_grad_index[i]][ec_grad_index[j]] / 8 + 3;
                    //std::cout<<indexKp<<"\t"<<indexKd<<"\t";
                   //gradSums[index] = gradSums[index] + (e_gradmembership[i] * ec_gradmembership[j])* Kp_rule_list[e_grad_index[i]][ec_grad_index[j]];
                    KpgradSums[indexKp] = (e_gradmembership[i] * ec_gradmembership[j]);
                    KdgradSums[indexKd] = (e_gradmembership[i] * ec_gradmembership[j]);
                }
                else
                {
                    continue;
                }

            }
        }

    }
    //�����������kp,kd,ki��Ӧ����ֵ
    void GetOUT()
    {
        for (int i = 0; i < num_area - 1; i++)
        {
            qdetail_kp += kp_menbership_values[i] * KpgradSums[i];
            qdetail_kd += kd_menbership_values[i] * KdgradSums[i];
        }
    }
    float Quantization(int maximum, int minimum, int x)
    {
        float qvalues = 48.0 * (x - minimum) / (maximum - minimum) - 24;
        //float qvalues=6.0*()
        return qvalues;

        //qvalues[1] = 3.0 * ecerro / (maximum - minimum);
    }
    int FuzzyPIDcontroller(int error, int erro_c, int erro_pre, int errp_ppre) {
        qerro = Quantization(e_membership_values[6], e_membership_values[0], error);
        qerro_c = Quantization(ec_membership_values[6], ec_membership_values[0], erro_c);
        Get_grad_membership(error, erro_c);
        //std::cout<<error<<"\t"<<qerro<<"\t"<<e_gradmembership[0]<<"\t"<<e_gradmembership[1]<<"\t";
        // std::cout<<e_grad_index[0]<<"\t"<<e_grad_index[1]<<"\t";
        // std::cout<<ec_grad_index[0]<<"\t"<<ec_grad_index[1]<<"\t";
        // std::cout<<ec_gradmembership[0]<<"\t"<<ec_gradmembership[1]<<"\t"<<std::endl;
        GetSumGrad();
        GetOUT();
        // detail_kp = Inverse_quantization(kp_menbership_values[6], kp_menbership_values[0], qdetail_kp);
        // detail_kd = Inverse_quantization(kd_menbership_values[6], kd_menbership_values[0], qdetail_kd);
        //std::cout<<detail_kp<<"\t"<<std::endl;
        kp = kp + qdetail_kp;
        kd = kd + qdetail_kd;
        qdetail_kp = 0;
        qdetail_kd = 0;
        if (kp < 1)
            kp = 1;
        if (kp > 2)
            kp = 2;
        if (kd < 1)
            kd = 1;
        if (kd > 10)
            kd = 10;
        std::cout << kp << "\t" << kd << "\t" << std::endl;
        float output = error * kp + erro_c * kd;
        return (int)output;
    }

};