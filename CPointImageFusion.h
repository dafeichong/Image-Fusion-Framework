/**************************************************************************

Copyright:CXX

Author: cxx

Date:2020-06-08

Description: Concrete class for image fusion inherited by CAbstractImageFusion 
			 override FindClearArea method using point grad

**************************************************************************/

#pragma once
#include "CAbstractImageFusion.h"

class CPointImageFusion : public CAbstractImageFusion
{
public:
	CPointImageFusion();
	virtual ~CPointImageFusion();
//--------------------------------------------------------------------------------------------------------------------
	/*
	PointGrad: �ҵ��ཹ������ͼ��ÿһ���������
	@parameter src: ԭʼ����ͼ��
	@parameter grad: �õ����ݶ�ͼ��
	*/
	void PointGrad(const cv::Mat& src, cv::Mat& grad);
//--------------------------------------------------------------------------------------------------------------------
	/*
	TenenGrad: �ҵ��ཹ������ͼ��ÿһ���������
	@parameter src: ԭʼ����ͼ��
	@parameter grad: �õ����ݶ�ͼ��
	*/
	void TenenGrad(const cv::Mat& src, cv::Mat& grad, int win_size = 3);
//--------------------------------------------------------------------------------------------------------------------
	/*
	SV: �����ݶȷ����ҵ��ཹ������ͼ��ÿһ���������
	@parameter src: ԭʼ����ͼ��
	@parameter grad: �õ����ݶ�ͼ��
	*/
	void SV(const cv::Mat& src, cv::Mat& grad, int win_size = 3);
//--------------------------------------------------------------------------------------------------------------------
	/*
	FFT: ���ø���ҶƵ�����ֵ�ҵ��ཹ������ͼ��ÿһ���������
	@parameter src: ԭʼ����ͼ��
	@parameter grad: �õ����ݶ�ͼ��
	*/
	void FFT(const cv::Mat& src, cv::Mat& grad, int win_size = 3);
//--------------------------------------------------------------------------------------------------------------------
	/*
	Variance: ���÷����ҵ��ཹ������ͼ��ÿһ���������
	@parameter src: ԭʼ����ͼ��
	@parameter grad: �õ����ݶ�ͼ��
	*/
	void Variance(const cv::Mat& src, cv::Mat& grad, int win_size = 3);
//--------------------------------------------------------------------------------------------------------------------
	/*
	GetVariance: ����ͼ��ķ���
	@parameter src: ԭʼ����ͼ��
	@return : ����õ��ķ���ֵ
	*/
	float GetVariance(const cv::Mat& src);
//--------------------------------------------------------------------------------------------------------------------
	/*
	GetFourier: ����ͼ���Ƶ�����ֵ
	@parameter src: ԭʼ����ͼ��
	@return : ����õ���Ƶ�����ֵ
	*/
	float GetFourier(const cv::Mat& src);
//--------------------------------------------------------------------------------------------------------------------
	/*
	GetSV: ����ͼ����ݶȷ���ֵ
	@parameter src: ԭʼ����ͼ��
	@return : ����õ����ݶȷ���ֵ
	*/
	float GetSV(const cv::Mat& src);
//--------------------------------------------------------------------------------------------------------------------





	//��¼���������ֵ 
	float max_val;
protected:
	//override functions
//--------------------------------------------------------------------------------------------------------------------
	/*
	FindClearArea: �ҵ��ཹ������ͼ��ÿһ���������
	@parameter images: �ཹ��ͼ������
	@parameter clear_images: �õ���������ͼ������
	*/
	virtual void FindClearArea(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& clear_images);
//--------------------------------------------------------------------------------------------------------------------
	/*
	IndexSmooth: ���ø�˹����������ƽ��������ƽ����������
	@parameter index_mat: ��������
	@parameter smooth_index_mat: ƽ�������������
	*/
	virtual void IndexSmooth(const cv::Mat& index_mat, cv::Mat& smooth_index_mat);
//--------------------------------------------------------------------------------------------------------------------
};

