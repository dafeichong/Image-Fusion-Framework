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
	PointGrad: 找到多焦面序列图像每一层的清晰点
	@parameter src: 原始输入图像
	@parameter grad: 得到的梯度图像
	*/
	void PointGrad(const cv::Mat& src, cv::Mat& grad);
//--------------------------------------------------------------------------------------------------------------------
	/*
	TenenGrad: 找到多焦面序列图像每一层的清晰点
	@parameter src: 原始输入图像
	@parameter grad: 得到的梯度图像
	*/
	void TenenGrad(const cv::Mat& src, cv::Mat& grad, int win_size = 3);
//--------------------------------------------------------------------------------------------------------------------
	/*
	SV: 利用梯度方差找到多焦面序列图像每一层的清晰点
	@parameter src: 原始输入图像
	@parameter grad: 得到的梯度图像
	*/
	void SV(const cv::Mat& src, cv::Mat& grad, int win_size = 3);
//--------------------------------------------------------------------------------------------------------------------
	/*
	FFT: 利用傅里叶频率域幅值找到多焦面序列图像每一层的清晰点
	@parameter src: 原始输入图像
	@parameter grad: 得到的梯度图像
	*/
	void FFT(const cv::Mat& src, cv::Mat& grad, int win_size = 3);
//--------------------------------------------------------------------------------------------------------------------
	/*
	Variance: 利用方差找到多焦面序列图像每一层的清晰点
	@parameter src: 原始输入图像
	@parameter grad: 得到的梯度图像
	*/
	void Variance(const cv::Mat& src, cv::Mat& grad, int win_size = 3);
//--------------------------------------------------------------------------------------------------------------------
	/*
	GetVariance: 计算图像的方差
	@parameter src: 原始输入图像
	@return : 计算得到的方差值
	*/
	float GetVariance(const cv::Mat& src);
//--------------------------------------------------------------------------------------------------------------------
	/*
	GetFourier: 计算图像的频率域幅值
	@parameter src: 原始输入图像
	@return : 计算得到的频率域幅值
	*/
	float GetFourier(const cv::Mat& src);
//--------------------------------------------------------------------------------------------------------------------
	/*
	GetSV: 计算图像的梯度方差值
	@parameter src: 原始输入图像
	@return : 计算得到的梯度方差值
	*/
	float GetSV(const cv::Mat& src);
//--------------------------------------------------------------------------------------------------------------------





	//记录最大清晰度值 
	float max_val;
protected:
	//override functions
//--------------------------------------------------------------------------------------------------------------------
	/*
	FindClearArea: 找到多焦面序列图像每一层的清晰点
	@parameter images: 多焦面图像序列
	@parameter clear_images: 得到的清晰度图像序列
	*/
	virtual void FindClearArea(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& clear_images);
//--------------------------------------------------------------------------------------------------------------------
	/*
	IndexSmooth: 利用高斯函数或其他平滑函数，平滑索引矩阵
	@parameter index_mat: 索引矩阵
	@parameter smooth_index_mat: 平滑后的索引矩阵
	*/
	virtual void IndexSmooth(const cv::Mat& index_mat, cv::Mat& smooth_index_mat);
//--------------------------------------------------------------------------------------------------------------------
};

