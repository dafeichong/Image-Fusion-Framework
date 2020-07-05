#include "CPointImageFusion.h"

CPointImageFusion::CPointImageFusion()
{

}

CPointImageFusion::~CPointImageFusion()
{

}
/*
double CSystemControl::SV(cv::Mat mat)
{
	double sv = 0.0;
	cv::Mat sobel_mat;
	cv::Sobel(mat, sobel_mat, CV_16S, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT);
	cv::Scalar mean = cv::mean(cv::abs(sobel_mat));
	double average = mean.val[0];  //x方向差分图的均值
	for (int i = 0; i < sobel_mat.rows; i++)
	{
		for (int j = 0; j < sobel_mat.cols; j++)
		{
			short* ptr = sobel_mat.ptr<short>(i, j);
			short value = *ptr;
			sv += (value - average) * (value - average); //平方
		}
	}
	std::cout << "the SV is " << sv << std::endl;
	return sv;
}

*/
void CPointImageFusion::PointGrad(const cv::Mat& src, cv::Mat& grad)
{
	int rows = src.rows;
	int cols = src.cols;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1)
			{
				grad.at<uchar>(i, j) = 0;
			}
			else
			{
				uchar up = src.at<uchar>(i - 1, j);
				uchar down = src.at<uchar>(i + 1, j);
				uchar right = src.at<uchar>(i, j + 1);
				uchar left = src.at<uchar>(i, j - 1);
				uchar val = abs(up - down) + abs(right - left);
				//越界判断
				if (val >= 255)
					val = 255;
				if (val <= 0)
					val = 0;
				grad.at<uchar>(i, j) = val;
			}
		}
	}
}

void CPointImageFusion::TenenGrad(const cv::Mat & src, cv::Mat & grad, int win_size)
{
	using namespace std;
	using namespace cv;

	float sv = 0.0;
	cv::Mat sobel_x, sobel_y;
	cv::Mat abs_grad_x, abs_grad_y;
	cv::Mat sobel;
	cv::Sobel(src, sobel_x, CV_16S, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(sobel_x, abs_grad_x);
	cv::Sobel(src, sobel_y, CV_16S, 0, 1, 3, 1, 1, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(sobel_y, abs_grad_y);
	cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel);

	int rows = src.rows;
	int cols = src.cols;
	int offset = win_size / 2;
	max_val = 0.0;
	vector<vector<float>> temp = vector<vector<float>>(rows, vector<float>(cols, 0.0));
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)

		{
			if (i >= rows - offset || i < offset || j >= cols - offset || j < offset)
			{
				continue;
			}
			else
			{
				float var = GetVariance(sobel(Rect(j - offset, i - offset, win_size, win_size)));
				max_val = max(max_val, var);
				temp[i][j] = var;

			}
		}
	}
	//将temp中的数据归一化为0-255
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (i >= rows - offset || i < offset || j >= cols - offset || j < offset)
			{
				continue;
			}
			else
			{
				uchar val = (uchar)(temp[i][j] / max_val * 255.0);
				//cout << (int)val << endl;
				grad.at<uchar>(i, j) = val;
			}
		}
	}
}

void CPointImageFusion::SV(const cv::Mat & src, cv::Mat & grad, int win_size)
{
	using namespace std;
	using namespace cv;

	float sv = 0.0;
	cv::Mat sobel_x, sobel_y;
	cv::Mat abs_grad_x, abs_grad_y;
	cv::Mat sobel;
	cv::Sobel(src, sobel_x, CV_16S, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(sobel_x, abs_grad_x);
	cv::Sobel(src, sobel_y, CV_16S, 0, 1, 3, 1, 1, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(sobel_y, abs_grad_y);
	cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel);

	int rows = src.rows;
	int cols = src.cols;
	int offset = win_size / 2;
	max_val = 0.0;
	vector<vector<float>> temp = vector<vector<float>>(rows, vector<float>(cols, 0.0));
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)

		{
			if (i >= rows - offset || i < offset || j >= cols - offset || j < offset)
			{
				continue;
			}
			else
			{
				float var = GetVariance(sobel(Rect(j - offset, i - offset, win_size, win_size)));
				max_val = max(max_val, var);
				temp[i][j] = var;

			}
		}
	}
	//将temp中的数据归一化为0-255
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (i >= rows - offset || i < offset || j >= cols - offset || j < offset)
			{
				continue;
			}
			else
			{
				uchar val = (uchar)(temp[i][j] / max_val * 255.0);
				//cout << (int)val << endl;
				grad.at<uchar>(i, j) = val;
			}
		}
	}
}

void CPointImageFusion::FFT(const cv::Mat & src, cv::Mat & grad, int win_size)
{
	using namespace std;
	using namespace cv;
	int rows = src.rows;
	int cols = src.cols;
	int offset = win_size / 2;
	max_val = 0.0;
	vector<vector<float>> temp = vector<vector<float>>(rows, vector<float>(cols, 0.0));
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (i >= rows - offset || i < offset || j >= cols - offset || j < offset)
			{
				temp[i][j] = 0.0;
			}
			else
			{
				float var = GetFourier(src(Rect(j - offset, i - offset, win_size, win_size)));
				max_val = max(max_val, var);
				temp[i][j] = var;
				
			}
		}
	}
	//将temp中的数据归一化为0-255
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (i >= rows - offset || i < offset || j >= cols - offset || j < offset)
			{
				continue;
			}
			else
			{
				uchar val = (uchar)(temp[i][j] / max_val * 255.0);
				//cout << (int)val << endl;
				grad.at<uchar>(i, j) = val;
			}
		}
	}
}

void CPointImageFusion::Variance(const cv::Mat & src, cv::Mat & grad, int win_size)
{
	using namespace std;
	using namespace cv;
	int rows = src.rows;
	int cols = src.cols;
	int offset = win_size / 2;
	max_val = 0.0;
	vector<vector<float>> temp = vector<vector<float>>(rows, vector<float>(cols, 0.0));
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)

		{
			if (i >= rows - offset || i < offset || j >= cols - offset || j < offset)
			{
				continue;
			}
			else
			{
				float var = GetVariance(src(Rect(j - offset, i - offset, win_size, win_size)));
				max_val = max(max_val, var);
				temp[i][j] = var;
				
			}
		}
	}
	//将temp中的数据归一化为0-255
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (i >= rows - offset || i < offset || j >= cols - offset || j < offset)
			{
				continue;
			}
			else
			{
				uchar val = (uchar)(temp[i][j] / max_val * 255.0);
				//cout << (int)val << endl;
				grad.at<uchar>(i, j) = val;
			}
		}
	}
}

float CPointImageFusion::GetVariance(const cv::Mat & src)
{
	using namespace std;
	double variance = 0.0;
	cv::Scalar mean = cv::mean(src);
	double average = mean.val[0];  //平均值
	int size = src.rows * src.cols;
	if (src.channels() != 1)
	{
		cout << "mat通道数不为1" << endl;
		return 0.0;
	}
	else
	{
		for (int i = 0; i < src.rows; i++)
		{
			for (int j = 0; j < src.cols; j++)
			{
				variance += pow(((&src.data[i * src.step])[j] - average), 2); //平方
			}
		}
		//std::cout << " the variance is " << variance << std::endl;
	}
	return variance;
}

float CPointImageFusion::GetFourier(const cv::Mat& src)
{
	float fourier = 0.0;
	int rows = cv::getOptimalDFTSize(src.rows);
	int cols = cv::getOptimalDFTSize(src.cols);
	cv::Mat padded;
	cv::copyMakeBorder(src, padded, 0, rows - src.rows, 0, cols - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
	cv::Mat planes[] = { cv::Mat_<float>(padded),cv::Mat::zeros(padded.size(),CV_32F) };
	cv::Mat complexI;
	cv::merge(planes, 2, complexI);
	cv::dft(complexI, complexI);
	cv::split(complexI, planes);
	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]);
	cv::Mat magnitudeImage = planes[0];
	fourier = cv::sum(magnitudeImage)[0];
	return fourier;
}

float CPointImageFusion::GetSV(const cv::Mat& src)
{
	float sv = 0.0;
	cv::Mat sobel_x, sobel_y;
	cv::Mat abs_grad_x, abs_grad_y;
	cv::Mat sobel;
	cv::Sobel(src, sobel_x, CV_16S, 1, 0, 3, 1, 1, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(sobel_x, abs_grad_x);
	cv::Sobel(src, sobel_y, CV_16S, 0, 1, 3, 1, 1, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(sobel_y, abs_grad_y);
	cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobel);
	cv::Scalar mean = cv::mean(cv::abs(sobel));
	double average = mean.val[0];  //x方向差分图的均值
	for (int i = 0; i < sobel.rows; i++)
	{
		for (int j = 0; j < sobel.cols; j++)
		{
			short* ptr = sobel_x.ptr<short>(i, j);
			short value = *ptr;
			sv += pow((value - average), 2); //平方
		}
	}
	//std::cout << "the SV is " << sv << std::endl;
	return sv;
}

void CPointImageFusion::FindClearArea(const std::vector<cv::Mat>& images, std::vector<cv::Mat>& clear_images)
{
	//clear_images已经申请好内存，并且每个点都为0
	int nums = images.size();
	for (int k = 0; k < nums; k++)
	{
		//PointGrad(images[k], clear_images[k]);
		//Variance(images[k], clear_images[k], 5);
		SV(images[k], clear_images[k], 5);
	}
}

void CPointImageFusion::IndexSmooth(const cv::Mat & index_mat, cv::Mat & smooth_index_mat)
{
	cv::GaussianBlur(index_mat, smooth_index_mat, cv::Size(47, 47), 0);
}

