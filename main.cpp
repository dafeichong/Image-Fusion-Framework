#include <QtCore/QCoreApplication>
#include "CAbstractImageFusion.h"
#include "CPointImageFusion.h"

void read_images_from_file(const char* input_filename, std::vector<cv::Mat>& images, int num_of_images)
{
	//read images into vector<Mat>
	for (int a = 0; a < num_of_images; a++)
	{
		std::string name = input_filename + cv::format("%d.jpg", a);
		cv::Mat img = cv::imread(name, CV_LOAD_IMAGE_ANYDEPTH);
		if (img.empty())
		{
			std::cerr << "whaa " << name << " can't be loaded!" << std::endl;
			continue;
		}
		images.push_back(img);
	}
}


int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
	CAbstractImageFusion* pimage_fusion = new CPointImageFusion();
	std::vector<cv::Mat> images;
	//from disk read multi-focus images
	read_images_from_file("../40_3/", images, 120);
	cv::Mat fusion_image;
	pimage_fusion->Fuse(images, fusion_image, false);
	//pimage_fusion->Fuse(images, fusion_image, index_mat, false);
	cv::imshow("1", fusion_image);
	cv::imwrite("点融合.jpg", fusion_image);
	cv::Mat binary_image;
	cv::threshold(fusion_image, binary_image, 128, 255, CV_THRESH_OTSU);
	cv::imwrite("二值图_高斯.jpg", binary_image);
	delete pimage_fusion;
    return a.exec();
}
