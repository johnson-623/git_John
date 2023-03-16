#include <string.h>
#include <iostream> 

#include <time.h> 
#include <math.h>
#include <fstream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>


#include "STLreader.h"
#include "zxhImageGipl.h" 
#include "zxhImageModelingLinear.h"

using namespace std;


int main(int argc, char* argv[])
{
	string mainfold, Datafold, TargetImageName, TargetLabelName, MeshWallName, mesh_name, LABloodPoolGaussianBlurName, \
		GoldLALabelName, MeshWallLabelName;

	//mainfold = "c:\\leili\\2020_miccai\\la2020\\train_data";
	//string  casename = "patient_1";

	mainfold = argv[1];
	string  casename = argv[2];

	string PathName = mainfold + "\\" + casename;
	TargetImageName = PathName + "\\enhanced.nii.gz";
	TargetLabelName = PathName + "\\en_seg_msp_M.nii.gz";
	mesh_name = PathName + "\\LA_Mesh_M.stl";
	MeshWallName = PathName + "\\LA_MeshWall_M.nii.gz";
	LABloodPoolGaussianBlurName = PathName + "\\LA_label_GauiisanBlur_M.nii.gz"; // never use LA_MeshWall_GauiisanBlur
	GoldLALabelName = PathName + "\\scarSegImgM.nii.gz";

	MeshWallLabelName = PathName + "\\LA_MeshWallLabel_M.nii.gz";



	zxhImageData SourceImage, SourceLabel, MeshWallImage, LABloodPoolGaussianBlur, GoldLALabel, MeshWallLabelImage;
	zxh::OpenImageSafe(&SourceImage, TargetImageName);
	zxh::OpenImageSafe(&SourceLabel, TargetLabelName);
	zxh::OpenImageSafe(&MeshWallImage, MeshWallName);
	zxh::OpenImageSafe(&LABloodPoolGaussianBlur, LABloodPoolGaussianBlurName);
	zxh::OpenImageSafe(&GoldLALabel, GoldLALabelName);



	MeshWallLabelImage.CloneFrom(&MeshWallImage);//copy from wall
	const int * Size = SourceImage.GetImageSize();

	STLreader *stlreader = new STLreader(mesh_name);
	vtkSmartPointer<vtkPolyData> LAMesh = stlreader->decimated;	//load in the mesh
	const int iNumOfMeshPoints = LAMesh->GetNumberOfPoints(); //the number of mesh point

	for (int ptId = 0; ptId < iNumOfMeshPoints; ptId++)
	{
		float MeshNode_P2I_Coor[] = { LAMesh->GetPoint(ptId)[0], LAMesh->GetPoint(ptId)[1], LAMesh->GetPoint(ptId)[2], 0 };
		SourceImage.GetImageInfo()->PhysicalToImage(MeshNode_P2I_Coor);//��������ת��ͼ������
		int scx = zxh::round(MeshNode_P2I_Coor[0]);
		int scy = zxh::round(MeshNode_P2I_Coor[1]);
		int scz = zxh::round(MeshNode_P2I_Coor[2]);
		//int scz = zxh::round(Size[2] - MeshNode_P2I_Coor[2]);
		
		bool bIsInsideImage = SourceImage.InsideImage(scx, scy, scz, 0);
		if (!bIsInsideImage)
		{
			std::cout << "error: node " << ptId << " not inside image\n";
			continue;
		}


		ZXHPixelTypeDefault MeshWallSurfaceValue = MeshWallImage.GetPixelGreyscaleClosest(scx, scy, scz, 0);
		if (MeshWallSurfaceValue == 0)//PV
			continue;

		//------------------------------Get New input world coordinate-------------------------------- 
		float InputWorldCoord[] = { scx, scy, scz, 0 };
		SourceImage.ImageToWorld(InputWorldCoord);

		zxhImageModelingLinear GradientMod;
		GradientMod.SetImage(&LABloodPoolGaussianBlur); ///////-- ���󣬲���ʹ��wall��smooth��ͼ��������---------------
		float  pwGrad[4] = { 0 };
		GradientMod.GetPixelGradientByWorld(pwGrad, InputWorldCoord[0], InputWorldCoord[1], InputWorldCoord[2], 0);
		float ix = pwGrad[0];
		float iy = pwGrad[1];
		float iz = pwGrad[2];
		float mag = sqrt(ix*ix + iy*iy + iz*iz);
		if (mag<ZXH_FloatInfinitesimal)
		{
			std::cout << "error: magnitude " << mag << " too small for node " << ptId << "\n";
			return -1;
		}
		float Ia = 0.5*ix / mag; // ÿ��ֻ��0.5mm��������ֹ���scar���С��1mm�ĵ�
		float Ib = 0.5*iy / mag;
		float Ic = 0.5*iz / mag;

		int ScarIndex = 0;
		int PVIndex = 0;
		for (int step = 0; step <= 20; step++) // 0.5mm for each step
		{
			// forward step
			float  NewInputWorldCoord[4] = { InputWorldCoord[0] + step*Ia, InputWorldCoord[1] + step*Ib, InputWorldCoord[2] + step*Ic, 0 };
			float NewInputImageCoord[] = { NewInputWorldCoord[0], NewInputWorldCoord[1], NewInputWorldCoord[2], 0 };
			SourceImage.WorldToImage(NewInputImageCoord);
			int cx = zxh::round(NewInputImageCoord[0]);
			int cy = zxh::round(NewInputImageCoord[1]);
			int cz = zxh::round(NewInputImageCoord[2]);
			bool bIsInsideImage = SourceImage.InsideImage(cx, cy, cz, 0);
			if (!bIsInsideImage)
				continue;//���������ĵ�



			float GoldLALabelValue = GoldLALabel.GetPixelGreyscaleClosest(cx, cy, cz, 0);
			if (GoldLALabelValue > 0)
			{
				ScarIndex++;
				break;
			}
			//remove PV and Mitral valve
			float LALabelValue = SourceLabel.GetPixelGreyscaleClosest(cx, cy, cz, 0);
			if (LALabelValue == 500)
			{
				PVIndex++;
				break;
			}
			// backward step
			float  BackInputWorldCoord[4] = { InputWorldCoord[0] - step*Ia, InputWorldCoord[1] - step*Ib, InputWorldCoord[2] - step*Ic, 0 };
			float BackInputImageCoord[] = { BackInputWorldCoord[0], BackInputWorldCoord[1], BackInputWorldCoord[2], 0 };
			SourceImage.WorldToImage(BackInputImageCoord);
			cx = zxh::round(BackInputImageCoord[0]);
			cy = zxh::round(BackInputImageCoord[1]);
			cz = zxh::round(BackInputImageCoord[2]);
			bIsInsideImage = SourceImage.InsideImage(cx, cy, cz, 0);
			if (!bIsInsideImage)
				continue;

			GoldLALabelValue = GoldLALabel.GetPixelGreyscaleClosest(cx, cy, cz, 0);
			if (GoldLALabelValue > 0)
			{
				ScarIndex++;
				break;
			}
			
			LALabelValue = SourceLabel.GetPixelGreyscaleClosest(cx, cy, cz, 0);
			/*if ((LALabelValue == 1) || (LALabelValue == 500))*/ //remove PV and Mitral valve
			if (LALabelValue == 500) //remove Mitral valve
			{
				PVIndex++;
				break;
			}

		}

		/* for (int kk = -10; kk < 10; kk++)
		{
		float  NewInputWorldCoord[4] = { InputWorldCoord[0] + kk*Ia, InputWorldCoord[1] + kk*Ib, InputWorldCoord[2] + kk*Ic, 0 };
		float NewInputImageCoord[] = { NewInputWorldCoord[0], NewInputWorldCoord[1], NewInputWorldCoord[2], 0 };
		SourceImage.WorldToImage(NewInputImageCoord);
		int cx = zxh::round(NewInputImageCoord[0]);
		int cy = zxh::round(NewInputImageCoord[1]);
		int cz = zxh::round(NewInputImageCoord[2]);

		float GoldLALabelValue = GoldLALabel.GetPixelGreyscaleClosest(cx, cy, cz, 0);
		if (GoldLALabelValue > 0)
		{
		ScarIndex++;
		}
		//remove PV and Mitral valve
		float LALabelValue = LALabel.GetPixelGreyscaleClosest(cx, cy, cz, 0);
		if ((LALabelValue == 1) || (LALabelValue == 500))
		{
		PVIndex++;
		}

		} */

		if (PVIndex > 0)//ֻҪ����PV,�͸�Ϊ0
		{
			MeshWallLabelImage.SetPixelByGreyscale(scx, scy, scz, 0, 0);
		}

		else if ((ScarIndex > 0) && (PVIndex == 0))//ֻҪ����scar���Ҳ���PV�����͸�Ϊ422
		{
			MeshWallLabelImage.SetPixelByGreyscale(scx, scy, scz, 0, 422);
		}


		else if ((ScarIndex == 0) && (PVIndex == 0))
		{
			MeshWallLabelImage.SetPixelByGreyscale(scx, scy, scz, 0, 421);
		}
	}

	const char *SegResultName = MeshWallLabelName.data();
	zxh::SaveImage(&MeshWallLabelImage, SegResultName);

	return 0;
}

