#include "STLreader.h"

STLreader::STLreader(std::string meshfilepath)//���캯��
{
	//---------------�������ɵ�mesh-----------------------------------------//
	vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();
	char * meshFileName = (char *)meshfilepath.c_str();  //��stringת��char��
	reader->SetFileName(meshFileName);
	reader->Update();

	//---------------��mesh���г�ȡ������������-----------------------------------------//
	vtkSmartPointer<vtkDecimatePro> decimation =
		vtkSmartPointer<vtkDecimatePro>::New();
	decimation->SetInputData(reader->GetOutput());
	decimation->SetTargetReduction(0.000000001);//��0.000000001��������Ƭ���Ƴ�
	decimation->Update();

	decimated = decimation->GetOutput();

	/*decimated = reader->GetOutput();*/

	/*
	//---------------��ʾ��ȡ���mesh-----------------------------------------//
	//���һЩֵ��������mesh
	vtkIdType ptId = 100;
	vtkIdList* clist = vtkIdList::New();

	std::cout << "ptId���꣺" << endl;
	std::cout << decimated->GetPoint(ptId)[0] << "\t";//�����±꣬�õ���������Χ�������ֵ
	std::cout << decimated->GetPoint(ptId)[1] << "\t";
	std::cout << decimated->GetPoint(ptId)[2] << endl;
	std::cout << endl;

	decimated->GetPointCells(ptId, clist);//�����±꣬�õ���Χ�������ֵ
	std::cout << "ptId��Χ������꣺" << endl;
	for (size_t j = 0; j < clist->GetNumberOfIds(); j++)
	{
		std::cout << decimated->GetPoint(clist->GetId(j))[0] << "\t";
		std::cout << decimated->GetPoint(clist->GetId(j))[1] << "\t";
		std::cout << decimated->GetPoint(clist->GetId(j))[2] << endl;
		std::cout << endl;
	}


	vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
	mapper->SetInputConnection(decimation->GetOutputPort());
	vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
	actor->SetMapper(mapper);

	vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
	vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
	renderWindow->AddRenderer(renderer);
	vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
	renderWindowInteractor->SetRenderWindow(renderWindow);

	renderer->AddActor(actor);
	renderer->SetBackground(.3, .6, .3); // Background color green 
	renderWindow->Render();
	renderWindowInteractor->Start();
	*/
}


STLreader::~STLreader()
{
}
