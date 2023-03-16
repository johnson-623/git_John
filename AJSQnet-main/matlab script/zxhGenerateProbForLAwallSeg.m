function [imgLab,Sub3D]=zxhGenerateProbForLAwallSeg(strWhsLabelFilename,savecaseid, savefold)
% author: Xiahai Zhuang
% date of creation:   2017-02
% current version:  
%	succ=zxhGenerateProbForLAwallSeg(strWhsLabelFilename, strPreSave, fstdForGenerateProb, iMaskDilateMM) 
% Save Files: strPreSave + "Label.nii.gz"; strPreSave +"Sub3D.nii.gz"(mask) strPreSave + "Prob_Label0/1/2/3.nii.gz"; all in 1x1x1 mm
%             Label with new label values for 0 (bk), 420 (LA blood), 429 (LA wall), 199 (other myo wall), 550 (other blood), 500 (LV)
%		* erode+dilate 1.5 mm for LA=420, aorta=820-825, RA=550, PA=850-855; all wall thickness init = 3 mm
%                        average wall thickness: 2.2\pm0.9mm, for MRI init 3 mm could be good
%		* dilated label - ori label = wall, set LA wall, and the other walls and LVmyo (200-250) to 200 
%		* dilate iMaskDilateMM (recommend: 4-6) on label 2 to generate a mask ---> LAwall+20mm for generating ROI of Sub3D
%		* zxhvolumelabelop -genprobf f imgref, f=fstdForGenerateProb (recommend: 2)
%    
	verbose=' -v 0 ' ;
	imgLab = [savefold savecaseid '_Label.nii.gz'];
% 	Sub3D = [savefold savecaseid '_Sub3D.nii.gz']; 
    Sub3D = [savefold 'LAwall_gd_new.nii.gz']; 

	img1 = [savefold savecaseid '__tmp1.nii.gz']; 
	img2 = [savefold savecaseid '__tmp2.nii.gz']; 
	img3 = [savefold savecaseid '__tmp3.nii.gz']; 
	img4 = [savefold savecaseid '__tmp4.nii.gz']; 
    
    %�ı�spacing��������nearest��ֵ��resize,��������dilation����erosion�ĳߴ��ȡֵ
	%command=['zxhtransform ' strWhsLabelFilename ' -o ' imgLab ' -resave -spacing 1 1 1 -nearest ' verbose]; system(command);
    copyfile( strWhsLabelFilename, imgLab, 'f' ) ;%-----add by lei 2020.06.08
    
    %��RA(550),RV(600),PA(850),AO(820)��vauel�ĳ�550
    %other blood ->550
	command=['zxhimageop -int ' imgLab ' -vr 550 1000 -vs 550 ' verbose]; system(command);
    
    %��maskimage���Ƹ�img1,img4
	copyfile( imgLab, img1, 'f' ) ;  
	copyfile( imgLab, img4, 'f' ) ;  
	
    
	% set all expect LA+LV to 0, erode 1.5mm from LA-bk, and set to LA wall
    
    %�ѳ��ˡ�LA(420),LV(500)��������Ϊ0
	command=['zxhimageop -int ' img1 ' -vr 0 400 -vr 550 1000 -vs 0 ' verbose]; system(command);
    
    %��0Ϊ������LA(420)Ϊǰ����erosion�õ�img2
    %command=['zxhimageop -int ' img1 ' -o ' img2 ' -DIs 0.5 0 420 ' verbose]; system(command); %-----add by lei 2020.03.05
	command=['zxhimageop -int ' img1 ' -o ' img3 ' -ERs 2 0 420 ' verbose]; system(command);%-----modify by lei 2020.06.08
    
    %img3= img1-img2,�õ��ľ��Ǹո�LA  erosion������la wall
	command=['zxhimageop -int ' img1 ' -int ' img3 ' -o ' img3 ' -sub ' verbose]; system(command);
    
    %�ѵõ���la wall ��Ϊmask ����˰�la wall ��ӽ�ȥ�������ⲿ�ֵ� value ���ó�429
	command=['zxhimageop -int ' imgLab ' -o ' imgLab ' -vmaskimage ' img3 ' 1 1000  -vs 429 ' verbose]; system(command);
	
	% set LA to 0, erode 1.5mm from otherblood-bk, and set to other wall; set LVmyo to other wall
    
    %��LAȥ��
	command=['zxhimageop -int ' img4 ' -vr 420 420 -vs 0 ' verbose]; system(command); 
    
    %�������桰��RA(550),RV(600),PA(850),AO(820)��vauel�ĳ�550��������֪������һ������LA����Ĳ���erosion
	command=['zxhimageop -int ' img4 ' -o ' img2 ' -ERs 1.5 0 550 ' verbose]; system(command); 
    
    %ͬ�ϣ�����õ��ľ���������blood��erosion������wall
	command=['zxhimageop -int ' img4 ' -int ' img2 ' -o ' img3 ' -sub ' verbose]; system(command);
    
    %���ⲿ�ֵ�wall��Ϊmask ����˰�la wall ��ӽ�ȥ�������ⲿ�ֵ� value ���ó�199
	command=['zxhimageop -int ' imgLab ' -o ' imgLab ' -vmaskimage ' img3 ' 1 1000  -vs 199 ' verbose]; system(command);
	 

	 % set LAwall dilate resp 1.5 mm 
	command=['zxhimageop -int ' imgLab ' -DIs 1.5 0 429 ' verbose]; system(command);
	 % set other wall dilate 1.5 mm;  
	command=['zxhimageop -int ' imgLab ' -DIs 1.5 0 199 ' verbose]; system(command);
    
    %��myocardial�� valueҲ��Ϊ199
	command=['zxhimageop -int ' imgLab ' -o ' imgLab ' -vr 199 250  -vs 199 ' verbose]; system(command);


    
    %��la wall��value��ֵ ���� Ϊ 100�� ����һ��la wall
	command=['zxhimageop -int ' imgLab ' -o ' Sub3D ' -vr 429 429 -VS 100 ' verbose]; system(command);
    %command=['zxhimageop -int ' imgLab ' -o ' Sub3D ' -imageinfosrc2 ' strWhsLabelFilename]; system(command);

	
	delete(img1,img2,img3,img4, imgLab); 
	
end

% zxhvolumelabelop MeanLabel_new.nii.gz  test1.nii.gz -gvoutrange 
% zxhvolumelabelop test1.nii.gz  test2.nii.gz -rmnonwh 

