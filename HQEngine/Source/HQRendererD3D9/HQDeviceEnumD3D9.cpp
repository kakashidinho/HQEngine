/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceD3D9PCH.h"
#include "HQDeviceEnumD3D9.h"
#include "../HQHashTable.h"
#include <stdio.h>

const int numBackFmt=9;//số lượng backbuffer format hỗ trợ
const D3DFORMAT BackFormat[]={
	D3DFMT_R8G8B8,   D3DFMT_A8R8G8B8, 
	D3DFMT_X8R8G8B8, D3DFMT_R5G6B5, 
	D3DFMT_A1R5G5B5, D3DFMT_X1R5G5B5,
	D3DFMT_X4R4G4B4, D3DFMT_A4R4G4B4,
	D3DFMT_A2B10G10R10 
};

const int numMulSample=4;//số lượng kiểu siêu lấy mẫu hỗ trợ
const D3DMULTISAMPLE_TYPE MulSample[]=
{
	D3DMULTISAMPLE_NONE,
	D3DMULTISAMPLE_2_SAMPLES,
	D3DMULTISAMPLE_4_SAMPLES,
	D3DMULTISAMPLE_8_SAMPLES
};


const int numDevType=2;//số lượng dạng device hỗ trợ
const D3DDEVTYPE DevType[]={
	D3DDEVTYPE_HAL,
	D3DDEVTYPE_REF
};

//********************************************
//constructor
//********************************************
HQDeviceEnumD3D9::HQDeviceEnumD3D9(){
	vsync = 0;//default vsync is off
	selectedVertexProcBehavior=D3DCREATE_SOFTWARE_VERTEXPROCESSING;
	selectedDepthStencilFmt=D3DFMT_D16;
	selectedMulSampleType=D3DMULTISAMPLE_NONE;
}
//********************************************
//Truy vấn khả năng của card đồ họa
//********************************************
//helper function
bool HQDeviceEnumD3D9::CheckDepthStencilFmt(IDirect3D9 * d3d,ComboInfo & combo){
	HRESULT hr=d3d->CheckDeviceFormat(combo.adapter,combo.devType,combo.displayFmt,
		D3DUSAGE_DEPTHSTENCIL,D3DRTYPE_SURFACE,combo.depthStencilFmt);
	if(FAILED(hr))
		return false;
	hr=d3d->CheckDepthStencilMatch(combo.adapter,combo.devType,combo.displayFmt,
		combo.backFmt,combo.depthStencilFmt);
	if(FAILED(hr))
		return false;
	return true;
}

//main function
void HQDeviceEnumD3D9::Enum(IDirect3D9 *d3d)
{
	if(!d3d)
		return;
	hq_uint32 numAdapter=d3d->GetAdapterCount();//truy vấn số lượng adapter
	
	HRESULT hr;

	for (hq_uint32 i=0;i<numAdapter;++i)
	{
		AdapterInfo adapterInfo;
		adapterInfo.adapter=i;

		d3d->GetAdapterIdentifier(i,0,&adapterInfo.adapterID);

		hq_uint32 numDisplay=0;
		D3DDISPLAYMODE displayMode;
		D3DFORMAT currentDisplayFormat;
		if(FAILED(d3d->GetAdapterDisplayMode(i , &displayMode)))
			continue;
		
		currentDisplayFormat = displayMode.Format;

		numDisplay=d3d->GetAdapterModeCount(i, currentDisplayFormat);//truy vấn số lượng kiểu hiển thị hỗ trợ với display format này
		for(hq_uint32 m=0;m<numDisplay;++m)
		{
			//truy vấn tất cả các kiểu hiển thị
			d3d->EnumAdapterModes(i, currentDisplayFormat ,m,&displayMode);
			adapterInfo.dModelist.PushBack(displayMode);
		}

		for(int j=0;j<numDevType;++j){//kiểm tra các dạng device hal , ref , sw
			DeviceInfo devInfo;
			devInfo.adapter=i;
			hr=d3d->GetDeviceCaps(i,DevType[j],&devInfo.dCaps);
			if(FAILED(hr))
				continue;
			
			//check if dxt compressed texture supported
			if (SUCCEEDED(d3d->CheckDeviceFormat( i,
												  DevType[j],
												  currentDisplayFormat,
												  0,
												  D3DRTYPE_TEXTURE,
												  D3DFMT_DXT1)))
				devInfo.dCaps.s3tc_dxtFlags |= D3DFMT_DXT1_SUPPORT;

			if (SUCCEEDED(d3d->CheckDeviceFormat( i,
												  DevType[j],
												  currentDisplayFormat,
												  0,
												  D3DRTYPE_TEXTURE,
												  D3DFMT_DXT3)))
				devInfo.dCaps.s3tc_dxtFlags |= D3DFMT_DXT3_SUPPORT;

			if (SUCCEEDED(d3d->CheckDeviceFormat( i,
												  DevType[j],
												  currentDisplayFormat,
												  0,
												  D3DRTYPE_TEXTURE,
												  D3DFMT_DXT5)))
				devInfo.dCaps.s3tc_dxtFlags |= D3DFMT_DXT5_SUPPORT;

			if (SUCCEEDED(d3d->CheckDeviceFormat( i,
												  DevType[j],
												  currentDisplayFormat,
												  0,
												  D3DRTYPE_CUBETEXTURE,
												  D3DFMT_DXT1)))
				devInfo.dCaps.s3tc_dxtFlags |= D3DFMT_DXT1_CUBE_SUPPORT;

			if (SUCCEEDED(d3d->CheckDeviceFormat( i,
												  DevType[j],
												  currentDisplayFormat,
												  0,
												  D3DRTYPE_CUBETEXTURE,
												  D3DFMT_DXT3)))
				devInfo.dCaps.s3tc_dxtFlags |= D3DFMT_DXT3_CUBE_SUPPORT;

			if (SUCCEEDED(d3d->CheckDeviceFormat( i,
												  DevType[j],
												  currentDisplayFormat,
												  0,
												  D3DRTYPE_CUBETEXTURE,
												  D3DFMT_DXT5)))
				devInfo.dCaps.s3tc_dxtFlags |= D3DFMT_DXT5_CUBE_SUPPORT;


			/*----------------*/
			devInfo.numVertexShaderSamplers = 0;
			devInfo.numPixelShaderSamplers = 0;
			if (devInfo.dCaps.VertexShaderVersion >= D3DVS_VERSION(3,0))
				devInfo.numVertexShaderSamplers = 4;
			if(devInfo.dCaps.PixelShaderVersion >= D3DVS_VERSION(2,0))
				devInfo.numPixelShaderSamplers = 16;
			else if(devInfo.dCaps.PixelShaderVersion >= D3DVS_VERSION(1,1))
				devInfo.numPixelShaderSamplers = 4;

			if (devInfo.dCaps.MaxSimultaneousTextures > 8)
				devInfo.dCaps.MaxSimultaneousTextures = 8;

			devInfo.devType=DevType[j];
			//kiểm tra từng dạng combo
			for(int l=0;l<numBackFmt;++l)//for all supported backbuffer format
			{
				bool windowed=false;
				//kiểm tra lần lượt 2 chế độ windowed và fullscreen
				for(int m=0;m<2;++m){
					windowed=!windowed;
					hr=d3d->CheckDeviceType(i,DevType[j],currentDisplayFormat,
						BackFormat[l],windowed);
					if(FAILED(hr))
						continue;

					ComboInfo combo;
					combo.adapter=i;
					combo.devType=DevType[j];
					combo.windowed=windowed;
					combo.displayFmt=currentDisplayFormat;
					combo.backFmt=BackFormat[l];

					//chọn depthStencil format tốt nhất
					bool success;
					combo.depthStencilFmt=D3DFMT_D24S8;
					success=CheckDepthStencilFmt(d3d,combo);
					if(!success){
						combo.depthStencilFmt=D3DFMT_D24X4S4;
						success=CheckDepthStencilFmt(d3d,combo);
					}
					if(!success){
						combo.depthStencilFmt=D3DFMT_D15S1;
						success=CheckDepthStencilFmt(d3d,combo);
					}
					combo.depthStencilFmt=D3DFMT_D32;
					success=CheckDepthStencilFmt(d3d,combo);
					if(!success){
						combo.depthStencilFmt=D3DFMT_D24X8;
						success=CheckDepthStencilFmt(d3d,combo);
					}
					if(!success){
						combo.depthStencilFmt=D3DFMT_D16;
						success=CheckDepthStencilFmt(d3d,combo);
					}
					if(!success)
						continue;

					//chọn kiểu xử lý đỉnh tốt nhất (hardware/software..v.v..v.v)
					if(devInfo.dCaps.DevCaps&D3DDEVCAPS_HWTRANSFORMANDLIGHT)//device có hỗ trợ hardware T&L
					{
						if(devInfo.dCaps.DevCaps&D3DDEVCAPS_PUREDEVICE)//support pure device ,fastest mode
							combo.vertProcBehavior=D3DCREATE_HARDWARE_VERTEXPROCESSING|D3DCREATE_PUREDEVICE;
						else //hardware mode
							combo.vertProcBehavior=D3DCREATE_HARDWARE_VERTEXPROCESSING;
					}
					else{
						//software mode
						combo.vertProcBehavior=D3DCREATE_SOFTWARE_VERTEXPROCESSING;
					}
					//chọn kiểu siêu lấy mẫu tốt nhất
					for(int n=numMulSample-1;n>=0;--n){//for all supported multisample type
						combo.maxMulSampType=MulSample[n];
						hr=d3d->CheckDeviceMultiSampleType(i,devInfo.devType,
							combo.backFmt,combo.windowed,combo.maxMulSampType,NULL);//check multisample type on back buffer
						if(SUCCEEDED(hr))
						{	
							hr=d3d->CheckDeviceMultiSampleType(i,devInfo.devType,
								combo.depthStencilFmt,combo.windowed,combo.maxMulSampType,NULL);//check multisample type on depth stencil buffer

							if(SUCCEEDED(hr))
								break;
						}
					}//for n
					devInfo.comboList.PushBack(combo);
				}//for (m)
			}//for (l)
			if(devInfo.comboList.GetSize())
				adapterInfo.dev.PushBack(devInfo);
		}//for (j)

		this->adapterlist.PushBack(adapterInfo);
	}

	this->selectedAdapter=&adapterlist.GetFront();

	this->selectedMode=&selectedAdapter->dModelist.GetFront();

	this->selectedDevice=&selectedAdapter->dev.GetFront();

	this->selectedCombo=&selectedDevice->comboList.GetFront();

	selectedVertexProcBehavior=selectedCombo->vertProcBehavior;
	selectedDepthStencilFmt=selectedCombo->depthStencilFmt;


}
//****************************************
//parse setting file
//****************************************
void HQDeviceEnumD3D9::ParseSettingFile(const char *settingFile,IDirect3D9* d3d){
	FILE * save=0;
	if(!settingFile)
		return;
	save=fopen(settingFile,"r");
	if(!save)
		return;
	int d1;
	UINT sWidth,sHeight,refreshRate;
	int windowed;

	fscanf(save,"Basic Settings\n");
	fscanf(save,"Width=%u\n",&sWidth);
	fscanf(save,"Height=%u\n",&sHeight);
	fscanf(save,"Windowed=%d\n",&windowed);
	fscanf(save,"RefreshRate=%u\n",&refreshRate);
	fscanf(save,"VSync=%d\n\n",&this->vsync);
	
	fscanf(save,"Advanced Settings\n");
	fscanf(save,"Adapter=%d\n",&d1);

	HQLinkedListNode<AdapterInfo>* adapterNode = adapterlist.GetRoot();
	HQLinkedListNode<DeviceInfo>* devNode ;
	HQLinkedListNode<ComboInfo>* comboNode;
	HQLinkedListNode<D3DDISPLAYMODE>* modeNode;


	//find adapter
	for(hq_uint32 i = 0; i < adapterlist.GetSize(); ++i , adapterNode = adapterNode->m_pNext){
		if((int)adapterNode->m_element.adapter==d1)
		{
			this->selectedAdapter=&adapterNode->m_element;
			break;
		}
	}

	fscanf(save,"Device Type=%d\n",&d1);
	//find device
	devNode = selectedAdapter->dev.GetRoot();
	for(hq_uint32 i = 0; i < selectedAdapter->dev.GetSize(); ++i, devNode = devNode->m_pNext)
	{
		if((int)devNode->m_element.devType==d1)
		{
			this->selectedDevice=&devNode->m_element;
			break;
		}
	}
	
	bool found = false;
	fscanf(save,"BackBuffer Format=%d\n",&d1);
	
	//find closest match display mode
	UINT minDiff = 9999;
	UINT diff;
	modeNode = selectedAdapter->dModelist.GetRoot();
	for(hq_uint32 i = 0; i < selectedAdapter->dModelist.GetSize(); ++i, modeNode = modeNode->m_pNext)
	{
		if(modeNode->m_element.Width==sWidth&&
			modeNode->m_element.Height==sHeight)
		{
			diff = abs((int)modeNode->m_element.RefreshRate - (int)refreshRate);
			if(diff < minDiff)
			{
				minDiff = diff;
				this->selectedMode=&modeNode->m_element;
				found = true;
			}
		}
	}
	if(!found)//not found, this is custom resolution, it only is allowed in windowed mode
	{
		this->customWindowedMode.Format = selectedAdapter->dModelist.GetFront().Format;
		this->customWindowedMode.RefreshRate = refreshRate;
		this->customWindowedMode.Width = sWidth;
		this->customWindowedMode.Height = sHeight;
		this->selectedMode = &customWindowedMode;

		windowed = 1;
	}
	//find combo
	comboNode = selectedDevice->comboList.GetRoot();
	for(hq_uint32 i = 0; i < selectedDevice->comboList.GetSize(); ++i, comboNode = comboNode->m_pNext)
	{
		if(comboNode->m_element.displayFmt==selectedMode->Format&&
			comboNode->m_element.backFmt==d1&&
			comboNode->m_element.windowed==(windowed!=0))
		{
			this->selectedCombo=&comboNode->m_element;
			break;
		}
	}
	
	//check depthstencil
	ComboInfo combo;
	memcpy(&combo,this->selectedCombo,sizeof(ComboInfo));
	//scan value from file
	fscanf(save,"DepthStencil Format=%d\n",(int*)&selectedDepthStencilFmt);
	if (selectedDepthStencilFmt == 81)//no depth stencil buffer
		selectedDepthStencilFmt = D3DFMT_FORCE_DWORD;
	else 
	{
		if (selectedDepthStencilFmt == 82)//only 8 bit stencil
			combo.depthStencilFmt=D3DFMT_D24S8;
		
		combo.depthStencilFmt=selectedDepthStencilFmt;
		//check if valid format
		if(!CheckDepthStencilFmt(d3d,combo))//không hỗ trợ
			selectedDepthStencilFmt=this->selectedCombo->depthStencilFmt;//fallback to default
	}

	fscanf(save,"Vertex Processing=%d\n",(int*)&this->selectedVertexProcBehavior);
	if(selectedVertexProcBehavior > selectedCombo->vertProcBehavior)
		selectedVertexProcBehavior = selectedCombo->vertProcBehavior;//fallback

	fscanf(save,"Multisample Type=%d\n",(int*)&this->selectedMulSampleType);
	if(selectedMulSampleType > selectedCombo->maxMulSampType)//lớn hơn giá trị lớn nhất device hỗ trợ
		selectedMulSampleType = selectedCombo->maxMulSampType;//fallback
	fclose(save);
}
//***********************
//save setting
//***********************
void HQDeviceEnumD3D9::SaveSettingFile(const char *settingFile){
	if(!settingFile)
		return;
	FILE *save=0;
	save=fopen(settingFile,"w");
	if(!save)
		return;
	fprintf(save,"Basic Settings\n");
	fprintf(save,"Width=%u\n",selectedMode->Width);
	fprintf(save,"Height=%u\n",selectedMode->Height);
	fprintf(save,"Windowed=%d\n",(int)selectedCombo->windowed);
	fprintf(save,"RefreshRate=%u\n",selectedMode->RefreshRate);
	fprintf(save,"VSync=%d\n\n",this->vsync);
	
	fprintf(save,"Advanced Settings\n");
	fprintf(save,"Adapter=%d\n",selectedAdapter->adapter);
	fprintf(save,"Device Type=%d\n",(int)selectedDevice->devType);
	fprintf(save,"BackBuffer Format=%d\n",(int)selectedCombo->backFmt);
	if (selectedDepthStencilFmt != D3DFMT_FORCE_DWORD)
		fprintf(save,"DepthStencil Format=%d\n",(int)selectedDepthStencilFmt);
	else
		fprintf(save,"DepthStencil Format=81\n");
	fprintf(save,"Vertex Processing=%d\n",(int)this->selectedVertexProcBehavior);
	fprintf(save,"Multisample Type=%d\n",(int)this->selectedMulSampleType);

	fclose(save);
}


//************************************************
//get all available fullscreen display resolutions
//************************************************

struct ResolutionHashFunc
{
	hq_uint32 operator()(const HQResolution &val) const
	{
		return (val.width << 16) | val.height; 
	}
};

class ResolutonEqual
{
public:
	bool operator() (const HQResolution & lhs , const HQResolution & rhs) const
	{
		if (lhs.width == rhs.width && lhs.height == rhs.height)
			return true;
		return false;
	}

};

void HQDeviceEnumD3D9::GetAllDisplayResolution(HQResolution *resolutionList , hq_uint32& numResolutions)
{

	HQResolution res;
	
	HQHashTable<HQResolution , bool  ,ResolutionHashFunc , ResolutonEqual> 
		existTable(selectedAdapter->dModelist.GetSize() * 2 + 1);
	
	HQLinkedListNode<D3DDISPLAYMODE> *pRNode = selectedAdapter->dModelist.GetRoot();
	if (resolutionList != NULL)//save resolutions array
	{
		hquint32 j = 0;
		for(hq_uint32 i = 0 ; i < selectedAdapter->dModelist.GetSize() && j < numResolutions ; ++i)
		{
			res.width = pRNode->m_element.Width;
			res.height = pRNode->m_element.Height;

			if (existTable.Add(res , true))//độ phân giải này chưa cho vào list
			{
				resolutionList[j++] = res;
			}

			pRNode = pRNode->m_pNext;
		}
	}
	else // count max number of resolutions
	{
		numResolutions = 0;
		for(hq_uint32 i = 0 ; i < selectedAdapter->dModelist.GetSize() ; ++i)
		{
			res.width = pRNode->m_element.Width;
			res.height = pRNode->m_element.Height;

			if (existTable.Add(res , true))//độ phân giải này chưa cho vào list
			{
				numResolutions ++;
			}

			pRNode = pRNode->m_pNext;
		}
	}
}
//****************************************
//change display mode
//****************************************
bool HQDeviceEnumD3D9::ChangeSelectedDisplay(hq_uint32 width,hq_uint32 height,bool windowed)
{
	D3DDISPLAYMODE *dmode = NULL;
	ComboInfo *combo = NULL;
	bool found = false;
	//tìm trong danh sách độ phân giải chuẩn
	//find closest match 
	UINT minDiff = 9999;
	UINT diff;
	HQLinkedListNode<D3DDISPLAYMODE> *pRNode = selectedAdapter->dModelist.GetRoot();
	for(hq_uint32 i = 0 ; i < selectedAdapter->dModelist.GetSize() ; ++i , pRNode = pRNode->m_pNext)
	{
		if(pRNode->m_element.Width == width && pRNode->m_element.Height == height && pRNode->m_element.Format == this->selectedMode->Format)
		{
			diff = abs((int)pRNode->m_element.RefreshRate - (int)this->selectedMode->RefreshRate);
			if(diff < minDiff)
			{
				dmode = &pRNode->m_element;
				minDiff = diff;
			}
		}
	}
	if(!dmode && windowed)// không tìm thấy trong danh sách độ phân giải chuẩn, nếu đang chọn chế độ windowed, cho phép tùy chọn độ phân giải bất kỳ
	{
		this->customWindowedMode.Format = this->selectedMode->Format;
		this->customWindowedMode.RefreshRate = this->selectedMode->RefreshRate;
		this->customWindowedMode.Width = width;
		this->customWindowedMode.Height = height;
		dmode = &customWindowedMode;
	}

	found = false;
	HQLinkedListNode<ComboInfo> *comboNode = selectedDevice->comboList.GetRoot();
	for(hq_uint32 i = 0; i < selectedDevice->comboList.GetSize(); ++i, comboNode = comboNode->m_pNext)
	{
		if(comboNode->m_element.backFmt == selectedCombo->backFmt && 
			comboNode->m_element.depthStencilFmt == selectedCombo->depthStencilFmt &&
			comboNode->m_element.displayFmt == selectedCombo->displayFmt && 
			comboNode->m_element.vertProcBehavior >= this->selectedVertexProcBehavior &&
			comboNode->m_element.windowed == windowed)
		{
			combo = &comboNode->m_element;
			found = true;
		}
	}
	
	if(combo != NULL && dmode != NULL)
	{
		this->selectedMode = dmode;
		this->selectedCombo = combo;
		return true;
	}
	return false;
}
