/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "HQDeviceD3D11PCH.h"
#include "HQDeviceEnumD3D11.h"

#if defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM

#include "../HQEngine/winstore/HQWinStoreFileSystem.h"
#include "../HQEngine/winstore/HQWinStoreUtil.h"

#include <ppltasks.h>

using namespace Windows::Storage;

#endif

#include <stdio.h>
#include <map>
#include <math.h>

enum FORMAT
{
    SFMT_R8G8B8               = 20,
    SFMT_A8R8G8B8             = 21,
    SFMT_X8R8G8B8             = 22,
    SFMT_R5G6B5               = 23,
    SFMT_X1R5G5B5             = 24,
    SFMT_A1R5G5B5             = 25,
    SFMT_A4R4G4B4             = 26,
    SFMT_R3G3B2               = 27,
    SFMT_A8                   = 28,
    SFMT_A8R3G3B2             = 29,
    SFMT_X4R4G4B4             = 30,
    SFMT_A2B10G10R10          = 31,
    SFMT_A8B8G8R8             = 32,
    SFMT_X8B8G8R8             = 33,
    SFMT_D32                  = 71,
    SFMT_D15S1                = 73,
    SFMT_D24S8                = 75,
    SFMT_D24X8                = 77,
    SFMT_D24X4S4              = 79,
    SFMT_D16                  = 80,
	SFMT_S8					  = 82,
	SFMT_NODEPTHSTENCIL		  = 81,
	SFMT_UNKNOWN			  = 0xffffffff,
};

const int numDisFmt=4;//số lượng display format hỗ trợ
const FORMAT DisFormat[]={
	SFMT_A8R8G8B8, SFMT_R5G6B5, 
	SFMT_A1R5G5B5, SFMT_A2B10G10R10
};


const int numMulSample=4;//số lượng kiểu siêu lấy mẫu hỗ trợ
const UINT MulSample[]=
{
	1,
	2,
	4,
	8
};


//helper functions
namespace helper{
	FORMAT GetFormat(DXGI_FORMAT dxFmt)
	{
		switch(dxFmt)
		{
		case DXGI_FORMAT_R8G8B8A8_UNORM : return SFMT_A8R8G8B8;
		case DXGI_FORMAT_R10G10B10A2_UNORM: return SFMT_A2B10G10R10;
		case DXGI_FORMAT_D32_FLOAT: return  SFMT_D32;
		case DXGI_FORMAT_D24_UNORM_S8_UINT: return SFMT_D24S8;
		case DXGI_FORMAT_D16_UNORM:  return SFMT_D16;
		case DXGI_FORMAT_FORCE_UINT: return SFMT_NODEPTHSTENCIL;
		default: return SFMT_UNKNOWN;
		}
	}
	DXGI_FORMAT GetDXGI_Format(FORMAT format)
	{
		switch(format)
		{
		case SFMT_R8G8B8: return DXGI_FORMAT_R8G8B8A8_UNORM;
		case SFMT_A8R8G8B8: return DXGI_FORMAT_R8G8B8A8_UNORM;
		case SFMT_X8R8G8B8: return DXGI_FORMAT_R8G8B8A8_UNORM;
		case SFMT_R5G6B5: 
#if defined HQ_WIN_STORE_PLATFORM || defined HQ_WIN_PHONE_PLATFORM
			return DXGI_FORMAT_B5G6R5_UNORM;
#else
			return DXGI_FORMAT_R8G8B8A8_UNORM;//backward compatible with direct3d9
#endif
		case SFMT_A1R5G5B5: 
#if defined HQ_WIN_STORE_PLATFORM || defined HQ_WIN_PHONE_PLATFORM
			return DXGI_FORMAT_B5G5R5A1_UNORM;
#else
			return DXGI_FORMAT_R8G8B8A8_UNORM;//backward compatible with direct3d9
#endif
		case SFMT_X1R5G5B5: return DXGI_FORMAT_R8G8B8A8_UNORM;//backward compatible with direct3d9
		case SFMT_X4R4G4B4: return DXGI_FORMAT_R8G8B8A8_UNORM;//backward compatible with direct3d9
		case SFMT_A4R4G4B4: return DXGI_FORMAT_R8G8B8A8_UNORM;//backward compatible with direct3d9
		case SFMT_A2B10G10R10: return DXGI_FORMAT_R10G10B10A2_UNORM;
		case SFMT_D32: return DXGI_FORMAT_D32_FLOAT;
		case SFMT_D24S8: case SFMT_S8 : return DXGI_FORMAT_D24_UNORM_S8_UINT;
		case SFMT_D24X8: return DXGI_FORMAT_D24_UNORM_S8_UINT;
		case SFMT_D16:  return DXGI_FORMAT_D16_UNORM;
		case SFMT_NODEPTHSTENCIL: return DXGI_FORMAT_FORCE_UINT;
		default: return DXGI_FORMAT_UNKNOWN ;
		}
	}
	
};
//********************************************
//constructor
//********************************************
HQDeviceEnumD3D11::HQDeviceEnumD3D11(IDXGIFactory * pFactory){
	this->pFactory = pFactory;

	selectedDepthStencilFmt=DXGI_FORMAT_D24_UNORM_S8_UINT;
	selectedDriverType = D3D_DRIVER_TYPE_HARDWARE;
	selectedMulSamplesCount=1;
	windowed = true;
#if defined HQ_WIN_PHONE_PLATFORM
	this->vsync = 1;//vsync always on
#else
	vsync = 0;//default vsync is off
#endif

	customWindowedMode.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
	customWindowedMode.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_PROGRESSIVE;
}
//********************************************
//Truy vấn khả năng của card đồ họa
//********************************************
//main function
void HQDeviceEnumD3D11::Enum()
{
	if(!pFactory)
		return;
	hq_uint32 i = 0;
	IDXGIAdapter * pAdapter; 

	while(pFactory->EnumAdapters(i,&pAdapter) != DXGI_ERROR_NOT_FOUND)
	{
		AdapterInfo adapterInfo(i);
		adapterInfo.adapter= pAdapter;
		pAdapter->AddRef();
		pAdapter->GetDesc(&adapterInfo.adapterDesc);
		
		IDXGIOutput* pOutput = NULL;
		DXGI_FORMAT dxFormat;
		DXGI_MODE_DESC * modeDescs= NULL;
		UINT numDisplay = 0;
		HRESULT hr = pAdapter->EnumOutputs(0,&pOutput);
		if (!FAILED(hr) && pOutput != NULL)
		{
			for(int j=0;j<numDisFmt;++j){//for all supported display format
				dxFormat = helper::GetDXGI_Format(DisFormat[j]);
				pOutput->GetDisplayModeList(dxFormat , 0 ,&numDisplay , NULL);//truy vấn số lượng kiểu hiển thị hỗ trợ với display format này
				
				modeDescs = (DXGI_MODE_DESC*)realloc(modeDescs , numDisplay * sizeof(DXGI_MODE_DESC));
				pOutput->GetDisplayModeList(dxFormat , 0 ,&numDisplay , modeDescs);

				for(hq_uint32 m=0;m<numDisplay;++m)
				{
					adapterInfo.dModelist.push_back(modeDescs[m]);
				}
			}
			
			if (adapterInfo.dModelist.size() == 0)//find at least a display mode
			{
				DXGI_MODE_DESC desc;
				desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
				desc.Width = 0;
				desc.Height = 0;
				desc.RefreshRate.Denominator = desc.RefreshRate.Numerator = 0;
				desc.Scaling = DXGI_MODE_SCALING_UNSPECIFIED;
				desc.ScanlineOrdering = DXGI_MODE_SCANLINE_ORDER_UNSPECIFIED;

				DXGI_MODE_DESC descOut;
				hr = pOutput->FindClosestMatchingMode(&desc, &descOut, NULL);
				if (SUCCEEDED( hr ))
				{
					pOutput->GetDisplayModeList(descOut.Format , 0 ,&numDisplay , NULL);//truy vấn số lượng kiểu hiển thị hỗ trợ với display format này
				
					modeDescs = (DXGI_MODE_DESC*)realloc(modeDescs , numDisplay * sizeof(DXGI_MODE_DESC));
					pOutput->GetDisplayModeList(dxFormat , 0 ,&numDisplay , modeDescs);

					for(hq_uint32 m=0;m<numDisplay;++m)
					{
						adapterInfo.dModelist.push_back(modeDescs[m]);
					}
				}
				else
				{
					//hack
					desc.Width = desc.Height = 1;
					desc.RefreshRate.Numerator = 60; 
					desc.RefreshRate.Denominator = 1;
					adapterInfo.dModelist.push_back(desc);
				}
			}//if (adapterInfo.dModelist.size() == 0)

			pOutput->Release();
			free(modeDescs);

		}

		this->adapterlist.push_back(adapterInfo);

		i++;//next adapter
	}
	std::list<AdapterInfo>::iterator adapterI;
	std::list<DXGI_MODE_DESC>::iterator modeI;

	adapterI=adapterlist.begin();
	this->selectedAdapter=&(*(adapterI));

	modeI=selectedAdapter->dModelist.begin();
	this->selectedMode=&(*modeI);

}
//****************************************
//parse setting file
//****************************************
void HQDeviceEnumD3D11::ParseSettingFile(const char *settingFile){

#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	//windows store and windows phone style
	if(!settingFile)
		return;

	auto File =  HQWinStoreFileSystem::OpenFileForRead(settingFile);
	if (File == nullptr)
		return;
	int d1;
	int lineNumber = 0;
	UINT sWidth,sHeight,refreshRate;
	DXGI_FORMAT dxFmt;

	//read each line
	char *line = HQ_NEW char[256];

	if (File->GetLine(line, 256)) 
		sscanf(line, "Basic Settings");
	if (File->GetLine(line, 256)) 
		sscanf(line, "Width=%u",&sWidth);
	if (File->GetLine(line, 256)) 
		sscanf(line, "Height=%u",&sHeight);
	if (File->GetLine(line, 256)) 
		sscanf(line, "Windowed=%d",&d1);
	if (File->GetLine(line, 256)) 
		sscanf(line, "RefreshRate=%u",&refreshRate);
	if (File->GetLine(line, 256)) 
		sscanf(line, "VSync=%d",&this->vsync);
	
	this->windowed= true;//this value is not relevant
#if defined HQ_WIN_PHONE_PLATFORM
	this->vsync = 1;//vsync always on
#endif

	if (File->GetLine(line, 256)) //empty line
		sscanf(line, "");
	if (File->GetLine(line, 256)) 
		sscanf(line, "Advanced Settings");
	if (File->GetLine(line, 256)) 
		sscanf(line, "Adapter=%d",&d1);

	d1 = 0;//windows phone always use default adapter for now. TO DO: future

	std::list<AdapterInfo>::iterator adapterI;
	std::list<DXGI_MODE_DESC>::iterator modeI;
	//find adapter
	for(adapterI=adapterlist.begin();adapterI!=adapterlist.end();adapterI++){
		if((int)(*adapterI).adapterIndex==d1)
		{
			this->selectedAdapter=&(*adapterI);
			break;
		}
	}

	if (File->GetLine(line, 256)) 
		sscanf(line, "Device Type=%d",&d1);
	if(d1 == 1)
		this->selectedDriverType = D3D_DRIVER_TYPE_HARDWARE;
	else if (d1 == 2)
		this->selectedDriverType = D3D_DRIVER_TYPE_WARP;//D3D_DRIVER_TYPE_REFERENCE;
	
	bool found = false;
	if (File->GetLine(line, 256)) 
		sscanf(line, "BackBuffer Format=%d",&d1);
	
	dxFmt = helper::GetDXGI_Format((FORMAT)d1);
	
	//find closest match display mode
	double minDiff = 99999;
	double diff;
	double thisModeRefresh;
	for(modeI=selectedAdapter->dModelist.begin();modeI!=selectedAdapter->dModelist.end();modeI++)
	{
		if((*modeI).Width==sWidth&&
			(*modeI).Height==sHeight&&
			(*modeI).Format==dxFmt)
		{
			thisModeRefresh = (double)(*modeI).RefreshRate.Numerator / (*modeI).RefreshRate.Denominator;
			diff = fabs(thisModeRefresh - refreshRate);
			if(diff < minDiff)
			{
				minDiff = diff;
				this->selectedMode=&(*modeI);
				found = true;
			}
		}
	}
	if(!found)//not found, this is custom resolution, it only is allowed in windowed mode
	{
		this->customWindowedMode.Format = dxFmt;
		this->customWindowedMode.RefreshRate.Numerator = refreshRate;
		this->customWindowedMode.RefreshRate.Denominator = 1;
		this->customWindowedMode.Width = sWidth;
		this->customWindowedMode.Height = sHeight;
		this->selectedMode = &customWindowedMode;

		windowed = 1;
	}
	
	//depth stencil format
	if (File->GetLine(line, 256)) 
		sscanf(line, "DepthStencil Format=%d",&d1);
	this->selectedDepthStencilFmt = helper::GetDXGI_Format((FORMAT)d1);

	if (File->GetLine(line, 256)) 
		sscanf(line, "Vertex Processing=%d",&this->unUseValue[0]);//not use , this value is for direct3d9 device

	if (File->GetLine(line, 256)) 
		sscanf(line, "Multisample Type=%u",&this->selectedMulSamplesCount);

#if defined HQ_WIN_STORE_PLATFORM//TO DO: windows store multisample is disable for now. If it is enable, seperate render target must be used
	this->selectedMulSamplesCount = 0;
#endif

	if(this->selectedMulSamplesCount == 0)
		this->selectedMulSamplesCount = 1;

	delete[] line;
	fclose(File);
#else
	//standard C file reading
	FILE * save=0;
	if(!settingFile)
		return;
	save=fopen(settingFile,"r");
	if(!save)
		return;
	int d1;
	UINT sWidth,sHeight,refreshRate;
	DXGI_FORMAT dxFmt;

	fscanf(save,"Basic Settings\n");
	fscanf(save,"Width=%u\n",&sWidth);
	fscanf(save,"Height=%u\n",&sHeight);
	fscanf(save,"Windowed=%d\n",&d1);
	fscanf(save,"RefreshRate=%u\n",&refreshRate);
	fscanf(save,"VSync=%d\n\n",&this->vsync);
	
	this->windowed= (d1 !=0);

	fscanf(save,"Advanced Settings\n");
	fscanf(save,"Adapter=%d\n",&d1);

	std::list<AdapterInfo>::iterator adapterI;
	std::list<DXGI_MODE_DESC>::iterator modeI;
	//find adapter
	for(adapterI=adapterlist.begin();adapterI!=adapterlist.end();adapterI++){
		if((int)(*adapterI).adapterIndex==d1)
		{
			this->selectedAdapter=&(*adapterI);
			break;
		}
	}

	fscanf(save,"Device Type=%d\n",&d1);
	if(d1 == 1)
		this->selectedDriverType = D3D_DRIVER_TYPE_HARDWARE;
	else if (d1 == 2)
		this->selectedDriverType = D3D_DRIVER_TYPE_WARP;//D3D_DRIVER_TYPE_REFERENCE;
	
	bool found = false;
	fscanf(save,"BackBuffer Format=%d\n",&d1);
	
	dxFmt = helper::GetDXGI_Format((FORMAT)d1);
	
	//find closest match display mode
	double minDiff = 99999;
	double diff;
	double thisModeRefresh;
	for(modeI=selectedAdapter->dModelist.begin();modeI!=selectedAdapter->dModelist.end();modeI++)
	{
		if((*modeI).Width==sWidth&&
			(*modeI).Height==sHeight&&
			(*modeI).Format==dxFmt)
		{
			thisModeRefresh = (double)(*modeI).RefreshRate.Numerator / (*modeI).RefreshRate.Denominator;
			diff = fabs(thisModeRefresh - refreshRate);
			if(diff < minDiff)
			{
				minDiff = diff;
				this->selectedMode=&(*modeI);
				found = true;
			}
		}
	}
	if(!found)//not found, this is custom resolution, it only is allowed in windowed mode
	{
		this->customWindowedMode.Format = dxFmt;
		this->customWindowedMode.RefreshRate.Numerator = refreshRate;
		this->customWindowedMode.RefreshRate.Denominator = 1;
		this->customWindowedMode.Width = sWidth;
		this->customWindowedMode.Height = sHeight;
		this->selectedMode = &customWindowedMode;

		windowed = 1;
	}
	
	//depth stencil format
	fscanf(save,"DepthStencil Format=%d\n",&d1);
	this->selectedDepthStencilFmt = helper::GetDXGI_Format((FORMAT)d1);

	fscanf(save,"Vertex Processing=%d\n",&this->unUseValue[0]);//not use , this value is for direct3d9 device

	fscanf(save,"Multisample Type=%u\n",&this->selectedMulSamplesCount);

	if(this->selectedMulSamplesCount == 0)
		this->selectedMulSamplesCount = 1;

	fclose(save);
#endif//#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
}
//***********************
//save setting
//***********************
void HQDeviceEnumD3D11::SaveSettingFile(const char *settingFile){
	if(!settingFile)
		return;
#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	//windows store and phone style
	auto File =  HQWinStoreFileSystem::OpenOrCreateFileForWrite(settingFile);
	if (File == nullptr)
	{
		return;
	}

	wchar_t buffer[256];

	File->UnicodeEncoding = Windows::Storage::Streams::UnicodeEncoding::Utf8;
	
	swprintf(buffer, 255, L"Basic Settings\n"); 
	File->WriteString(ref new Platform::String(buffer));
	
	swprintf(buffer, 255, L"Width=%u\n",selectedMode->Width);
	File->WriteString(ref new Platform::String(buffer));
	swprintf(buffer, 255, L"Height=%u\n",selectedMode->Height);
	File->WriteString(ref new Platform::String(buffer));
	swprintf(buffer, 255, L"Windowed=%d\n",(int)windowed);
	File->WriteString(ref new Platform::String(buffer));
	swprintf(buffer, 255, L"RefreshRate=%u\n",(UINT)(selectedMode->RefreshRate.Numerator / selectedMode->RefreshRate.Denominator));
	File->WriteString(ref new Platform::String(buffer));
	swprintf(buffer, 255, L"VSync=%d\n\n",this->vsync);
	File->WriteString(ref new Platform::String(buffer));
	
	swprintf(buffer, 255, L"Advanced Settings\n");
	File->WriteString(ref new Platform::String(buffer));
	swprintf(buffer, 255, L"Adapter=%d\n",selectedAdapter->adapterIndex);
	File->WriteString(ref new Platform::String(buffer));
	if(selectedDriverType == D3D_DRIVER_TYPE_HARDWARE)
	{
		swprintf(buffer, 255, L"Device Type=%d\n",1);
	}
	else
	{
		swprintf(buffer, 255, L"Device Type=%d\n",2);
	}
	File->WriteString(ref new Platform::String(buffer));
	swprintf(buffer, 255, L"BackBuffer Format=%d\n",(int)helper::GetFormat(selectedMode->Format));
	File->WriteString(ref new Platform::String(buffer));
	swprintf(buffer, 255, L"DepthStencil Format=%d\n",(int)helper::GetFormat(selectedDepthStencilFmt));
	File->WriteString(ref new Platform::String(buffer));
	swprintf(buffer, 255, L"Vertex Processing=%d\n",this->unUseValue[0]);
	File->WriteString(ref new Platform::String(buffer));
	swprintf(buffer, 255, L"Multisample Type=%u\n",this->selectedMulSamplesCount > 1 ?this->selectedMulSamplesCount : 0);
	File->WriteString(ref new Platform::String(buffer));
	
	HQWinStoreUtil::Wait(File->StoreAsync());
	HQWinStoreUtil::Wait(File->FlushAsync());
	//File->Close();
	HQ_DELETE (File);
#else
	//standard C file style
	FILE *save=0;
	save=fopen(settingFile,"w");
	if(!save)
		return;
	fprintf(save,"Basic Settings\n");
	fprintf(save,"Width=%u\n",selectedMode->Width);
	fprintf(save,"Height=%u\n",selectedMode->Height);
	fprintf(save,"Windowed=%d\n",(int)windowed);
	fprintf(save,"RefreshRate=%u\n",(UINT)(selectedMode->RefreshRate.Numerator / selectedMode->RefreshRate.Denominator));
	fprintf(save,"VSync=%d\n\n",this->vsync);

	fprintf(save,"Advanced Settings\n");
	fprintf(save,"Adapter=%d\n",selectedAdapter->adapterIndex);
	if(selectedDriverType == D3D_DRIVER_TYPE_HARDWARE)
		fprintf(save,"Device Type=%d\n",1);
	else
		fprintf(save,"Device Type=%d\n",2);
	fprintf(save,"BackBuffer Format=%d\n",(int)helper::GetFormat(selectedMode->Format));
	fprintf(save,"DepthStencil Format=%d\n",(int)helper::GetFormat(selectedDepthStencilFmt));
	fprintf(save,"Vertex Processing=%d\n",this->unUseValue[0]);
	fprintf(save,"Multisample Type=%u\n",this->selectedMulSamplesCount > 1 ?this->selectedMulSamplesCount : 0);

	fclose(save);
#endif//#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
}

/*----------------return selected adapter, if it is the default adapter return NULL-------------------------------*/
IDXGIAdapter* HQDeviceEnumD3D11::GetSelectedAdapter()
{
	if (this->selectedAdapter == NULL || ((this->selectedAdapter == &this->adapterlist.front()) && this->selectedDriverType != D3D_DRIVER_TYPE_HARDWARE))
		return NULL;
	//D3D_DRIVER_TYPE_HARDWARE is always the first adapter, we can return it instead of NULL
	return this->selectedAdapter->adapter;
}

/*------------------return driver unknown if default adapter is not selected--------------------------*/
D3D_DRIVER_TYPE HQDeviceEnumD3D11::GetSelectedDriverType()
{
	if (this->selectedAdapter == NULL || ((this->selectedAdapter == &this->adapterlist.front()) && this->selectedDriverType != D3D_DRIVER_TYPE_HARDWARE))
		return this->selectedDriverType;

	//D3D_DRIVER_TYPE_HARDWARE is always the first adapter, we can return D3D_DRIVER_TYPE_UNKNOWN so the actual adapter can be passed to D3D11CreateDevice instead of NULL
	return D3D_DRIVER_TYPE_UNKNOWN;
}

const DXGI_ADAPTER_DESC& HQDeviceEnumD3D11::GetSelectedAdapterDesc() const
{
	return this->selectedAdapter->adapterDesc;
}

//************************************************
//get all available fullscreen display resolutions
//************************************************
struct BooleanWrapper
{
private:
	bool val;
public:
	BooleanWrapper()
	{
		val = false;
	}
	BooleanWrapper(bool val)
	{
		this->val = val;
	}
	BooleanWrapper(const BooleanWrapper& src)
	{
		this->val = src.val;
	}

	bool operator== (bool val)
	{
		return this->val == val;
	}
};
class Less
{
public:
	bool operator() (const HQResolution & lhs , const HQResolution & rhs) const
	{
		if (lhs.width < rhs.width)
			return true;
		if (lhs.width > rhs.width)
			return false;
		if (lhs.height < rhs.height)
			return true;
		return false;
	}

};

void HQDeviceEnumD3D11::GetAllDisplayResolution(HQResolution *resolutionList , hq_uint32& numResolutions)
{
	HQResolution res;

	std::map<HQResolution , BooleanWrapper , Less> existList;
	
	if (resolutionList != NULL)//save resolutions to array
	{
		hquint32 j = 0;
		for(std::list<DXGI_MODE_DESC>::iterator i = this->selectedAdapter->dModelist.begin();
			i!= selectedAdapter->dModelist.end() && j < numResolutions ; ++i)
		{
			res.width = i->Width;
			res.height = i->Height;
			BooleanWrapper &exist = existList[res];
			if (exist == false)//độ phân giải này chưa cho vào list
			{
				exist = true;
				resolutionList[j++] = res;
			}
		}
	}
	else//count max number of resolutions
	{
		numResolutions = 0;
		for(std::list<DXGI_MODE_DESC>::iterator i = this->selectedAdapter->dModelist.begin();
			i!= selectedAdapter->dModelist.end() ; ++i )
		{
			res.width = i->Width;
			res.height = i->Height;
			BooleanWrapper &exist = existList[res];
			if (exist == false)//độ phân giải này chưa cho vào list
			{
				exist = true;
				++numResolutions;
			}
		}
	}
}
//****************************************
//change display mode
//****************************************
bool HQDeviceEnumD3D11::ChangeSelectedDisplay(hq_uint32 width,hq_uint32 height,bool windowed)
{
	DXGI_MODE_DESC *dmode = NULL;
	//tìm trong danh sách độ phân giải chuẩn
	double minDiff = 99999;
	double diff;
	//find closest match
	for(std::list<DXGI_MODE_DESC>::iterator i = this->selectedAdapter->dModelist.begin();
		i!= selectedAdapter->dModelist.end() ; ++i)
	{
		if(i->Width == width && i->Height == height && 
			i->Format == this->selectedMode->Format )
		{
			diff = fabs((double)i->RefreshRate.Numerator / i->RefreshRate.Denominator -
			(double)this->selectedMode->RefreshRate.Numerator / this->selectedMode->RefreshRate.Denominator);
			if(diff < minDiff)
			{
				minDiff = diff;
				dmode=&(*i);
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

	if(dmode != NULL)
	{
		this->selectedMode = dmode;
		return true;
	}
	return false;
}
