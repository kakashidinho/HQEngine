/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _DEV_ENUM_
#define _DEV_ENUM_
#include <d3d11.h>
#include "../HQRendererCoreType.h"


#include <list>

struct AdapterInfo{
	AdapterInfo(UINT index)
	{
		adapterIndex = index;
		adapter = NULL;
	}
	~AdapterInfo()
	{
		if(adapter)
			adapter->Release();
	}
	UINT adapterIndex;
	IDXGIAdapter* adapter;
	DXGI_ADAPTER_DESC adapterDesc;//mô tả của adapter
	std::list<DXGI_MODE_DESC> dModelist;//display mode list
};

class HQDeviceEnumD3D11{
private:
	std::list<AdapterInfo> adapterlist;
	
	IDXGIFactory * pFactory;

	DXGI_MODE_DESC customWindowedMode;//lưu tùy chọn dành cho chế độ windowed , không giới hạn kích thước width x height như chế độ fullscreen

	int unUseValue[1];

	AdapterInfo* selectedAdapter;
	D3D_DRIVER_TYPE selectedDriverType;
public:
	HQDeviceEnumD3D11(IDXGIFactory * pFactory);
	
	void ParseSettingFile(const char *settingFile);
	void SaveSettingFile(const char* settingFile);
	void Enum();
	void GetAllDisplayResolution(HQResolution *resolutionList , hq_uint32& numResolutions);
	bool ChangeSelectedDisplay(hq_uint32 width,hq_uint32 height,bool windowed);

	IDXGIAdapter* GetSelectedAdapter();
	const DXGI_ADAPTER_DESC& GetSelectedAdapterDesc() const;
	D3D_DRIVER_TYPE GetSelectedDriverType();
	/*-----------------------------*/
	DXGI_MODE_DESC* selectedMode;
	int vsync;//vsync enable or not
	DXGI_FORMAT selectedDepthStencilFmt;
	UINT selectedMulSamplesCount;
	bool windowed;
};
#endif
