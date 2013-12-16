#ifndef _DEV_ENUM_
#define _DEV_ENUM_
#include <d3d9.h>
#include "../HQRendererCoreType.h"
#include "../HQLinkedList.h"

#define D3DFMT_DXT1_SUPPORT 0x1
#define D3DFMT_DXT3_SUPPORT 0x2
#define D3DFMT_DXT5_SUPPORT 0x4
#define D3DFMT_DXT1_CUBE_SUPPORT 0x8
#define D3DFMT_DXT3_CUBE_SUPPORT 0x10
#define D3DFMT_DXT5_CUBE_SUPPORT 0x20

struct HQD3DCaps9 : public D3DCAPS9
{
	HQD3DCaps9()
		: s3tc_dxtFlags(0)
	{
	}
	DWORD s3tc_dxtFlags;
};

struct ComboInfo{
	hq_uint32 adapter;//chỉ số của adapter 
	D3DDEVTYPE devType;//dạng device :HAL,REF
	bool windowed;//chế độ cửa sổ?
	D3DFORMAT displayFmt;//display format
	D3DFORMAT backFmt;//backbuffer format
	D3DFORMAT depthStencilFmt;//Depth/Stencil Format

	DWORD vertProcBehavior;//xử lý đỉnh software,hardware,hay kết hợp cả 2?
	D3DMULTISAMPLE_TYPE maxMulSampType;//kiểu siêu lấy mẫu tốt nhất có thể đạt dc 
};

struct DeviceInfo{
	hq_uint32 adapter;//chỉ số của adapter 
	D3DDEVTYPE devType;//dạng device :HAL,REF
	HQD3DCaps9 dCaps;//các khả năng của device
	UINT numPixelShaderSamplers;
	UINT numVertexShaderSamplers;
	HQLinkedList<ComboInfo> comboList;
};

struct AdapterInfo{
	hq_uint32 adapter;//chỉ số của adapter 
	D3DADAPTER_IDENTIFIER9 adapterID;//adapter ID
	HQLinkedList<D3DDISPLAYMODE> dModelist;//display mode
	HQLinkedList<DeviceInfo> dev;//các dạng device (HAL,REF)
};

struct HQDeviceEnumD3D9{
	/*--methods---*/
	HQDeviceEnumD3D9();

	bool CheckDepthStencilFmt(IDirect3D9 * d3d,ComboInfo & combo);
	
	void ParseSettingFile(const char *settingFile,IDirect3D9* d3d);
	void SaveSettingFile(const char* settingFile);
	void Enum(IDirect3D9* d3d);
	void GetAllDisplayResolution(HQResolution *resolutionList , hq_uint32& numResolutions);
	bool ChangeSelectedDisplay(hq_uint32 width,hq_uint32 height,bool windowed);

	/*--------attributes----------*/
	HQLinkedList<AdapterInfo> adapterlist;
	

	D3DDISPLAYMODE customWindowedMode;//lưu tùy chọn dành cho chế độ windowed , không giới hạn kích thước width x height như chế độ fullscreen
	D3DDISPLAYMODE* selectedMode;
	int vsync;//vsync enable or not
	AdapterInfo* selectedAdapter;
	DeviceInfo* selectedDevice;
	ComboInfo* selectedCombo;
	D3DMULTISAMPLE_TYPE selectedMulSampleType;
	D3DFORMAT selectedDepthStencilFmt;
	DWORD selectedVertexProcBehavior;
};
#endif