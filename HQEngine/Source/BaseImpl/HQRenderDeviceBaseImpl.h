/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef RDEVICE_BASE_IMPL
#define RDEVICE_BASE_IMPL
#include "../HQRenderDevice.h"
#include "HQBaseImplCommon.h"
#include "../HQLoggableObject.h"


class HQBaseRenderDevice : public HQRenderDevice, public HQA16ByteObject ,public HQLoggableObject
{
protected:
	//description string
	char *desc;

	hq_uint32 sWidth;//back buffer width
	hq_uint32 sHeight;//back buffer height
	HQViewPort currentVP;//current viewport

	char * settingFileDir;
	
	hq_uint32 flags;
public:
	HQBaseRenderDevice(const char *desc , const char *logPrefix , bool flushLog)
		:HQLoggableObject(logPrefix , flushLog)
	{
		this->flags = 0;
		this->settingFileDir = NULL;
		this->desc = HQ_NEW char [strlen(desc) + 1];
		strcpy(this->desc , desc);
	}

	~HQBaseRenderDevice()
	{
		SafeDeleteArray(this->settingFileDir);
		SafeDeleteArray(this->desc);
	}
	
	bool IsRunning(){return ((flags & RUNNING)!=0);};
	bool IsDeviceLost() {return false;}
	
	hq_uint32 GetWidth(){return sWidth;};
	hq_uint32 GetHeight(){return sHeight;};
	bool IsWindowed() {return ((flags & WINDOWED)!=0);};
	bool IsVSyncEnabled() {return ((flags & VSYNC_ENABLE)!=0);};

	inline hq_uint32 GetFlags() {return this->flags;}
	
	const HQViewPort & GetViewPort()  const {return this->currentVP;}
	
	void CopySettingFileDir(const char *settingFileDir)
	{
		hq_uint32 len = strlen(settingFileDir);
		SafeDeleteArray(this->settingFileDir);
		this->settingFileDir = HQ_NEW char [len + 1];
		strcpy(this->settingFileDir , settingFileDir);
	}
	/*------------------------*/
	hq_uint32 GetMaxVertexAttribs() {return 16;}//common value
	bool IsVertexAttribDataTypeSupported(HQVertexAttribDataType dataType)
	{
		return true;
	}
	//normally in d3d9 and opengl: sampler units = texture units
	hq_uint32 GetMaxShaderTextures()
	{
		return GetMaxShaderSamplers();
	}
	hq_uint32 GetMaxShaderStageTextures(HQShaderType shaderStage)
	{
		return GetMaxShaderStageSamplers(shaderStage);
	}
	/*------------------------*/
	void GetScreenCoordr(const HQMatrix4 &viewProj , const HQVector4& vPos  , HQPoint<hqint32> &pointOut)//truy vấn tọa độ của 1 điểm trong hệ tọa độ màn hình từ 1 điểm có tọa độ vPos trong hệ tọa độ thế giới
	{
		hq_float32 X,Y,W_over_2,H_over_2;
		
		X=(hq_float32)this->currentVP.x;
		Y=(hq_float32)this->currentVP.y;
		W_over_2=this->currentVP.width/2.0f;
		H_over_2=this->currentVP.height/2.0f;
		
		HQ_DECL_STACK_VECTOR4( projVec );//tọa độ điểm sau khi nhân ma trận viewproj

		HQVector4TransformCoord(&vPos,&viewProj,&projVec);
		
		hq_float32 invW=1.0f/projVec.w;
		//chuyển từ hệ tọa độ chuẩn hóa [-1,1] sang hệ tọa độ màn hình
		pointOut.x=(long)(X + (1.0f + projVec.x * invW)*W_over_2);
		pointOut.y=(long)(Y + (1.0f - projVec.y * invW)*H_over_2);
	}

	void GetRayr(const HQMatrix4 &view ,const HQMatrix4 &proj , hq_float32 zNear,
				const HQPoint<hqint32>& point, HQRay3D & rayOut)
	{
		HQ_DECL_STACK_2VAR(HQVector4, projVec, HQMatrix4, invView);

		hq_float32 X,Y,_2_over_W,_2_over_H;
		
		projVec.w=zNear;

		X=(hq_float32)this->currentVP.x;
		Y=(hq_float32)this->currentVP.y;
		_2_over_W=2.0f / this->currentVP.width;
		_2_over_H=2.0f / this->currentVP.height;
	

		//từ hệ tọa độ màn hình sang hệ tọa độ chuẩn hóa của device
		projVec.x=_2_over_W*(point.x - X) - 1;
		projVec.y=_2_over_H*(Y - point.y) + 1;

		//từ hệ tọa độ chuẩn hóa sang hệ tọa độ nhìn
		projVec.x*=(projVec.w / proj._11) ;
		projVec.y*=(projVec.w / proj._22) ;
		projVec.z=zNear;

		//từ hệ tọa độ nhìn sang hệ tọa độ thế giới
		HQMatrix4Inverse(&view,&invView);

		HQVector4TransformCoord(&HQVector4::Origin(),&invView,&(rayOut.O));
		HQVector4TransformNormal(&projVec,&invView,&(rayOut.D));
	}
	
	const char * GetDeviceDesc()
	{
		return this->desc;
	}
};

#endif
