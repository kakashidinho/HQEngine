/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef RDEVICE_BASE_IMPL
#define RDEVICE_BASE_IMPL
#include "../HQRenderDevice.h"
#include "../HQUtilMath.h"
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
	
	const char * GetDeviceDesc()
	{
		return this->desc;
	}
};

#endif
