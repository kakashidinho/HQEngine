/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _HQ_BASE_RENDER_TARGET_MANAGER_H_
#define _HQ_BASE_RENDER_TARGET_MANAGER_H_

#include "../HQReturnVal.h"
#include "../HQRenderTargetManager.h"
#include "../HQLoggableObject.h"
#include "HQTextureManagerBaseImpl.h"
#include "../HQItemManager.h"
#include "string.h"

struct HQBaseCustomRenderBuffer
{
	HQBaseCustomRenderBuffer(hq_uint32 width ,hq_uint32 height ,
		HQMultiSampleType multiSampleType)
	{
		this->width = width;
		this->height = height;
		this->multiSampleType = multiSampleType;
	}
	virtual ~HQBaseCustomRenderBuffer() {
	}
	virtual void OnLostDevice() {}
	virtual void OnResetDevice() {}
	bool IsTexture() {return this->GetTextureID() != HQ_NOT_AVAIL_ID;}
	virtual HQSharedPtr<HQTexture> GetTexture() {return HQSharedPtr<HQTexture>::null;};
	virtual hq_uint32 GetTextureID() {return HQ_NOT_AVAIL_ID;}
	virtual void *GetData() = 0;
	
	HQMultiSampleType multiSampleType;
	hq_uint32 width , height;
};



struct HQBaseRenderTargetTexture : public HQBaseCustomRenderBuffer
{
	HQBaseRenderTargetTexture(hq_uint32 _width ,hq_uint32 _height ,
							  HQMultiSampleType _multiSampleType,
							  hq_uint32 numMipmaps,
							  hq_uint32 textureID , HQSharedPtr<HQTexture> pTex)
		: HQBaseCustomRenderBuffer(_width , _height , 
								_multiSampleType) ,
		  pTexture(pTex)
	{
		this->numMipmaps = numMipmaps;
		this->textureID = textureID;
	}
	~HQBaseRenderTargetTexture(){
	}
	HQSharedPtr<HQTexture> GetTexture() {return pTexture;}
	hq_uint32 GetTextureID() {return textureID;}

	hq_uint32 numMipmaps;
	hq_uint32 textureID;
	HQSharedPtr<HQTexture> pTexture;
};

struct HQActiveRenderTarget
{
	HQActiveRenderTarget()
	{
		pRenderTarget = HQSharedPtr<HQBaseCustomRenderBuffer>::null;
		cubeFace = HQ_CTF_POS_X;
	};
	HQSharedPtr<HQBaseCustomRenderBuffer> pRenderTarget;
	HQCubeTextureFace cubeFace;
};

class HQSavedActiveRenderTargetsImpl: public HQSavedActiveRenderTargets
{
public:
	HQSavedActiveRenderTargetsImpl(hquint32 maxNumActiveRenderTargets)
	{
		this->activeRenderTargets =  HQ_NEW HQActiveRenderTarget[maxNumActiveRenderTargets];
		numActiveRenderTargets = 0;
	}

	~HQSavedActiveRenderTargetsImpl()
	{
		delete[] this->activeRenderTargets;
	}

	const HQActiveRenderTarget& operator [] (hquint32 index) const
	{
		return activeRenderTargets[index];
	}

	HQActiveRenderTarget& operator [] (hquint32 index)
	{
		return activeRenderTargets[index];
	}


public:
	hquint32			   numActiveRenderTargets;
	HQSharedPtr<HQBaseCustomRenderBuffer> pActiveDepthStencilBuffer;
private:
	HQActiveRenderTarget * activeRenderTargets;
};

class HQBaseRenderTargetManager: public HQRenderTargetManager, public HQLoggableObject
{
protected:
	bool currentUseDefaultBuffer;//is currently using default back buffer and depth stencil buffer
	
	hq_uint32 numActiveRenderTargets;//number of current active render targets;
	hq_uint32 maxActiveRenderTargets;

	HQSharedPtr<HQBaseCustomRenderBuffer> pActiveDepthStencilBuffer;//current active depth stencil buffer
	HQActiveRenderTarget *activeRenderTargets;

	HQItemManager<HQBaseCustomRenderBuffer> renderTargets;
	HQItemManager<HQBaseCustomRenderBuffer> depthStencilBuffers;
	
	
	
	hq_uint32 renderTargetWidth , renderTargetHeight;//current active render target 's size
	
	HQBaseTextureManager *pTextureManager;
	
	static void GetString(HQRenderTargetFormat format , char *str)
	{
		switch(format)
		{
		case HQ_RTFMT_R_FLOAT32:
			strcpy(str , "HQ_RTFMT_R_FLOAT32");
			break;
		case HQ_RTFMT_R_FLOAT16:
			strcpy(str , "HQ_RTFMT_R_FLOAT16");
			break;
		case HQ_RTFMT_RGBA_32:
			strcpy(str , "HQ_RTFMT_RGBA_32");
			break;
		case HQ_RTFMT_A_UINT8:
			strcpy(str , "HQ_RTFMT_A_UINT8");
			break;
		case HQ_RTFMT_R_UINT8:
			strcpy(str , "HQ_RTFMT_R_UINT8");
			break;
		}
	}
	static void GetString(HQDepthStencilFormat format , char *str)
	{
		switch(format)
		{
		case HQ_DSFMT_DEPTH_16:
			strcpy(str , "HQ_DSFMT_DEPTH_16");
			break;
		case HQ_DSFMT_DEPTH_24:
			strcpy(str , "HQ_DSFMT_DEPTH_24");
			break;
		case HQ_DSFMT_DEPTH_32:
			strcpy(str , "HQ_DSFMT_DEPTH_32");
			break;
		case HQ_DSFMT_STENCIL_8:
			strcpy(str , "HQ_DSFMT_STENCIL_8");
			break;
		case HQ_DSFMT_DEPTH_24_STENCIL_8:
			strcpy(str , "HQ_DSFMT_DEPTH_24_STENCIL_8");
			break;
		}
	}
	static void GetString(HQMultiSampleType type , char *str)
	{
		switch(type)
		{
		case HQ_MST_NONE:
			strcpy(str , "HQ_MST_NONE");
			break;
		case HQ_MST_2_SAMPLES:
			strcpy(str , "HQ_MST_2_SAMPLES");
			break;
		case HQ_MST_4_SAMPLES:
			strcpy(str , "HQ_MST_4_SAMPLES");
			break;
		case HQ_MST_8_SAMPLES:
			strcpy(str , "HQ_MST_8_SAMPLES");
			break;
		}
	}

public:
	HQBaseRenderTargetManager(hq_uint32 maxActiveRenderTargets, 
							  HQBaseTextureManager *pTexMan,
							  HQLogStream* logFileStream , const char *logPrefix ,
							  bool flushLog)
		 :HQLoggableObject(logFileStream , logPrefix , flushLog)
	{
		this->currentUseDefaultBuffer = true;
		this->numActiveRenderTargets = 0;
		this->maxActiveRenderTargets = maxActiveRenderTargets;
		this->pTextureManager = pTexMan;

		if(maxActiveRenderTargets > 0)
		{
			this->activeRenderTargets = HQ_NEW HQActiveRenderTarget[maxActiveRenderTargets];
		}
		else
		{
			this->activeRenderTargets = NULL;
		}
	}
	~HQBaseRenderTargetManager() {
		SafeDeleteArray(this->activeRenderTargets);
	};
	
	virtual void OnLostDevice() {}
	virtual void OnResetDevice() {}

	inline bool IsUsingDefaultFrameBuffer()
	{
		return currentUseDefaultBuffer;
	}
	
	inline hq_uint32 GetRTWidth () {return renderTargetWidth ;}
	inline hq_uint32 GetRTHeight () {return renderTargetHeight ;}

	virtual void OnBackBufferResized(hquint32 width, hquint32 height) {
		if (IsUsingDefaultFrameBuffer())
		{
			renderTargetWidth = width;
			renderTargetHeight = height;
		}	
	}

	inline hq_uint32 GetNumActiveRenderTargets()
	{
		return numActiveRenderTargets;
	}

	HQSavedActiveRenderTargets* CreateAndSaveRenderTargetsList()
	{
		HQSavedActiveRenderTargetsImpl *savedList = HQ_NEW HQSavedActiveRenderTargetsImpl(this->maxActiveRenderTargets);

		this->SaveRenderTargetsList(savedList);

		return savedList;
	}

	HQReturnVal SaveRenderTargetsList(HQSavedActiveRenderTargets* savedList)
	{
		HQSavedActiveRenderTargetsImpl *savedListImpl = dynamic_cast<HQSavedActiveRenderTargetsImpl *> (savedList);

		if (savedListImpl == NULL)
			return HQ_FAILED;


		//copy the active render targets info
		for (hquint32 i = 0; i < maxActiveRenderTargets; ++i)
			(*savedListImpl)[i] = this->activeRenderTargets[i];

		savedListImpl->pActiveDepthStencilBuffer = this->pActiveDepthStencilBuffer;
		savedListImpl->numActiveRenderTargets = this->numActiveRenderTargets;

		return HQ_OK;
	}

	HQReturnVal RestoreRenderTargets(const HQSavedActiveRenderTargets *savedList)
	{
		const HQSavedActiveRenderTargetsImpl *savedListImpl = dynamic_cast<const HQSavedActiveRenderTargetsImpl *> (savedList);

		if (savedListImpl == NULL)
			return HQ_FAILED;

		return RestoreRenderTargetsImpl(*savedListImpl);
	}


	HQReturnVal RemoveRenderTarget(hq_uint32 renderTargetID)
	{
		HQSharedPtr<HQBaseCustomRenderBuffer> pRenderTarget = renderTargets.GetItemPointer(renderTargetID);
		if(pRenderTarget == NULL)
			return HQ_FAILED_INVALID_ID;
		
		pTextureManager->RemoveTexture(pRenderTarget->GetTextureID());
		return (HQReturnVal)renderTargets.Remove(renderTargetID);
	}
	void RemoveAllRenderTarget()
	{
		this->ActiveRenderTargets(NULL , HQ_NOT_AVAIL_ID , 0);
		
		HQItemManager<HQBaseCustomRenderBuffer>::Iterator ite;

		this->renderTargets.GetIterator(ite);

		while (!ite.IsAtEnd())
		{
			this->pTextureManager->RemoveTexture(ite->GetTextureID());
			++ite;
		}
		this->renderTargets.RemoveAll();
	}

	HQReturnVal RemoveDepthStencilBuffer(hq_uint32 bufferID)
	{
		return (HQReturnVal)depthStencilBuffers.Remove(bufferID);
	}
	void RemoveAllDepthStencilBuffer()
	{
		this->depthStencilBuffers.RemoveAll();
	}

	//implement dependent
	virtual HQReturnVal RestoreRenderTargetsImpl(const HQSavedActiveRenderTargetsImpl &savedList) = 0;
};


class DummyRenderTargetManager : public HQBaseRenderTargetManager
{
public:
	DummyRenderTargetManager()
		:HQBaseRenderTargetManager(0 , NULL , NULL,"",false)
	{
	}
	HQReturnVal GenerateMipmaps(hq_uint32 renderTargetTextureID)
	{
		return HQ_FAILED;
	}

	HQReturnVal CreateRenderTargetTexture(hq_uint32 width , hq_uint32 height,
								  bool hasMipmaps,
								  HQRenderTargetFormat format , 
								  HQMultiSampleType multisampleType,
								  HQTextureType textureType,
								  hq_uint32 *pRenderTargetID_Out,
								  hq_uint32 *pTextureID_Out)
	{
		return HQ_FAILED;
	}
	
	HQReturnVal CreateDepthStencilBuffer(hq_uint32 width , hq_uint32 height,
										HQDepthStencilFormat format,
										HQMultiSampleType multisampleType,
										hq_uint32 *pDepthStencilBufferID_Out)
	{
		return HQ_FAILED;
	}
	HQReturnVal ActiveRenderTarget(const HQRenderTargetDesc &renderTargetDesc , 
									hq_uint32 depthStencilBufferID )
	{
		return HQ_FAILED;
	}
	HQReturnVal ActiveRenderTargets(const HQRenderTargetDesc *renderTargetDescs , 
									hq_uint32 depthStencilBufferID ,
								   hq_uint32 numRenderTargets = 1//number of render targers
								   )
	{
		return HQ_FAILED;
	}

	virtual HQReturnVal RestoreRenderTargetsImpl(const HQSavedActiveRenderTargetsImpl &savedList) { return HQ_FAILED; }
};

#endif
