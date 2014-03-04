/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
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

struct HQRenderTargetInfo
{
	HQRenderTargetInfo()
	{
		pRenderTarget = HQSharedPtr<HQBaseCustomRenderBuffer>::null;
		cubeFace = HQ_CTF_POS_X;
	};
	HQSharedPtr<HQBaseCustomRenderBuffer> pRenderTarget;
	HQCubeTextureFace cubeFace;
};

struct HQBaseRenderTargetGroup
{
	HQBaseRenderTargetGroup(hquint32 _numRenderTargets)
		: numRenderTargets(_numRenderTargets),
		  renderTargets(HQ_NEW HQRenderTargetInfo[numRenderTargets])
	{
	}

	virtual ~HQBaseRenderTargetGroup()
	{
		delete[] this->renderTargets;
	}

	const HQRenderTargetInfo& operator [] (hquint32 index) const
	{
		return renderTargets[index];
	}

	HQRenderTargetInfo& operator [] (hquint32 index)
	{
		return renderTargets[index];
	}

	const hquint32			   numRenderTargets;
	HQSharedPtr<HQBaseCustomRenderBuffer> pDepthStencilBuffer;
	hquint32 commonWidth;//common width of render targets in group
	hquint32 commonHeight;//common width of render targets in group

	HQRenderTargetInfo * const renderTargets;
};

class HQBaseRenderTargetManager: public HQRenderTargetManager, public HQLoggableObject
{
protected:
	bool currentUseDefaultBuffer;//is currently using default back buffer and depth stencil buffer
	
	hquint32 currentActiveRTGroup;//current active render targets group
	hq_uint32 maxActiveRenderTargets;

	HQItemManager<HQBaseCustomRenderBuffer> renderTargets;
	HQItemManager<HQBaseCustomRenderBuffer> depthStencilBuffers;
	HQItemManager<HQBaseRenderTargetGroup> rtGroups;
	
	
	
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
		this->currentActiveRTGroup = HQ_NULL_ID;
		this->maxActiveRenderTargets = maxActiveRenderTargets;
		this->pTextureManager = pTexMan;
	}
	~HQBaseRenderTargetManager() {
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
		HQBaseRenderTargetGroup* group = rtGroups.GetItemRawPointer(this->currentActiveRTGroup);
		if (group == NULL)
			return 1;
		return group->numRenderTargets;
	}


	HQReturnVal RemoveRenderTarget(hq_uint32 renderTargetID)
	{
		HQSharedPtr<HQBaseCustomRenderBuffer> pRenderTarget = renderTargets.GetItemPointer(renderTargetID);
		if(pRenderTarget == NULL)
			return HQ_FAILED_INVALID_ID;
		
		//remove associated texture
		pTextureManager->RemoveTexture(pRenderTarget->GetTextureID());
		return (HQReturnVal)renderTargets.Remove(renderTargetID);
	}

	void RemoveAllRenderTarget()
	{
		this->ActiveRenderTargets(HQ_NOT_AVAIL_ID);
		
		HQItemManager<HQBaseCustomRenderBuffer>::Iterator ite;

		this->renderTargets.GetIterator(ite);

		//remove associated textures
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

	HQReturnVal RemoveRenderTargetGroup(hq_uint32 groupID){
		if (groupID == this->currentActiveRTGroup)
			this->ActiveRenderTargets(HQ_NULL_ID);
		return (HQReturnVal)rtGroups.Remove(groupID);
	}
	void RemoveAllRenderTargetGroup() {
		this->ActiveRenderTargets(HQ_NULL_ID);
		rtGroups.RemoveAll();
	}

	HQReturnVal CreateRenderTargetGroup(const HQRenderTargetDesc *renderTargetDescs , 
									hq_uint32 depthStencilBufferID ,
									hq_uint32 numRenderTargets,//number of render targers
									hq_uint32 *pRenderTargetGroupID_out
									)
	{
		HQBaseRenderTargetGroup* newGroup = NULL;
		//call sub class method
		HQReturnVal re = this->CreateRenderTargetGroupImpl(renderTargetDescs, depthStencilBufferID, numRenderTargets, &newGroup);
		
		if (HQFailed(re))
		{
			return re;
		}

		if (!this->rtGroups.AddItem(newGroup, pRenderTargetGroupID_out)){
			delete newGroup;
			return HQ_FAILED_MEM_ALLOC;
		}

		return re;
	}

	///
	///Set the render targets in group {renderTargetGroupID} as main render targets	
	///
	HQReturnVal ActiveRenderTargets(hquint32 renderTargetGroupID) {
		HQSharedPtr<HQBaseRenderTargetGroup> group = this->rtGroups.GetItemPointer(renderTargetGroupID);

		if (group == NULL)
		{
			this->currentActiveRTGroup = HQ_NULL_ID;
		}
		else
			this->currentActiveRTGroup = renderTargetGroupID;

		//call sub class method
		return this->ActiveRenderTargetsImpl(group);
	}

	///
	///Get current render targets group
	///
	virtual hquint32 GetActiveRenderTargets() {
		return this->currentActiveRTGroup;
	}

	//implement dependent
	virtual HQReturnVal ActiveRenderTargetsImpl(HQSharedPtr<HQBaseRenderTargetGroup>& group) = 0;
	virtual HQReturnVal CreateRenderTargetGroupImpl(const HQRenderTargetDesc *renderTargetDescs , 
									hq_uint32 depthStencilBufferID ,
									hq_uint32 numRenderTargets,//number of render targers
									HQBaseRenderTargetGroup **ppRenderTargetGroupOut
									) = 0;
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


	virtual HQReturnVal ActiveRenderTargetsImpl(HQSharedPtr<HQBaseRenderTargetGroup>& group){
		return HQ_FAILED;
	}

	virtual HQReturnVal CreateRenderTargetGroupImpl(const HQRenderTargetDesc *renderTargetDescs , 
									hq_uint32 depthStencilBufferID ,
									hq_uint32 numRenderTargets,//number of render targers
									HQBaseRenderTargetGroup **ppRenderTargetGroupOut
									)
	{
		return HQ_FAILED;
	}
};

#endif
