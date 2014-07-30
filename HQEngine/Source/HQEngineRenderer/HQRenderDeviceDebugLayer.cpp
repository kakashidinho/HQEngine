/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQPlatformDef.h"
#include "../HQEngineCustomHeap.h"
#include "HQRendererDeviceDebugLayer.h"
#include "HQReturnValDebugString.h"

char debugString[2048];

#if defined WIN32 || (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
#	define sprintf _snprintf
#	define __OutputDebugString OutputDebugStringA( debugString )

#else
#	define sprintf snprintf
#	define __OutputDebugString printf( debugString )

#endif


void HQRenderDeviceDebugLayer::SetDevice(HQRenderDevice * pDevice)
{
	this->m_pDevice = pDevice;
	if (this->m_pDevice != NULL)
	{
		this->textureMan = m_pDevice->GetTextureManager();
		this->vStreamMan = m_pDevice->GetVertexStreamManager();
		this->shaderMan = m_pDevice->GetShaderManager();
		this->renderTargetMan = m_pDevice->GetRenderTargetManager();
		this->stateMan = m_pDevice->GetStateManager();
	}
}

HQReturnVal HQRenderDeviceDebugLayer::Init(HQRenderDeviceInitInput input,const char* settingFileDir ,HQLogStream* logFileStream , const char *additionalSettings)
{
	HQReturnVal re = m_pDevice->Init(input , settingFileDir , logFileStream , additionalSettings);
	
	if (HQFailed(re))
	{
		if (additionalSettings != NULL)
			sprintf(debugString , 2048 , "HQRenderDevice::Init(%p , \"%s\" , %p , \"%s\") return %s\n" ,input , settingFileDir , logFileStream , additionalSettings , HQReturnValToString(re));
		else
			sprintf(debugString , 2048 , "HQRenderDevice::Init(%p , \"%s\" , %p , NULL) return %s\n" ,input , settingFileDir , logFileStream  , HQReturnValToString(re));
	
		__OutputDebugString;
	}

	this->textureMan = m_pDevice->GetTextureManager();
	this->vStreamMan = m_pDevice->GetVertexStreamManager();
	this->renderTargetMan = m_pDevice->GetRenderTargetManager();
	this->stateMan = m_pDevice->GetStateManager();
	this->shaderMan = m_pDevice->GetShaderManager();

	return re;
}

HQReturnVal HQRenderDeviceDebugLayer::BeginRender(HQBool clearPixel, HQBool clearDepth, HQBool clearStencil, hquint32 numRTsToClear)
{
	HQReturnVal re = m_pDevice->BeginRender(clearPixel, clearDepth, clearStencil, numRTsToClear);
	
	if (HQFailed(re))
	{
		sprintf(debugString , 2048 , "HQRenderDevice::BeginRender(%d , %d , %d , %d) return %s\n" ,
			(int)clearPixel, (int)clearDepth, (int)clearStencil, numRTsToClear, HQReturnValToString(re));
		
		__OutputDebugString;
	}
	
	return re;
}
HQReturnVal HQRenderDeviceDebugLayer::EndRender()
{
	HQReturnVal re = m_pDevice->EndRender();
	if (HQFailed(re))
	{
		sprintf(debugString , 2048 , "HQRenderDevice::EndRender() return %s\n" ,
			HQReturnValToString(re));
		
		__OutputDebugString;
	}
	return re;
}

HQReturnVal HQRenderDeviceDebugLayer::DisplayBackBuffer()
{
	HQReturnVal re = m_pDevice->DisplayBackBuffer();
	if (HQFailed(re))
	{
		sprintf(debugString , 2048 , "HQRenderDevice::DisplayBackBuffer() return %s\n" ,
			HQReturnValToString(re));
		
		__OutputDebugString;
	}
	return re;
}


HQReturnVal HQRenderDeviceDebugLayer::Clear(HQBool clearPixel, HQBool clearDepth, HQBool clearStencil, hquint32 numRTsToClear)
{
	HQReturnVal re = m_pDevice->Clear(clearPixel, clearDepth, clearStencil, numRTsToClear);
	if (HQFailed(re))
	{
		sprintf(debugString , 2048 , "HQRenderDevice::Clear(%d , %d , %d , %d) return %s\n" ,
			(int)clearPixel, (int)clearDepth, (int)clearStencil, numRTsToClear, HQReturnValToString(re));
		
		__OutputDebugString;
	}
	return re;
}
void HQRenderDeviceDebugLayer::SetClearColorf(hq_float32 red,hq_float32 green,hq_float32 blue,hq_float32 alpha)
{
	m_pDevice->SetClearColorf(red , green , blue , alpha);
}//color range:0.0f->1.0f
void HQRenderDeviceDebugLayer::SetClearColori(hq_uint32 red,hq_uint32 green,hq_uint32 blue,hq_uint32 alpha)
{
	m_pDevice->SetClearColori(red , green , blue , alpha);
}//color range:0->255
void HQRenderDeviceDebugLayer::SetClearDepthVal(hq_float32 val)
{
	m_pDevice->SetClearDepthVal(val);
}
void HQRenderDeviceDebugLayer::SetClearStencilVal(hq_uint32 val)
{
	m_pDevice->SetClearStencilVal(val);
}

HQReturnVal HQRenderDeviceDebugLayer::SetDisplayMode(hq_uint32 width,hq_uint32 height,bool windowed)
{
	HQReturnVal re = m_pDevice->SetDisplayMode(width, height ,windowed);
	if (HQFailed(re))
	{
		sprintf(debugString , 2048 , "HQRenderDevice::SetDisplayMode(%u , %u , %d) return %s\n" ,
			width, height , (int)windowed , HQReturnValToString(re));
		
		__OutputDebugString;
	}
	return re;
}//thay đổi chế độ hiển thị màn hình


HQReturnVal HQRenderDeviceDebugLayer::OnWindowSizeChanged(hq_uint32 width,hq_uint32 height)
{
	HQReturnVal re = m_pDevice->OnWindowSizeChanged(width, height);
	if (HQFailed(re))
	{
		sprintf(debugString , 2048 , "HQRenderDevice::OnWindowSizeChanged(%u , %u ) return %s\n" ,
			width, height , HQReturnValToString(re));
		
		__OutputDebugString;
	}
	return re;
}

HQReturnVal HQRenderDeviceDebugLayer::SetViewport(const HQViewPort &viewport)
{
	HQReturnVal re = m_pDevice->SetViewport(viewport);
	if (HQFailed(re))
	{
		sprintf(debugString , 2048 , "HQRenderDevice::SetViewport( {%u , %u , %u , %u} ) return %s\n" ,
			viewport.x , viewport.y , viewport.width , viewport.height , HQReturnValToString(re));
		
		__OutputDebugString;
	}
	return re;

}

HQReturnVal HQRenderDeviceDebugLayer::SetViewports(const HQViewPort * viewports, hquint32 numViewports)
{
	HQReturnVal re = m_pDevice->SetViewports(viewports, numViewports);
	if (HQFailed(re))
	{
		sprintf(debugString, 2048, "HQRenderDevice::SetViewports( {%p , %u} ) return %s\n",
			viewports, numViewports, HQReturnValToString(re));

		__OutputDebugString;
	}
	return re;
}

void HQRenderDeviceDebugLayer::EnableVSync(bool enable)
{
	m_pDevice->EnableVSync(enable);
}
void HQRenderDeviceDebugLayer::SetPrimitiveMode(HQPrimitiveMode primitiveMode) 
{
	m_pDevice->SetPrimitiveMode(primitiveMode);
}
HQReturnVal HQRenderDeviceDebugLayer::Draw(hq_uint32 vertexCount , hq_uint32 firstVertex) 
{
	HQReturnVal re = m_pDevice->Draw(vertexCount , firstVertex);
	if (HQFailed(re))
	{
		sprintf(debugString , 2048 , "HQRenderDevice::Draw(%u , %u) return %s\n" ,
			vertexCount , firstVertex , HQReturnValToString(re));
		
		__OutputDebugString;
	}
	return re;
}
HQReturnVal HQRenderDeviceDebugLayer::DrawPrimitive(hq_uint32 primitiveCount , hq_uint32 firstVertex) 
{
	HQReturnVal re = m_pDevice->DrawPrimitive(primitiveCount , firstVertex) ;
	if (HQFailed(re))
	{
		sprintf(debugString , 2048 , "HQRenderDevice::DrawPrimitive(%u , %u) return %s\n" ,
			primitiveCount , firstVertex , HQReturnValToString(re));
		
		__OutputDebugString;
	}
	return re;
}
HQReturnVal HQRenderDeviceDebugLayer::DrawIndexed(hq_uint32 numVertices , hq_uint32 indexCount , hq_uint32 firstIndex )
{
	HQReturnVal re = m_pDevice->DrawIndexed(numVertices , indexCount , firstIndex );
	if (HQFailed(re))
	{
		sprintf(debugString , 2048 , "HQRenderDevice::DrawIndexed(%u , %u , %u) return %s\n" ,
			numVertices , indexCount , firstIndex , HQReturnValToString(re));
		
		__OutputDebugString;
	}
	return re;
}
HQReturnVal HQRenderDeviceDebugLayer::DrawIndexedPrimitive(hq_uint32 numVertices , hq_uint32 primitiveCount , hq_uint32 firstIndex )
{
	HQReturnVal re = m_pDevice->DrawIndexedPrimitive(numVertices , primitiveCount , firstIndex );
	if (HQFailed(re))
	{
		sprintf(debugString , 2048 , "HQRenderDevice::DrawIndexedPrimitive(%u , %u , %u) return %s\n" ,
			numVertices , primitiveCount , firstIndex , HQReturnValToString(re));
		
		__OutputDebugString;
	}
	
	return re;
}

HQReturnVal HQRenderDeviceDebugLayer::DrawInstancedIndirect(HQDrawIndirectArgsBuffer* buffer, hquint32 elementIndex)
{
	HQReturnVal re = m_pDevice->DrawInstancedIndirect(buffer, elementIndex);
	if (HQFailed(re))
	{
		sprintf(debugString, 2048, "HQRenderDevice::DrawInstancedIndirect(...) return %s\n",
			HQReturnValToString(re));

		__OutputDebugString;
	}

	return re;
}

HQReturnVal HQRenderDeviceDebugLayer::DrawIndexedInstancedIndirect(HQDrawIndexedIndirectArgsBuffer* buffer, hquint32 elementIndex)
{
	HQReturnVal re = m_pDevice->DrawIndexedInstancedIndirect(buffer, elementIndex);
	if (HQFailed(re))
	{
		sprintf(debugString, 2048, "HQRenderDevice::DrawIndexedInstancedIndirect(...) return %s\n",
			HQReturnValToString(re));

		__OutputDebugString;
	}

	return re;
}


HQReturnVal HQRenderDeviceDebugLayer::DispatchCompute(hquint32 numGroupX, hquint32 numGroupY, hquint32 numGroupZ)
{
	HQReturnVal re = m_pDevice->DispatchCompute(numGroupX, numGroupY, numGroupZ);
	if (HQFailed(re))
	{
		sprintf(debugString, 2048, "HQRenderDevice::DispatchCompute(%u , %u , %u) return %s\n",
			numGroupX, numGroupY, numGroupZ, HQReturnValToString(re));

		__OutputDebugString;
	}

	return re;
}


HQReturnVal HQRenderDeviceDebugLayer::DispatchComputeIndirect(HQComputeIndirectArgsBuffer* buffer, hquint32 elementIndex)
{
	HQReturnVal re = m_pDevice->DispatchComputeIndirect(buffer, elementIndex);
	if (HQFailed(re))
	{
		sprintf(debugString, 2048, "HQRenderDevice::DispatchComputeIndirect(...) return %s\n",
			HQReturnValToString(re));

		__OutputDebugString;
	}

	return re;
}
