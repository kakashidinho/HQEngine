#ifndef HQ_COMMON_VERTEX_STREAM_GL_INL_H
#define HQ_COMMON_VERTEX_STREAM_GL_INL_H

#include "HQCommonVertexStreamManGL.h"

/*---------vertex stream---------------*/
inline HQVertexStreamGL::HQVertexStreamGL()
{
	this->attribInfoStack = NULL;
	this->stride = 0;
}

inline void HQVertexStreamGL::InsertToStack(HQVertexAttribInfoNodeGL *node)
{
	node->pNext = this->attribInfoStack;
	this->attribInfoStack = node;
}

/*---------template delegate implementation-----------*/

template <class VertexStreamManager>
HQ_FORCE_INLINE HQReturnVal HQVertexStreamManDelegateGL<VertexStreamManager>::SetVertexBuffer(
								   VertexStreamManager *manager,
									HQSharedPtr<HQBufferGL>& vBuffer , 
									HQVertexStreamGL & stream , 
									hq_uint32 stride)
{
	bool resetInputLayout = false;
	bool enableVertAttrib = false;

	if (stream.vertexBuffer != vBuffer)
	{
		if (vBuffer == NULL)//bind null buffer to this stream
		{
			HQVertexAttribInfoNodeGL *pNode = stream.attribInfoStack;
			//disable vertex attributes come from this stream
			while (pNode != NULL)
			{
				VertexStreamManager::DisableVertexAttribArray(pNode->pAttribInfo->attribIndex);
				pNode = pNode->pNext;
			}
		}
		else
		{
			if (stream.vertexBuffer == NULL)//need to enable vertex attributes come from this stream
				enableVertAttrib = true;
			resetInputLayout = true;
		}

		stream.vertexBuffer = vBuffer;
	}
	if (vBuffer != NULL && stream.stride != stride)
	{
		stream.stride = stride;
		resetInputLayout = true;
	}
	
	if (resetInputLayout && stream.attribInfoStack != NULL)
	{
		//vertex buffer bound to this stream or stream 's stride has changed , we need to reset vertex attribute pointer
		manager->BindVertexBuffer(vBuffer->bufferName);
		HQBufferGL *bufferRawPtr = vBuffer.GetRawPointer();
		HQVertexAttribInfoNodeGL *pNode = stream.attribInfoStack;
		if (enableVertAttrib)
		{
			do
			{
				VertexStreamManager::EnableVertexAttribArray(pNode->pAttribInfo->attribIndex);

				VertexStreamManager::SetVertexAttribPointer(*pNode->pAttribInfo , stride , bufferRawPtr);
				pNode = pNode->pNext;
			}while (pNode != NULL);
		}
		else
		{
			do
			{
				VertexStreamManager::SetVertexAttribPointer(*pNode->pAttribInfo , stride , bufferRawPtr);
				pNode = pNode->pNext;
			}while (pNode != NULL);
		}
		
	}
	return HQ_OK;
}

template <class VertexStreamManager>
HQ_FORCE_INLINE HQReturnVal HQVertexStreamManDelegateGL<VertexStreamManager>::SetVertexInputLayout(
	VertexStreamManager *manager,//vertex stream manager
	HQSharedPtr<HQVertexInputLayoutGL>& pVLayout,
	HQSharedPtr<HQVertexInputLayoutGL>& activeInputLayout,//current active layout
	HQVertexStreamGL * streams,//streams
	hquint32 maxVertexAttribs,//max number of vertex attributes
	HQVertexAttribInfoNodeGL *vAttribInfoNodeCache//vertex attributes info cache
	) 
{
	HQVertexInputLayoutGL *pVLayoutRaw = pVLayout.GetRawPointer();

	if (activeInputLayout != pVLayoutRaw)
	{
		if (pVLayoutRaw != NULL)
		{
			hq_uint32 flag;
			hq_uint32 diffFlags = 0xffffffff;
			if (activeInputLayout != NULL)
				diffFlags = pVLayoutRaw->flags ^ activeInputLayout->flags;

			for (hq_uint32 i = 0 ; i < maxVertexAttribs ; ++i)
			{
				streams[i].attribInfoStack = NULL;//reset attibute info stack of this stream
				//find vertex attributes that need to be disabled
				flag = 0x1 << i;
				if ((diffFlags & flag) != 0 && (pVLayoutRaw->flags & flag) == 0)//this vertex attribute slot need to be disabled
				{
					VertexStreamManager::DisableVertexAttribArray(i);
				}

				if (i < pVLayoutRaw->numAttribs)
				{
					HQVertexAttribInfoGL &vAttribInfo = pVLayoutRaw->attribs[i];
					HQVertexStreamGL &stream = streams[vAttribInfo.streamIndex];
					HQVertexAttribInfoNodeGL &attribInfoCache = vAttribInfoNodeCache[vAttribInfo.attribIndex];

					if (stream.vertexBuffer != NULL)
					{
						flag = 0x1 << vAttribInfo.attribIndex;
						if ((diffFlags & flag) != 0)//this vertex attribute slot need to be enabled
						{
							VertexStreamManager::EnableVertexAttribArray(vAttribInfo.attribIndex);
						}
						if (attribInfoCache.pAttribInfo == NULL || 
							attribInfoCache.pAttribInfo->streamIndex != vAttribInfo.streamIndex ||
							attribInfoCache.pAttribInfo->size != vAttribInfo.size ||
							attribInfoCache.pAttribInfo->offset != vAttribInfo.offset ||
							attribInfoCache.pAttribInfo->dataType != vAttribInfo.dataType ||
							attribInfoCache.pAttribInfo->normalized != vAttribInfo.normalized)
						{
							HQBufferGL *rawBufferPtr = stream.vertexBuffer.GetRawPointer();
							manager->BindVertexBuffer(rawBufferPtr->bufferName);
							VertexStreamManager::SetVertexAttribPointer(vAttribInfo , stream.stride , rawBufferPtr);
						}
					}

					//insert this vertex attribute info to stream's stack
					attribInfoCache.pAttribInfo = &vAttribInfo;
					stream.InsertToStack(&attribInfoCache);
				}
			}//for (hq_uint32 i = 0 ; i < MaxVertexAttrib ; ++i)
		}//if (pVLayoutRaw != NULL)
		else
		{//disable vertex attributes
			for (hq_uint32 i = 0 ; i < activeInputLayout->numAttribs ; ++i)
			{
				HQVertexAttribInfoGL &vAttribInfo = activeInputLayout->attribs[i];
				
				if (streams[vAttribInfo.streamIndex].vertexBuffer != NULL)
					VertexStreamManager::DisableVertexAttribArray(vAttribInfo.attribIndex);
				streams[vAttribInfo.streamIndex].attribInfoStack = NULL;//clear attributes info stack 
			}
		}

		activeInputLayout = pVLayout;
	}

	return HQ_OK;

}

#endif