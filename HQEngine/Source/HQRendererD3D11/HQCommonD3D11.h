/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_COMMON_D3D11_H
#define HQ_COMMON_D3D11_H

#include "d3d11.h"
#include "../BaseImpl/HQBaseImplCommon.h"
#include "HQLoggableObject.h"
#include "HQRendererCoreType.h"

#ifndef HQ_FORCE_INLINE
#	ifdef _MSC_VER
#		if (_MSC_VER >= 1200)
#			define HQ_FORCE_INLINE __forceinline
#		else
#			define HQ_FORCE_INLINE __inline
#		endif
#	else
#		if defined DEBUG|| defined _DEBUG
#			define HQ_FORCE_INLINE inline
#		else
#			define HQ_FORCE_INLINE inline __attribute__ ((always_inline))
#		endif
#	endif
#endif

HQReturnVal CopyD3D11BufferContent(void *dest, ID3D11Buffer * resource);
HQReturnVal CopyD3D11Texture2DContent(void *dest, ID3D11Texture2D * resource, size_t sizeToCopy);

//base buffer object. Note: D3D11 resource's creation is done outside this class
struct HQBufferD3D11 : public virtual HQMappableResource, public virtual HQGraphicsResourceRawRetrievable, public HQBaseIDObject
{
	HQBufferD3D11(bool isDynamic, hq_uint32 size)
	{
		this->pD3DBuffer = NULL;
		this->isDynamic = isDynamic;
		this->size = size;
	}
	virtual ~HQBufferD3D11()
	{
		SafeRelease(this->pD3DBuffer);
	}

	virtual hquint32 GetSize() const { return size; }
	virtual HQReturnVal Unmap();
	virtual HQReturnVal Update(hq_uint32 offset, hq_uint32 size, const void * pData);
	virtual HQReturnVal GenericMap(void ** ppData, HQMapType mapType, hquint32 offset, hquint32 size);
	virtual HQReturnVal CopyContent(void *dest);

	//implement HQGraphicsResourceRawRetrievable
	virtual void * GetRawHandle() { return pD3DBuffer; }

	ID3D11DeviceContext* pD3DContext;
	ID3D11Buffer *pD3DBuffer;
	HQLoggableObject *pLog;
	bool isDynamic;
	hq_uint32 size;
};


/*---------------unordered access view for buffer object--------------*/
//------------HQBufferUAViewKey----------------------
struct HQBufferUAViewKey{
	HQBufferUAViewKey()
	:firstElement(0), numElements(0), hashCode(0)
	{
	}
	HQBufferUAViewKey(const HQBufferUAViewKey& src)
		:firstElement(src.firstElement), numElements(src.numElements), hashCode(src.hashCode)
	{
	}

	HQBufferUAViewKey & operator = (const HQBufferUAViewKey& src)
	{
		this->firstElement = src.firstElement;
		this->numElements = src.numElements;
		this->hashCode = src.hashCode;

		return *this;
	}

	void Init(hquint32 firstElement, hquint32 numElements)
	{
		this->firstElement = firstElement;
		this->numElements = numElements;
		this->hashCode = 29 * firstElement + numElements;
	}
	hquint32 HashCode() const{
		return hashCode;
	}

	bool Equal(const HQBufferUAViewKey* key2) const{
		return firstElement == key2->firstElement && numElements == key2->numElements;
	}

	hquint32 GetFirstElementIdx() const{ return firstElement; }
	hquint32 GetNumElements() const{ return numElements; }

private:
	hquint32 firstElement;
	hquint32 numElements;
	hquint32 hashCode;
};

/*------------------HQBufferUAView----------------------*/
struct HQBufferUAView{
	HQBufferUAView()
	: pD3DUAV(NULL)
	{

	}
	bool Init(ID3D11Device* creator, ID3D11Buffer * resource,
		const D3D11_UNORDERED_ACCESS_VIEW_DESC &desc,
		const HQBufferUAViewKey& srckey
		)
	{
		if (FAILED(creator->CreateUnorderedAccessView(resource, &desc, &this->pD3DUAV)))
			return false;

		this->key = srckey;

		return true;
	}
	~HQBufferUAView()
	{
		SafeRelease(pD3DUAV);
	}

	const HQBufferUAViewKey& GetKey() const { return key; }
	ID3D11UnorderedAccessView* GetD3DView() const { return pD3DUAV; }
private:
	HQBufferUAViewKey key;
	ID3D11UnorderedAccessView *pD3DUAV;
};

//unordered access supported buffer object's view cache
struct HQBufferUAView_CacheD3D11 {
	HQBufferUAView_CacheD3D11()
	: pCachedView(NULL), pUavTable(NULL), pBaseViewDesc(NULL), pBufferDesc(NULL)
	{
	}

	virtual ~HQBufferUAView_CacheD3D11(){
		SafeDelete(pBaseViewDesc);
		SafeDelete(pBufferDesc);
		SafeDelete(pUavTable);
	}

	bool InitUAVCache(ID3D11Device* creator, ID3D11Buffer* resource, hquint32 _totalElements, hquint32 _elementSize)
	{
		this->totalElements = _totalElements;
		this->elementSize = _elementSize;

		//init view desc
		this->cachedKey.Init(0, this->totalElements);

		this->pBaseViewDesc = HQ_NEW D3D11_UNORDERED_ACCESS_VIEW_DESC();
		this->pBufferDesc = HQ_NEW D3D11_BUFFER_DESC();

		resource->GetDesc(pBufferDesc);

		this->pBaseViewDesc->ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
#ifdef HQ_USE_STRUCTURED_BUFFER
		this->pBaseViewDesc->Format = DXGI_FORMAT_UNKNOWN;
		this->pBaseViewDesc->Buffer.FirstElement = 0;
		this->pBaseViewDesc->Buffer.NumElements = this->totalElements;
		this->pBaseViewDesc->Buffer.Flags = 0;
#elif defined HQ_USE_RAW_BUFFER
		this->pBaseViewDesc->Format = DXGI_FORMAT_R32_TYPELESS;
		this->pBaseViewDesc->Buffer.FirstElement = 0;
		this->pBaseViewDesc->Buffer.NumElements = this->totalElements * this->elementSize / 4;
		this->pBaseViewDesc->Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
#else
		if ((pBufferDesc->MiscFlags & D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS) != 0
			|| (pBufferDesc->MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS) != 0)
		{
			if (pBufferDesc->MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS)
			{
				this->pBaseViewDesc->Format = DXGI_FORMAT_R32_TYPELESS;
				this->pBaseViewDesc->Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
			}
			else
			{
				this->pBaseViewDesc->Format = DXGI_FORMAT_R32_UINT;
				this->pBaseViewDesc->Buffer.Flags = 0;
			}
			this->pBaseViewDesc->Buffer.FirstElement = 0;
			this->pBaseViewDesc->Buffer.NumElements = this->totalElements * this->elementSize / 4;
		}
		else//structured buffer
		{
			this->pBaseViewDesc->Format = DXGI_FORMAT_UNKNOWN;
			this->pBaseViewDesc->Buffer.FirstElement = 0;
			this->pBaseViewDesc->Buffer.NumElements = this->totalElements;
			this->pBaseViewDesc->Buffer.Flags = 0;
		}
#endif

		//create full buffer view
		HQBufferUAView* fullView = HQ_NEW HQBufferUAView();
		if (!fullView->Init(creator, resource, *this->pBaseViewDesc, this->cachedKey))
		{
			delete fullView;
			return false;
		}

		//cache view
		this->pCachedView = fullView->GetD3DView();
		//add to table
		this->pUavTable = HQ_NEW UAVTableType();
		this->pUavTable->Add(&fullView->GetKey(), fullView);

		return true;
	}

	//Note: resource must be the same as value passing to InitUAVCache()
	HQ_FORCE_INLINE ID3D11UnorderedAccessView* GetOrCreateUAView(ID3D11Device* creator, ID3D11Buffer* resource, hquint32 firstIndex, hquint32 numElements){
		//try to compare with cached view
		if (this->cachedKey.GetFirstElementIdx() == firstIndex && this->cachedKey.GetNumElements() == numElements)
			return this->pCachedView;
		
		//try to find in table
		if (this->pUavTable == NULL)
			return NULL;
		bool found = false;
		this->cachedKey.Init(firstIndex, numElements);

		HQSharedPtr<HQBufferUAView> view = this->pUavTable->GetItem(&this->cachedKey, found);

		if (!found)
		{
			//create new view
#ifdef HQ_USE_STRUCTURED_BUFFER
			this->pBaseViewDesc->Buffer.FirstElement = firstIndex;
			this->pBaseViewDesc->Buffer.NumElements = numElements;
#elif defined HQ_USE_RAW_BUFFER
			this->pBaseViewDesc->Buffer.FirstElement = firstIndex * this->elementSize / 4;
			this->pBaseViewDesc->Buffer.NumElements = numElements * this->elementSize / 4;
#else
			if ((pBufferDesc->MiscFlags & D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS) != 0
				|| (pBufferDesc->MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS) != 0)
			{
				this->pBaseViewDesc->Buffer.FirstElement = firstIndex * this->elementSize / 4;
				this->pBaseViewDesc->Buffer.NumElements = numElements * this->elementSize / 4;
			}
			else {
				this->pBaseViewDesc->Buffer.FirstElement = firstIndex;
				this->pBaseViewDesc->Buffer.NumElements = numElements;
			}
#endif

			HQBufferUAView* newView = HQ_NEW HQBufferUAView();
			if (!newView->Init(creator, resource, *this->pBaseViewDesc, this->cachedKey))
			{
				delete newView;
				return NULL;
			}

			view = newView;

			//add to table
			this->pUavTable->Add(&view->GetKey(), view);

		}//if (!found)

		//cache the view and return
		this->pCachedView = view->GetD3DView();
		return this->pCachedView;
	}


	hquint32 totalElements;
	hquint32 elementSize;

	D3D11_UNORDERED_ACCESS_VIEW_DESC* pBaseViewDesc;
	D3D11_BUFFER_DESC* pBufferDesc;

	ID3D11UnorderedAccessView* pCachedView;
	HQBufferUAViewKey cachedKey;

	typedef HQClosedPtrKeyHashTable<const HQBufferUAViewKey*, HQSharedPtr<HQBufferUAView> > UAVTableType;
	UAVTableType *pUavTable;
};


#ifdef WIN32
#	pragma warning( push )
#	pragma warning( disable : 4250 )//dominance inheritance of HQBufferD3D11
#endif

/*-------------HQGenericBufferD3D11, can be base of UAV supported or non-supported buffer------------------*/
enum HQGenericBufferD3D11Type{
	HQ_VERTEX_BUFFER_D3D11,
	HQ_INDEX_BUFFER_D3D11,
	HQ_DRAW_INDIRECT_BUFFER_D3D11,
	HQ_SHADER_USE_ONLY_D3D11
};

//Note: D3D11 resource's creation is done outside this class
struct HQGenericBufferD3D11 : public HQBufferUAView_CacheD3D11, public HQBufferD3D11, public HQGraphicsBufferRawRetrievable {
	typedef HQLinkedList<hquint32, HQPoolMemoryManager> SlotList;

	struct BufferSlot//represent a (input/output) slot that buffer can bind to 
	{
		SlotList::LinkedListNodeType *bufferLink;//for fast removal
		HQSharedPtr<HQGenericBufferD3D11> pBuffer;//buffer reference

		void BindAsInput(hquint32 slotIndex, HQSharedPtr<HQGenericBufferD3D11>& pBuffer);
		void UnbindAsInput();

		void BindAsUAV(hquint32 slotIndex, HQSharedPtr<HQGenericBufferD3D11>& pBuffer);
		void UnbindAsUAV();
	};


	//constructor
	HQGenericBufferD3D11(HQGenericBufferD3D11Type _type, 
						bool dynamic, 
						hquint32 size, 
						const HQSharedPtr<HQPoolMemoryManager>& inputBoundListsMemManager)
	: HQBufferD3D11(dynamic, size),
		type(_type),
		inputBoundSlots(inputBoundListsMemManager),
		uavBoundSlots(s_uavBoundSlotsMemManager)
	{
		HQ_ASSERT(inputBoundListsMemManager != NULL);
		HQ_ASSERT(s_uavBoundSlotsMemManager != NULL);
	}

	bool InitUAVCache(ID3D11Device* creator, hquint32 _totalElements, hquint32 _elementSize)
	{
		return this->HQBufferUAView_CacheD3D11::InitUAVCache(creator, this->pD3DBuffer, _totalElements, _elementSize);
	}

	HQ_FORCE_INLINE ID3D11UnorderedAccessView* GetOrCreateUAView(ID3D11Device* creator, hquint32 firstIndex, hquint32 numElements)
	{
		return this->HQBufferUAView_CacheD3D11::GetOrCreateUAView(creator, this->pD3DBuffer, firstIndex, numElements);
	}

	HQGenericBufferD3D11Type type;

	SlotList inputBoundSlots; //list of input slots that this buffer is bound to
	SlotList uavBoundSlots; //list of UAV slots that this buffer is bound to

	static HQSharedPtr<HQPoolMemoryManager> s_uavBoundSlotsMemManager;//must be created before any buffer's creation
};

#ifdef WIN32
#	pragma warning( pop )
#endif

/*------------HQGenericBufferD3D11::BufferSlot---------------------*/
//{slotIndex} is index of buffer slot, of course buffer doesn't know about its index
HQ_FORCE_INLINE void HQGenericBufferD3D11::BufferSlot::BindAsInput(hquint32 slotIndex, HQSharedPtr<HQGenericBufferD3D11>& pNewBuffer)
{
	if (this->pBuffer != pNewBuffer){
		this->UnbindAsInput();//unlink old buffer

		if (pNewBuffer != NULL)
			this->bufferLink = pNewBuffer->inputBoundSlots.PushBack(slotIndex);
			
		this->pBuffer = pNewBuffer;
	}
}

HQ_FORCE_INLINE void HQGenericBufferD3D11::BufferSlot::UnbindAsInput()
{
	if (this->pBuffer != NULL)
	{
		this->pBuffer->inputBoundSlots.RemoveAt(this->bufferLink);
		this->pBuffer.ToNull();
	}
}

//{slotIndex} is index of buffer slot, of course buffer doesn't know about its index
HQ_FORCE_INLINE void HQGenericBufferD3D11::BufferSlot::BindAsUAV(hquint32 slotIndex, HQSharedPtr<HQGenericBufferD3D11>& pNewBuffer)
{

	if (this->pBuffer != pNewBuffer){
		this->UnbindAsUAV();//unlink old buffer

		if (pNewBuffer != NULL)
			this->bufferLink = pNewBuffer->uavBoundSlots.PushBack(slotIndex);

		this->pBuffer = pNewBuffer;
	}
}

HQ_FORCE_INLINE void HQGenericBufferD3D11::BufferSlot::UnbindAsUAV()
{
	if (this->pBuffer != NULL)
	{
		this->pBuffer->uavBoundSlots.RemoveAt(this->bufferLink);
		this->pBuffer.ToNull();
	}
}

#endif