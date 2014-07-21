/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _HQ_COMMON_H_
#define _HQ_COMMON_H_

#include "../HQPlatformDef.h"
#include "../HQPrimeNumber.h"
#include "../HQClosedStringHashTable.h"
#include "../HQItemManager.h"

/*---------------flags--------------------*/
#ifdef HQ_ANDROID_PLATFORM
#	define DEVICE_LOST_POSSIBLE
#endif

#define RUNNING 0x1
#define RENDER_BEGUN 0x2
#define WINDOWED 0x4
#define DEVICE_LOST 0x8
#define VSYNC_ENABLE 0x10
#define USEVSHADER 0x20
#define USEGSHADER 0x40
#define USEPSHADER 0x80
#define USEFSHADER 0x80
#define USESHADER (USEVSHADER | USEGSHADER | USEPSHADER)

#define CLEAR_COLOR_OR_DEPTH_CHANGED 0x100
#define INACTIVE 0x200



/*-------------------------------------------------*/

#if !defined WIN32 && !defined HQ_WIN_STORE_PLATFORM &&  !defined HQ_WIN_PHONE_PLATFORM
#	ifndef DWORD
#		define DWORD hq_uint32
#	endif
#	ifndef WORD
#		define WORD hq_ushort16
#	endif
#	ifndef UINT
#		define UINT hq_uint32
#	endif
#endif

#if _MSC_VER >= 1800
#	define __FINAL__ final
#else
#	define __FINAL__
#endif


#ifndef SafeDelete
#	define SafeDelete(p){if(p != NULL){delete p;p=NULL;}}
#endif
#ifndef SafeDeleteArray
#	define SafeDeleteArray(p){if(p != NULL){delete[] p;p=NULL;}}
#endif

#ifndef SafeDeleteTypeCast
#	define SafeDeleteTypeCast(casting_type, ptr) {if(ptr != NULL) {delete static_cast<casting_type> (ptr); ptr=NULL; } }
#endif

#ifndef SafeRelease
#define SafeRelease(p) {if(p){p->Release();p=0;}}
#endif
#ifndef SafeDelete
#define SafeDelete(p){if(p){delete p;p=0;}}
#endif
#ifndef SafeDeleteArray
#define SafeDeleteArray(p){if(p){delete[] p;p=0;}}
#endif

#define COLOR_ARGB(a,r,g,b) (((a&0xff)<<24) | ((r&0xff)<<16) | ((g&0xff)<<8) | ((b&0xff)) )
#define COLOR_XRGB(r,g,b) COLOR_ARGB(1,r,g,b)
#define COLOR_COLORVALUE(a,r,g,b) ((((DWORD)(a*255)&0xff)<<24) |\
								  (((DWORD)(r*255)&0xff)<<16) |\
								  (((DWORD)(g*255)&0xff)<<8) |\
								  (((DWORD)(b*255)&0xff)) )

#define GetRfromARGB(c) ((c & 0x00ff0000)>>16)
#define GetGfromARGB(c) ((c & 0x0000ff00)>>8)
#define GetBfromARGB(c) ((c & 0x000000ff))
#define GetAfromARGB(c) ((c & 0xff000000)>>24)


/*-------------------------------*/
//interface for handling resetable event
class HQResetable {
public:
	virtual void OnLost() = 0;
	virtual void OnReset() = 0;
};

/*----------base hash table using prime number of buckets-----------*/
template
<
	class Key,
	class T,
	class HashFunction = HQDefaultHashFunc<Key>,
	class ProbingFunction = HQQuadradticProbing,
	class KeyEqual = HQDefaultKeyEqual<Key>,
	class MemoryManager = HQDefaultMemManager
>
class HQClosedPrimeHashTable : public HQClosedHashTable<Key, T, HashFunction, ProbingFunction, KeyEqual, MemoryManager>
{
public:
	typedef HQClosedHashTable<Key, T, HashFunction, ProbingFunction, KeyEqual, MemoryManager > parentType;

	/*----create hash table with 17 buckets and max load factor 0.4----------*/
	HQClosedPrimeHashTable(const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager()) : parentType(17, 0.4f, pMemoryManager) {}
	/*----create hash table with max load factor 0.4----------*/
	HQClosedPrimeHashTable(hq_uint32 numBuckets, const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager()) : parentType(numBuckets, 0.4f, pMemoryManager) {}

	/*-----------------------*/
	HQClosedPrimeHashTable(hq_uint32 numBuckets, hq_float32 maxLoadFactor, const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager()) : parentType(numBuckets, maxLoadFactor, pMemoryManager) {}
protected:
	hq_uint32 GetNewSize()
	{
		//resize table size to next prime number
		hq_uint32 i = this->m_numBuckets + 1;
		while (!HQIsPrime(i) || (hq_float32)this->m_numItems / i > 0.5f)
		{
			++i;
		}
		return i;
	}
};

/*----------base hash table using pointer typed key-----------------*/


template <class Key>
//hash function for a pointer typed key
struct HQPtrKeyHashFunc {
	hquint32 operator() (const Key& key) const {
		return key->HashCode();
	}
};

template <class Key>
struct HQEnginePtrKeyEqual {
	bool operator () (const Key & val1, const Key &val2)const
	{
		return val1->Equal(val2);
	}
};

//hash table for a pointer typed key
template
<
	class PointerTypedKey,
	class T,
	class ProbingFunction = HQQuadradticProbing,
	class MemoryManager = HQDefaultMemManager
>
class HQClosedPtrKeyHashTable : public HQClosedPrimeHashTable<PointerTypedKey, T, HQPtrKeyHashFunc<PointerTypedKey>, ProbingFunction, HQEnginePtrKeyEqual<PointerTypedKey>, MemoryManager>
{
public:
	typedef HQClosedPrimeHashTable<PointerTypedKey, T, HQPtrKeyHashFunc<PointerTypedKey>, ProbingFunction, HQEnginePtrKeyEqual<PointerTypedKey>, MemoryManager > parentType;

	/*----create hash table with 17 buckets and max load factor 0.4----------*/
	HQClosedPtrKeyHashTable(const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager()) : parentType(pMemoryManager) {}

	HQClosedPtrKeyHashTable(hq_uint32 numBuckets, hq_float32 maxLoadFactor, const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager()) : parentType(numBuckets, maxLoadFactor, pMemoryManager) {}
};

/*------- hash table using string key--------*/
template <class T>
class HQClosedStringPrimeHashTable : public HQClosedStringHashTable<T>
{
public:
	typedef HQClosedStringHashTable<T> ParentType;
	HQClosedStringPrimeHashTable() : ParentType(3, 0.5f) {}

protected:
	hq_uint32 GetNewSize()
	{
		//resize table size to next prime number
		hq_uint32 i = this->m_numBuckets + 1;
		while (!HQIsPrime(i) || (hq_float32)this->m_numItems / i > 0.5f)
		{
			++i;
		}
		return i;
	}
};


class HQBaseIDObject {
public:
	HQBaseIDObject() { ID = HQ_NULL_ID; }
	virtual ~HQBaseIDObject(){}

	hquint32 GetID() const { return ID; }
	void SetID(hquint32 id) { this->ID = id; }
private:
	hquint32 ID;
};

/*------------HQIDItemManager----------------*/
template <class T> //T must be subclass of HQBaseIDObject
class HQIDItemManager : public HQItemManager<T> {
public:
	typedef HQItemManager<T> ParentType;
	template <class compatibleT>
	bool AddItem(T* pItem, compatibleT **ppItem)
	{
		hquint32 itemID;
		bool re = ParentType::AddItem(pItem, &itemID);
		if (re)
		{
			if (ppItem != NULL)
				*ppItem = pItem;
			if (pItem != NULL)
				pItem->SetID(itemID);
		}
		return re;
	}

	bool AddItem(const HQSharedPtr<T> & pItem){
		hquint32 itemID;
		bool re = ParentType::AddItem(pItem, &itemID);
		if (re)
		{
			//work around const 
			HQSharedPtr<T> const_casted_ptr = pItem;
			if (const_casted_ptr != NULL)
				const_casted_ptr->SetID(itemID);
		}
		return re;
	}

	template <class compatibleT>
	bool AddItem(const HQSharedPtr<T> & pItem, compatibleT **ppItem){
		hquint32 itemID;
		bool re = ParentType::AddItem(pItem, &itemID);
		if (re)
		{
			if (ppItem != NULL)
				*ppItem = pItem.GetRawPointer();
			//work around const 
			HQSharedPtr<T> const_casted_ptr = pItem;
			if (const_casted_ptr != NULL)
				const_casted_ptr->SetID(itemID);
		}
		return re;
	}

	template <class compatibleT>
	hqint32 Remove(compatibleT* pItem)//remove item 
	{
		T * pIDItem = static_cast<T*> (pItem);
		if (pIDItem == NULL)
			return ParentType::Remove(HQ_NULL_ID);
		return ParentType::Remove(pIDItem->GetID());
	}

	template <class compatibleT>
	HQSharedPtr<T> GetItemPointer(compatibleT* cptr)//get shared pointer to item
	{
		T *ptr = static_cast<T*>(cptr);
		if (ptr == NULL)
			return HQSharedPtr<T>::null;
		return ParentType::GetItemPointer(ptr->GetID());
	}

	template <class compatibleT>
	T* GetItemRawPointer(compatibleT* cptr)
	{
		T *ptr = static_cast<T*>(cptr);
		if (ptr == NULL)
			return NULL;
		return ParentType::GetItemRawPointer(ptr->GetID());
	}

	template <class compatibleT>
	HQSharedPtr<T> GetItemPointerNonCheck(compatibleT* cptr)//get shared pointer to item
	{
		T *ptr = static_cast<T*>(cptr);
		return ParentType::GetItemPointerNonCheck(ptr->GetID());
	}

	template <class compatibleT>
	T* GetItemRawPointerNonCheck(compatibleT* cptr)//get shared pointer to item
	{
		T *ptr = static_cast<T*>(cptr);
		return ParentType::GetItemRawPointerNonCheck(ptr->GetID());
	}
};


/*----------------sysmtem memory buffer-----------------------*/
class HQSysMemBuffer;

class HQSysMemBufferListener {
public:
	virtual bool BufferMapping(HQSysMemBuffer* buffer, HQMapType mapType) { return true; }//return false if dont want buffer to be mapped
	virtual bool BufferUpdating(HQSysMemBuffer* buffer) { return true; }//return false if dont want buffer to be updated
	virtual void BufferChangeEnded(HQSysMemBuffer* buffer) {}//subclass should implement this
};

class HQSysMemBuffer : public virtual HQMappableResource, public  HQSysMemBufferListener{
public:
	typedef HQSysMemBufferListener Listener;

	HQSysMemBuffer() { this->listener = this; pRawBuffer = NULL; size = 0; }
	HQSysMemBuffer(Listener *_listener):listener(_listener) { pRawBuffer = NULL; size = 0; }
	virtual ~HQSysMemBuffer() { DeallocRawBuffer(); }

	virtual hquint32 GetSize() const { return size; }///mappable size
	virtual HQReturnVal Update(hq_uint32 offset, hq_uint32 size, const void * pData) {
		if (listener->BufferUpdating(this) == false)
			return HQ_FAILED;
		if (offset > this->size)
			return HQ_FAILED;
		if (size == 0)
			size = this->size - offset;
#if defined DEBUG || defined _DEBUG
		if (pData == NULL)
			return HQ_FAILED;
#endif

		memcpy((hqubyte8*)pRawBuffer + offset, pData, size);

		listener->BufferChangeEnded(this);

		return HQ_OK;
	}
	virtual HQReturnVal Unmap() {
		listener->BufferChangeEnded(this);
		return HQ_OK;
	}

	virtual HQReturnVal GenericMap(void ** ppData, HQMapType mapType = HQ_MAP_DISCARD, hquint32 offset = 0, hquint32 size = 0){
		if (listener->BufferMapping(this, mapType) == false)
			return HQ_FAILED;
		if (ppData == NULL)
			return HQ_FAILED;
		*ppData = (hqubyte8*)pRawBuffer + offset;

		return HQ_OK;
	}

	virtual HQReturnVal CopyContent(void * dest){
		memcpy(dest, pRawBuffer, this->size);
		return HQ_OK;
	}

	const void * GetRawBuffer() const { return pRawBuffer; }


protected:
	virtual void DeallocRawBuffer()
	{
		if (pRawBuffer != NULL) {
			delete[]((hqubyte8*)pRawBuffer);
			pRawBuffer = NULL;
		}

		size = 0;
	}
	virtual void AllocRawBuffer(hquint32 size) {
		DeallocRawBuffer();
		this->size = size;
		this->pRawBuffer = HQ_NEW hqubyte8[size];
	}
	
	Listener *listener;

	hquint32 size;
	void * pRawBuffer;

};
#endif
