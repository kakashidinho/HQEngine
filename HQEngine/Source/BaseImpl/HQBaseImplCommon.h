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

	/*----create hash table with 16 buckets and max load factor 0.75----------*/
	HQClosedPrimeHashTable(const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager()) : parentType(pMemoryManager) {}
	/*----create hash table with max load factor 0.75----------*/
	HQClosedPrimeHashTable(hq_uint32 numBuckets, const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager()) : parentType(numBuckets, pMemoryManager) {}

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
	class Key,
	class T,
	class ProbingFunction = HQQuadradticProbing,
	class MemoryManager = HQDefaultMemManager
>
class HQClosedPtrKeyHashTable : public HQClosedPrimeHashTable<Key, T, HQPtrKeyHashFunc<Key>, ProbingFunction, HQEnginePtrKeyEqual<Key>, MemoryManager>
{
public:
	typedef HQClosedPrimeHashTable<Key, T, HQPtrKeyHashFunc<Key>, ProbingFunction, HQEnginePtrKeyEqual<Key>, MemoryManager > parentType;

	/*----create hash table with 16 buckets and max load factor 0.75----------*/
	HQClosedPtrKeyHashTable(const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager()) : parentType(pMemoryManager) {}
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

#endif
