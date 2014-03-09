/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_ENGINE_COMMON_INTERNAL_H
#define HQ_ENGINE_COMMON_INTERNAL_H

#include "../HQEngineApp.h"
#include "../HQClosedHashTable.h"
#include "../HQPrimeNumber.h"

#include <string.h>
#include <stdio.h>


//named object
class HQEngineNamedObjImpl: public virtual HQEngineNamedObj{
public:
	HQEngineNamedObjImpl()
		:m_name(NULL)
	{
	}
	HQEngineNamedObjImpl(const char* _name) {
		SetName(_name);
	}
	~HQEngineNamedObjImpl(){
		if (m_name) delete[] m_name;
	}

	void SetName(const char* _name){
		if (m_name) delete[] m_name;

		size_t len = strlen(_name);
		m_name = new char[len + 1];
		strncpy(m_name, _name, len);
		m_name[len] = '\0';
	}
	virtual const char* GetName() const {return m_name;}
protected:
	char* m_name;
};

/*------------graphics related object, needs reference to render device--------------------------*/
class HQGraphicsRelatedObj {
public:
	HQGraphicsRelatedObj() {
		m_renderDevice = HQEngineApp::GetInstance()->GetRenderDevice();
	}
protected:
	HQEngineAppRenderDevice * m_renderDevice;
};

/*---------named graphics related object------*/
class HQNamedGraphicsRelatedObj: public HQGraphicsRelatedObj, public virtual HQEngineNamedObjImpl{
public:
	HQNamedGraphicsRelatedObj(): HQEngineNamedObjImpl() {}
	HQNamedGraphicsRelatedObj(const char* _name) : HQEngineNamedObjImpl(_name) {}
};

/*----------base hash table using pointer typed key-----------------*/
template <class Key>
//hash function for a pointer typed key
struct HQEnginePtrKeyHashFunc {
	hquint32 operator() (const Key& key) const {
		return key->HashCode();
	}
};

template <class Key>
struct HQEnginePtrKeyEqual {
	bool operator () (const Key & val1 , const Key &val2)const  
	{
		return val1->Equal(val2);
	}
};

//hash table for a pointer typed key
template
<
class Key ,
class T ,
class ProbingFunction = HQQuadradticProbing ,
class MemoryManager = HQDefaultMemManager
>
class HQEnginePtrKeyHashTable: public HQClosedHashTable<Key, T, HQEnginePtrKeyHashFunc<Key> , ProbingFunction, HQEnginePtrKeyEqual<Key>,  MemoryManager>
{
public:
	typedef HQClosedHashTable<Key, T , HQEnginePtrKeyHashFunc<Key> , ProbingFunction, HQEnginePtrKeyEqual<Key>,  MemoryManager > parentType;

	/*----create hash table with 16 buckets and max load factor 0.75----------*/
	HQEnginePtrKeyHashTable(const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager()) : parentType(pMemoryManager) {}
protected:
	hq_uint32 GetNewSize()
	{
		//resize table size to next prime number
		hq_uint32 i = this->m_numBuckets + 1;
		while ( !HQIsPrime(i) || (hq_float32)this->m_numItems / i > 0.5f )
		{
			++i;
		}
		return i;
	}
};

/*------- hash table using string key--------*/
template <class T>
class HQEngineStringHashTable : public HQClosedStringHashTable<T>
{
public:
	typedef HQClosedStringHashTable<T> ParentType;
	HQEngineStringHashTable() : ParentType(3 , 0.5f) {}

protected:
	hq_uint32 GetNewSize()
	{
		//resize table size to next prime number
		hq_uint32 i = this->m_numBuckets + 1;
		while ( !HQIsPrime(i) || (hq_float32)this->m_numItems / i > 0.5f )
		{
			++i;
		}
		return i;
	}
};

namespace HQEngineHelper
{
//C functions for data stream
void seek_datastream (void* fileHandle, long offset, int origin);
size_t tell_datastream (void* fileHandle);
size_t read_datastream ( void * ptr, size_t size, size_t count, void * stream );

//for easy porting from standard C io
HQENGINE_API long ftell ( HQDataReaderStream * stream );
HQENGINE_API int fseek ( HQDataReaderStream * stream, long int offset, int origin );
HQENGINE_API void rewind( HQDataReaderStream* stream );
HQENGINE_API int fgetc(HQDataReaderStream *stream);
HQENGINE_API size_t fread ( void * ptr, size_t size, size_t count, HQDataReaderStream * stream );
HQENGINE_API int fclose ( HQDataReaderStream * stream );

};//namespace HQEngineHelper


#endif