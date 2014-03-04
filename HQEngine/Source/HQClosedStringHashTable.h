/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_CLOSED_STRING_HASH_TABLE_H
#define HQ_CLOSED_STRING_HASH_TABLE_H

#include "HQClosedHashTable.h"


/*------------------hash table with string key,this hash table resolves collision by using open addressing-----------*/

template <class T ,class ProbingFunc = HQQuadradticProbing , class MemoryManager = HQDefaultMemManager>
class HQClosedStringHashTable : public HQClosedHashTable<std::string, T , HQDefaultStringHashFunc<char , std::string> , ProbingFunc , HQDefaultKeyEqual<std::string> , MemoryManager>
{
public:
    typedef HQClosedHashTable<std::string, T , HQDefaultStringHashFunc<char , std::string> , ProbingFunc , HQDefaultKeyEqual<std::string> , MemoryManager> parentType;

	HQClosedStringHashTable(const parentType &src , const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager()) : parentType(src , pMemoryManager) {}

	/*----create hash table with 16 buckets and max load factor 0.75----------*/
	HQClosedStringHashTable(const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager()) : parentType(pMemoryManager) {}

	/*----create hash table with max load factor 0.75----------*/
	HQClosedStringHashTable(hq_uint32 numBuckets , const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager()) : parentType(numBuckets , pMemoryManager) {}

	/*-----------------------*/
	HQClosedStringHashTable(hq_uint32 numBuckets , hq_float32 maxLoadFactor , const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager()) : parentType(numBuckets , maxLoadFactor , pMemoryManager) {}

	/*---------------------*/
	using parentType::Find;
	using parentType::GetItemPointer;
	using parentType::GetItem;

	/*--------additional methods-------------*/
	bool Find(const char* key , hq_uint32 &foundIndex) const
	{
		if (this->m_numItems == 0)
			return false;
		hq_uint32 hashCode = this->m_hashFunction(key);
		hq_uint32 i = 0 ;
		bool found = false;
		while(!found && i < this->m_numBuckets)
		{
			foundIndex = this->m_probeFunction(hashCode , i) % this->m_numBuckets;

			if (this->m_buckets[foundIndex].m_isEmpty == 0 &&
				!this->m_buckets[foundIndex].m_entry.m_first.compare(key))
			{
				found = true;
			}
			else
				++i;
		}
		return found;
	}
	T * GetItemPointer(const char* key)
	{
		hq_uint32 foundIndex;
		if (this->Find(key , foundIndex))
		{
			return &this->GetItemNonCheck(foundIndex);
		}

		return NULL;
	}
	const T * GetItemPointer(const char* key)const
	{
		hq_uint32 foundIndex;
		if (this->Find(key , foundIndex))
		{
			return &this->GetItemNonCheck(foundIndex);
		}

		return NULL;
	}
	T& GetItem(const char* key , bool &isFound)
	{
		hq_uint32 foundIndex;
		if (isFound = this->Find(key , foundIndex))
		{
			return this->GetItemNonCheck(foundIndex);
		}
		else
			return this->m_invalidValue;
	}

	const T& GetItem(const char* key , bool &isFound) const
	{
		hq_uint32 foundIndex;
		if (isFound = this->Find(key , foundIndex))
		{
			return this->GetItemNonCheck(foundIndex);
		}
		else
			return this->m_invalidValue;
	}
};


#endif
