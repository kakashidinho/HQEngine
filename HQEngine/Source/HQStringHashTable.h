/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_STRING_HASH_TABLE_H
#define HQ_STRING_HASH_TABLE_H

#include "HQHashTable.h"


/*------------------hash table with string key . This class resolves collision by using seperate chainning-----------*/

template <class T ,class MemoryManager = HQDefaultMemManager>
class HQStringHashTable : public HQHashTable<std::string, T , HQDefaultStringHashFunc<char , std::string> , HQDefaultKeyEqual<std::string> , MemoryManager>
{
public:
    typedef HQHashTable<std::string, T , HQDefaultStringHashFunc<char , std::string> , HQDefaultKeyEqual<std::string> , MemoryManager> parentType;

	HQStringHashTable(const parentType &src , const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager()) : parentType(src , pMemoryManager) {}

	/*----create hash table with 16 buckets and max load factor 0.75----------*/
	HQStringHashTable() : parentType() {}

	/*----create hash table with max load factor 0.75----------*/
	HQStringHashTable(hq_uint32 numBuckets , const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager()) : parentType(numBuckets , pMemoryManager) {}
	/*---------------------*/
	using parentType::Find;
	using parentType::GetItemPointer;
	using parentType::GetItem;

	/*--------additional methods-------------*/
	HQLinkedListNode<HQPair<std::string , T> >* Find(const char* key) const
	{
		hq_uint32 slot = this->m_hashFunction(key) % this->m_numBuckets;
		HQLinkedListNode<HQPair<std::string , T> >* foundNode = NULL;
		typename HQLinkedList<HQPair<std::string , T> >::Iterator ite;

		this->m_buckets[slot].GetIterator(ite);

		while(foundNode == NULL && !ite.IsAtEnd())
		{
			if (!ite->m_first.compare(key))
			{
				foundNode = ite.GetNode();
			}
			else
				++ite;
		}
		return foundNode;
	}
	T * GetItemPointer(const char* key)
	{
		HQLinkedListNode<HQPair<std::string , T> >* foundNode = this->Find(key);
		if (foundNode != NULL)
		{
			return &foundNode->m_element.m_second;
		}

		return NULL;
	}
	T * GetItemPointer(const char* key)const
	{
		HQLinkedListNode<HQPair<std::string , T> >* foundNode = this->Find(key);
		if (foundNode != NULL)
		{
			return &foundNode->m_element.m_second;
		}

		return NULL;
	}

	T& GetItem(const char* key , bool &isFound)
	{
		HQLinkedListNode<HQPair<std::string , T> >* foundNode = this->Find(key);
		if (foundNode != NULL)
		{
			isFound = true;
			return foundNode->m_element.m_second;
		}
		else
		{
			isFound = false;
			return this->m_invalidValue;
		}
	}
	const T& GetItem(const char* key , bool &isFound) const
	{
		HQLinkedListNode<HQPair<std::string , T> >* foundNode = this->Find(key);
		if (foundNode != NULL)
		{
			isFound = true;
			return foundNode.m_element.m_second;
		}
		else
		{
			isFound = false;
			return this->m_invalidValue;
		}
	}
};


#endif
