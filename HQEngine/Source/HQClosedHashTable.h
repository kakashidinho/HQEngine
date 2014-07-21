/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_CLOSED_HASH_TABLE_H
#define HQ_CLOSED_HASH_TABLE_H

#include <iostream>
#include <string.h>//memset
#include "HQMemoryManager.h"
#include "HQDefaultHashFunction.h"
#include "HQDefaultKeyEqual.h"
#include "HQPair.h"
#include "HQSharedPointer.h"

struct HQLinearProbing
{
	inline hq_uint32 operator() (unsigned hashCode , hq_uint32 probeStep) const
	{
		if (probeStep == 0)
			return hashCode;
		return (hashCode + probeStep);//h(k , i) = h(k) + i
	}
};

struct HQQuadradticProbing
{
	inline hq_uint32 operator() (unsigned hashCode , hq_uint32 probeStep) const
	{
		if (probeStep == 0)
			return hashCode;
		return (hashCode + probeStep * probeStep);//h(k , i) = h(k) + i^2
	}
};

typedef hq_ubyte8 HQBucketBoolType;

template <class Key , class T >
struct HQHashTableBucket
{
	typedef HQPair<Key , T> EntryType;
	HQHashTableBucket() : m_isEmpty(1) {}
	HQHashTableBucket(HQBoolType isEmpty) : m_isEmpty(isEmpty) {}
	HQHashTableBucket(Key key , T value) : m_isEmpty(0) , m_entry(key , value) {}

	~HQHashTableBucket(){}

	EntryType m_entry;
	HQBucketBoolType m_isEmpty;
};

/*-----------this hash table resolves collision by using open addressing. Note: destructor will not be called when an item is removed from table-----*/
//class MemoryManager will be used for malloc and free multiple HashTableBucketType objects
template
<
class Key ,
class T ,
class HashFunction = HQDefaultHashFunc<Key> ,
class ProbingFunction = HQQuadradticProbing ,
class KeyEqual = HQDefaultKeyEqual<Key> ,
class MemoryManager = HQDefaultMemManager
>

class HQClosedHashTable
{
public:
	typedef HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> HashTableType;
	typedef HQHashTableBucket<Key , T > HashTableBucketType;
	/*------iterator class - for traversing through list of items in this manager---*/
	class Iterator
	{
		friend class HQClosedHashTable;
	private:
		HashTableBucketType* m_buckets;
		hq_uint32 currentPos;
		hq_uint32 maxPos;

		static Key m_invalidKey;
	public:
		Iterator& operator ++();//prefix addition  . shift to next item
		Iterator operator ++(hq_int32);//suffic addition . shift to next item

		Iterator& operator --();//prefix subtraction . shift to prev item
		Iterator operator --(hq_int32);//suffix subtraction . shift to prev item

		bool IsAtBegin(){return (currentPos == 0);};//is at first bucket
		bool IsAtEnd(){return (currentPos == maxPos);};//is at invalid location (ie location after the last bucket)

		void Rewind(){currentPos = 0;};//go to first bucket

		T* operator->();
		T& operator*();
		T* GetItemPointer();
		T* GetItemPointerNonCheck();
		const Key& GetKey();
	};

	/*----------copy constructor---------------*/
	HQClosedHashTable(const HQClosedHashTable &src , const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager());

	/*----create hash table with 16 buckets and max load factor 0.75----------*/
	HQClosedHashTable(const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager())
		: m_numBuckets(16)  , m_maxLoadFactor(0.75f) , m_pMemManager(pMemoryManager)
	{
		m_numItems = 0;
		m_buckets = this->MallocBuckets(m_numBuckets);
		if (m_buckets == NULL)
		{
			throw std::bad_alloc();
		}
	}
	/*----create hash table with max load factor 0.75----------*/
	HQClosedHashTable(hq_uint32 numBuckets , const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager())
		: m_numBuckets(numBuckets)  , m_maxLoadFactor(0.75f) , m_pMemManager(pMemoryManager)
	{
		m_numItems = 0;
		m_buckets = this->MallocBuckets(m_numBuckets);
		if (m_buckets == NULL)
		{
			throw std::bad_alloc();
		}
	}
	HQClosedHashTable(hq_uint32 numBuckets , hq_float32 maxLoadFactor , const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager())
		: m_numBuckets(numBuckets)  , m_maxLoadFactor(maxLoadFactor) , m_pMemManager(pMemoryManager)
	{
		m_numItems = 0;
		m_buckets = this->MallocBuckets(m_numBuckets);
		if (m_buckets == NULL)
		{
			throw std::bad_alloc();
		}
	}

	~HQClosedHashTable()
	{
		RemoveAll();
		this->FreeBuckets(m_buckets , m_numBuckets);
	}

	HQClosedHashTable & operator = (const HQClosedHashTable & src);//copy operator

	hq_uint32 GetNumItems() {return m_numItems;}
	hq_uint32 GetNumBuckets() {return m_numBuckets;}
	hq_float32 GetMaxLoadFactor() {return m_maxLoadFactor;}

	bool Add(const Key& key , const T & value , hq_uint32 *pBucketIndex = NULL);//key and data will be copied from <key> & <val> using copy operator (=)

	void RemoveAt(hq_uint32 bucketIndex) {if (bucketIndex < this->m_numBuckets) this->RemoveAtNonCheck(bucketIndex);}
	void Remove(const Key & key);
	void RemoveAll() ;

	T* GetItemPointer(const Key& key);
	const T* GetItemPointer(const Key& key)const;
	T* GetItemPointerAt(hq_uint32 bucketIndex);
	const T* GetItemPointerAt(hq_uint32 bucketIndex)const;

	T& GetItem(const Key& key , bool &isFound);
	const T& GetItem(const Key& key , bool &isFound) const;
	T& GetItemAt(hq_uint32 bucketIndex , bool &isFound);
	const T& GetItemAt(hq_uint32 bucketIndex , bool &isFound) const;

	bool Find(const Key& key , hq_uint32 &foundIndex) const;//Find existing key . Note : This class resolves collision by using quadraric probing

	void GetIterator(Iterator& iterator);
protected:
	HashTableBucketType *MallocBuckets(hq_uint32 numBuckets) ;

	void FreeBuckets(HashTableBucketType * buckets , hq_uint32 numBuckets);

	virtual hq_uint32 GetNewSize () {return this->m_numBuckets * 2;}//resize number of buckets after ratio between number of used slot and number of buckets exceeds load factor
	bool Resize();//resize buckets after ratio between number of used slot and number of buckets exceeds load factor

	bool FindEmptySlot(const Key& key , hq_uint32 &foundIndex);//this class resolves collision by using quadraric probing
	void RemoveAtNonCheck(hq_uint32 bucketIndex);

	T& GetItemNonCheck(hq_uint32 bucketIndex);
	const T& GetItemNonCheck(hq_uint32 bucketIndex) const;

	HashTableBucketType* m_buckets;

	static T m_invalidValue;
	static Key m_invalidKey;

	hq_uint32 m_numBuckets;
	hq_uint32 m_numItems;//num allocated slot
	hq_float32 m_maxLoadFactor;
	HashFunction m_hashFunction;
	ProbingFunction m_probeFunction;
	KeyEqual m_keyEqual;
	HQSharedPtr<MemoryManager> m_pMemManager;
};
template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
T HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager>:: m_invalidValue;
template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
Key HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager>:: m_invalidKey;


template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager>
	:: HQClosedHashTable(const HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> &src , const HQSharedPtr<MemoryManager> &pMemoryManager)
		:m_numBuckets(src.m_numBuckets)  , m_maxLoadFactor(src.m_maxLoadFactor) ,  m_numItems(src.m_numItems) , m_pMemManager(pMemoryManager)
{
	m_buckets = this->MallocBuckets(m_numBuckets);
	if (m_buckets == NULL)
	{
		throw std::bad_alloc();
	}
	for (hq_uint32 i = 0 ; i < m_numBuckets ; ++i)
	{
		this->m_buckets[i] = src.m_buckets[i];
	}
}

template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> &
	HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> ::
		operator =(const HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> &src)
{
	/*------release old data-------------*/
	this->RemoveAll();
	this->FreeBuckets(m_buckets , m_numBuckets);

	/*-----------------------------------*/
	this->m_numBuckets = src.m_numBuckets;


	m_buckets = this->MallocBuckets(m_numBuckets);

	if (m_buckets != NULL)
	{
		for (hq_uint32 i = 0 ; i < m_numBuckets ; ++i)
		{
			this->m_buckets[i] = src.m_buckets[i];
		}

		this->m_maxLoadFactor = src.m_maxLoadFactor;
		this->m_numItems = src.m_numItems;
	}
	else
		this->m_numBuckets = 0;

	return *this;
}

template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
HQHashTableBucket<Key , T> *HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> ::MallocBuckets(hq_uint32 numBuckets)
{
	HQHashTableBucket<Key , T> *l_buckets = static_cast<HQHashTableBucket<Key , T>*>(m_pMemManager->Malloc(sizeof(HQHashTableBucket<Key , T>) * numBuckets));
	if (l_buckets == NULL)
		return NULL;
	HQHashTableBucket<Key , T> * ptr ;
	for (hq_uint32 i = 0 ; i < numBuckets; ++i)
	{
		ptr = new (l_buckets + i) HQHashTableBucket<Key , T>();
	}
	return l_buckets;
}

template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
void HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> ::FreeBuckets(HashTableBucketType * buckets , hq_uint32 numBuckets)
{

	for (hq_uint32 i = 0 ; i < numBuckets ; ++i)
		buckets[i].~HashTableBucketType();
	this->m_pMemManager->Free(buckets);
}

template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
bool HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> ::Resize()
{
	HashTableBucketType* newPtr = this->MallocBuckets(this->GetNewSize());
	if (newPtr == NULL)
		return false;
	HashTableBucketType* oldBuckets = this->m_buckets;
	hq_uint32 oldNumBuckets = this->m_numBuckets;
	this->m_buckets = newPtr;
	this->m_numBuckets = this->GetNewSize ();
	hq_uint32 foundIndex;
	bool failed = false;
	hq_uint32 i = 0;

	//change the order of items in table
	while(!failed && i < oldNumBuckets)
	{
		if (!oldBuckets[i].m_isEmpty)
		{
			if (this->FindEmptySlot(oldBuckets[i].m_entry.m_first, foundIndex))
			{
				this->m_buckets[foundIndex] = oldBuckets[i];
			}
			else
				failed = true;
		}

		++i;
	}
	if (failed)
	{
		this->FreeBuckets(this->m_buckets , m_numBuckets);
		this->m_buckets = oldBuckets;
		this->m_numBuckets = oldNumBuckets;

		return failed;
	}

	this->FreeBuckets(oldBuckets ,oldNumBuckets );

	return true;
};

template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
bool HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> :: FindEmptySlot(const Key& key , hq_uint32 &foundIndex)
{
	hq_uint32 hashCode = m_hashFunction(key) % this->m_numBuckets;
	hq_uint32 i = 0 ;
	bool found = false;
	while(!found && i < this->m_numBuckets)
	{
		foundIndex = this->m_probeFunction(hashCode , i) % this->m_numBuckets;

		if (m_buckets[foundIndex].m_isEmpty == 1)
			found = true;
		else ++i;
	}
	return found;
}

template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
bool HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> :: Find(const Key& key , hq_uint32 &foundIndex) const
{
	if (this->m_numItems == 0)
		return false;
	hq_uint32 hashCode = m_hashFunction(key) % this->m_numBuckets;
	hq_uint32 i = 0 ;
	bool found = false;
	while(!found && i < this->m_numBuckets)
	{
		foundIndex = this->m_probeFunction(hashCode , i) % this->m_numBuckets;

		if (m_buckets[foundIndex].m_isEmpty == 0 &&
			m_keyEqual(m_buckets[foundIndex].m_entry.m_first , key))
		{
			found = true;
		}
		else
			++i;
	}
	return found;
}

template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
bool HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> :: Add(const Key& key , const T & value , hq_uint32 *pBucketIndex)
{
	hq_uint32 index;
	if (this->Find(key , index))//this key already exists
		return false;
	if ((hq_float32)this->m_numItems / this->m_numBuckets >= this->m_maxLoadFactor)//ratio between number of used slot and number of buckets exceeds load factor
	{
		if (!this->Resize())//can't resize
			return false;
	}
	if (this->FindEmptySlot(key , index))
	{
		this->m_buckets[index].m_isEmpty = 0;
		this->m_buckets[index].m_entry.m_first = key;
		this->m_buckets[index].m_entry.m_second = value;
		if (pBucketIndex != NULL)
			*pBucketIndex = index;
		this->m_numItems ++;

		if ((hq_float32)this->m_numItems / this->m_numBuckets >= this->m_maxLoadFactor)//ratio between number of used slot and number of buckets exceeds load factor
		{
			this->Resize();
		}

		return true;
	}
	return false;
}

template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
void HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> :: RemoveAtNonCheck(hq_uint32 index)
{
	if (m_buckets[index].m_isEmpty == 0)
	{
		m_buckets[index].m_isEmpty = 1;
		m_buckets[index].m_entry.m_first = HashTableType::m_invalidKey;
		m_buckets[index].m_entry.m_second = HashTableType::m_invalidValue;
		this->m_numItems --;
	}
}

template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
void HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> :: Remove(const Key & key)
{
	hq_uint32 foundIndex;
	if (this->Find(key , foundIndex))
		this->RemoveAtNonCheck(foundIndex);
}

template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
void HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> :: RemoveAll()
{
	for (hq_uint32 i = 0; i < this->m_numBuckets ; ++i)
	{
		this->RemoveAtNonCheck(i);
	}
}

template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
inline T& HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> :: GetItemNonCheck(hq_uint32 bucketIndex)
{
	return m_buckets[bucketIndex].m_entry.m_second;
}
template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
inline const T& HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> :: GetItemNonCheck(hq_uint32 bucketIndex) const
{
	return m_buckets[bucketIndex].m_entry.m_second;
}

template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
T* HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> :: GetItemPointer(const Key& key)
{
	hq_uint32 foundIndex;
	if (this->Find(key , foundIndex))
	{
		return &this->GetItemNonCheck(foundIndex);
	}

	return NULL;
}
template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
const T* HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> :: GetItemPointer(const Key& key) const
{
	hq_uint32 foundIndex;
	if (this->Find(key , foundIndex))
	{
		return &this->GetItemNonCheck(foundIndex);
	}

	return NULL;
}
template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
T* HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> :: GetItemPointerAt(hq_uint32 bucketIndex)
{
	if (bucketIndex < this->m_numBuckets && m_buckets[bucketIndex].m_isEmpty == 0)
		return &this->GetItemNonCheck(bucketIndex);
	return NULL;
}
template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
const T* HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> :: GetItemPointerAt(hq_uint32 bucketIndex) const
{
	if (bucketIndex < this->m_numBuckets && m_buckets[bucketIndex].m_isEmpty == 0)
		return &this->GetItemNonCheck(bucketIndex);
	return NULL;
}

template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
T& HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> :: GetItem(const Key& key , bool &isFound)
{
	hq_uint32 foundIndex;
	if ((isFound = this->Find(key , foundIndex)) == true)
	{
		return this->GetItemNonCheck(foundIndex);
	}
	else
		return this->m_invalidValue;
}
template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
const T& HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> :: GetItem(const Key& key , bool &isFound) const
{
	hq_uint32 foundIndex;
	if ((isFound = this->Find(key , foundIndex)) == true)
	{
		return this->GetItemNonCheck(foundIndex);
	}
	else
		return this->m_invalidValue;
}
template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
T& HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> :: GetItemAt(hq_uint32 bucketIndex , bool &isFound)
{
	if ((isFound = (bucketIndex < this->m_numBuckets  && m_buckets[bucketIndex].m_isEmpty == 0)) == true)
		return this->GetItemNonCheck(bucketIndex);
	else
		return this->m_invalidValue;
}
template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
const T& HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> :: GetItemAt(hq_uint32 bucketIndex , bool &isFound) const
{
	if ((isFound = (bucketIndex < this->m_numBuckets  && m_buckets[bucketIndex].m_isEmpty == 0)) == true)
		return this->GetItemNonCheck(bucketIndex);
	else
		return this->m_invalidValue;
}

template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
void HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> ::
	GetIterator(typename HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager> :: Iterator & iterator)
{
	iterator.m_buckets = this->m_buckets;
	iterator.currentPos = 0;
	iterator.maxPos = this->m_numBuckets;
	if(this->m_buckets != NULL && m_buckets[0].m_isEmpty == 1)
	{
		++iterator;
	}
}


/*-----------iterator class------------*/
template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
Key HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager>::Iterator::m_invalidKey;


template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
typename HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager>::Iterator
	HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager>::Iterator::operator ++(hq_int32 i)
{
	Iterator preAdd = *this;
	while(currentPos < maxPos )
	{
		++currentPos;
		if(currentPos!=maxPos && m_buckets[currentPos].m_isEmpty == 0)
			break;
	}
	return preAdd;
}

template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
typename HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager>::Iterator&
	HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager>::Iterator::operator ++()
{
	while(currentPos < maxPos )
	{
		++currentPos;
		if(currentPos!=maxPos && m_buckets[currentPos].m_isEmpty == 0)
			break;
	}
	return *this;
}

template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
typename HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager>::Iterator
	HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager>::Iterator::operator --(hq_int32 i)
{
	Iterator preSub = *this;
	while(currentPos > 0 )
	{
		if(m_buckets[--currentPos].m_isEmpty == 0)
			break;
	}
	return preSub;
}

template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
typename HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager>::Iterator&
	HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager>::Iterator::operator --()
{
	while(currentPos > 0 )
	{
		if(m_buckets[--currentPos].m_isEmpty == 0)
			break;
	}
	return *this;
}

template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
inline T* HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager>::Iterator::operator->()
{
	HQ_ASSERT(currentPos != maxPos && m_buckets[currentPos].m_isEmpty == 0);

	return &m_buckets[currentPos].m_entry.m_second;
}

template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
inline T& HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager>::Iterator::operator*()
{
	HQ_ASSERT(currentPos != maxPos && m_buckets[currentPos].m_isEmpty == 0);

	return m_buckets[currentPos].m_entry.m_second;
}
template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
inline T* HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager>::Iterator::GetItemPointer()
{
	if(currentPos == maxPos || m_buckets[currentPos].m_isEmpty == 1)
	{
		return NULL;
	}
	return &m_buckets[currentPos].m_entry.m_second;
}
template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
inline T* HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager>::Iterator::GetItemPointerNonCheck()
{
	return &m_buckets[currentPos].m_entry.m_second;
}

template <class Key ,class T ,class HashFunction , class ProbingFunction , class KeyEqual ,class MemoryManager >
const Key& HQClosedHashTable<Key , T , HashFunction , ProbingFunction , KeyEqual , MemoryManager>::Iterator::GetKey()
{
	if(currentPos == maxPos || m_buckets[currentPos].m_isEmpty == 1)
	{
		return m_invalidKey;
	}
	return m_buckets[currentPos].m_entry.m_first;
}

#include "HQClosedStringHashTable.h"

#endif
