/*********************************************************************
*Copyright 2011 Le Hoang Quyen. All rights reserved.
*********************************************************************/
#ifndef HQ_HASH_TABLE_H
#define HQ_HASH_TABLE_H

#include <iostream>
#include <string.h>//memset
#include "HQMemoryManager.h"
#include "HQDefaultHashFunction.h"
#include "HQDefaultKeyEqual.h"
#include "HQPair.h"
#include "HQLinkedList.h"


/*-----------this hash table resolves collision using seperate chaining-----*/
template
<
class Key ,
class T ,
class HashFunction = HQDefaultHashFunc<Key> ,
class KeyEqual = HQDefaultKeyEqual<Key> ,
class MemoryManager = HQDefaultMemManager //Memory manager for malloc and free memory block that has size sizeof(HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager>::BucketNodeType)
>

class HQHashTable
{
protected:
	typedef HQLinkedList<HQPair<Key , T> , MemoryManager> LinkedListType;
public:
	typedef HQLinkedListNode<HQPair<Key , T> > BucketNodeType;
	/*------iterator class - for traversing through list of items in this manager---*/
	class Iterator
	{
		friend class HQHashTable;
	private:
		LinkedListType * m_buckets;
		typename LinkedListType::Iterator m_listIte;
		hq_uint32 currentPos;
		hq_uint32 maxPos;

		static Key m_invalidKey;
	public:
		Iterator& operator ++();//prefix addition . shift to next item (shift right)
		Iterator operator ++(hq_int32);//suffic addition . shift to next item (shift right)

		Iterator& operator --();//prefix subtraction . shift to prev item (shift left)
		Iterator operator --(hq_int32);//suffix subtraction . shift to prev item (shift left)

		bool IsAtBegin(){return (currentPos == 0 && this->m_listIte.IsAtBegin());};//is at location that can't shift left
		bool IsAtEnd(){return (currentPos == maxPos);};//is at invalid location (ie location that can't shift right)

		void Rewind(){currentPos = 0;};//go to location that can't shift left

		T* operator->();
		T& operator*();
		T* GetItemPointer();
		T* GetItemPointerNonCheck();
		const Key& GetKey();
	};


	/*----------copy constructor---------------*/
	HQHashTable(const HQHashTable &src , const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager());

	/*----create hash table with 16 buckets and max load factor 0.75----------*/
	HQHashTable(const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager()) : m_numBuckets(16)   , m_pListMemoryManager(pMemoryManager)
	{
		m_numItems = m_numSlots = 0;
		this->MallocBuckets();

	}
	HQHashTable(hq_uint32 numBuckets , const HQSharedPtr<MemoryManager> &pMemoryManager = HQ_NEW MemoryManager()) : m_numBuckets(numBuckets)   , m_pListMemoryManager(pMemoryManager)
	{
		m_numItems = m_numSlots = 0;
		this->MallocBuckets();
	}

	~HQHashTable()
	{
		FreeBuckets();
	}

	HQHashTable & operator = (const HQHashTable & src);//copy operator

	hq_uint32 GetNumItems() {return m_numItems;}
	hq_uint32 GetNumBuckets() {return m_numBuckets;}

	bool Add(const Key& key , const T & value);

	void RemoveAt(hq_uint32 bucketIndex) {if (bucketIndex < this->m_numBuckets) this->RemoveAtNonCheck(bucketIndex);}
	void Remove(const Key & key);
	void RemoveAll() ;

	T* GetItemPointer(const Key& key);
	const T* GetItemPointer(const Key& key)const;

	T& GetItem(const Key& key , bool &isFound);
	const T& GetItem(const Key& key , bool &isFound) const;

	bool Contain(const Key& key) const;//Find existing key

	void GetIterator(Iterator& iterator);
protected:
	void MallocBuckets() ;
	void FreeBuckets() ;

	inline HQLinkedListNode<HQPair<Key , T> >* Find(const Key& key) const  { hq_uint32 index = this->FindSlot(key) ; return this->Find(index , key);}//check if key is stored
	HQLinkedListNode<HQPair<Key , T> >* Find(hq_uint32 slot , const Key& key) const;//check if key is stored in this slot
	inline hq_uint32 FindSlot(const Key& key) const {return m_hashFunction(key) % m_numBuckets;}

	LinkedListType * m_buckets;

	static T m_invalidValue;

	hq_uint32 m_numBuckets;
	hq_uint32 m_numSlots;//num allocated slots
	hq_uint32 m_numItems;//number of items stored in table
	HashFunction m_hashFunction;
	KeyEqual m_keyEqual;
	HQSharedPtr<MemoryManager> m_pListMemoryManager;
};
template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
T HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager>:: m_invalidValue;

template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager> :: HQHashTable(const HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager> &src , const HQSharedPtr<MemoryManager> &pMemoryManager)
:m_numBuckets(src.m_numBuckets)  ,  m_numItems(src.GetNumItems) , m_numSlots(src.m_numSlots)   , m_pListMemoryManager(pMemoryManager)
{
	this->MallocBuckets(pMemoryManager);
	for (hq_uint32 i = 0 ; i < m_numBuckets ; ++i)
	{
		this->m_buckets[i] = src.m_buckets[i];
	}
}

template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager> &
	HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager> ::
		operator =(const HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager> &src)
{
	/*------release old data-------------*/
	this->FreeBuckets();

	/*-----------------------------------*/
	this->m_numBuckets = src.m_numBuckets;

	this->MallocBuckets();
	if (m_buckets == NULL)
	{
		this->m_numBuckets = 0;
		return *this;
	}

	for (hq_uint32 i = 0 ; i < m_numBuckets ; ++i)
	{
		this->m_buckets[i] = src.m_buckets[i];
	}

	this->m_numItems = src.m_numItems;
	this->m_numSlots = src.m_numSlots;

	return *this;
}


template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
void HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager> :: MallocBuckets()
{
	m_buckets = static_cast<LinkedListType*>(malloc(sizeof(HQLinkedList<HQPair<Key , T> , MemoryManager>) * m_numBuckets));
	if (m_buckets == NULL)
	{
		throw std::bad_alloc();
	}
	HQLinkedList<HQPair<Key , T> , MemoryManager> * ptr ;
	for (hq_uint32 i = 0 ; i < m_numBuckets; ++i)
	{
		ptr = new (m_buckets + i) HQLinkedList<HQPair<Key , T> , MemoryManager>(m_pListMemoryManager);
	}
}

template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
void HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager> :: FreeBuckets()
{
	for (hq_uint32 i = 0 ; i < m_numBuckets ; ++i)
		m_buckets[i].~LinkedListType();
	free(m_buckets);
}


template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
bool HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager> :: Contain(const Key& key) const
{
	if (this->m_numItems == 0)
		return false;
	return (this->Find(key) != NULL);
}

template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
HQLinkedListNode<HQPair<Key , T> >* HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager> :: Find(hq_uint32 slot , const Key& key) const
{
	HQLinkedListNode<HQPair<Key , T> >* foundNode = NULL;
	typename HQLinkedList<HQPair<Key , T> , MemoryManager>::Iterator ite;

	this->m_buckets[slot].GetIterator(ite);

	while(foundNode == NULL && !ite.IsAtEnd())
	{
		if (m_keyEqual(ite->m_first , key))
		{
			foundNode = ite.GetNode();
		}
		else
			++ite;
	}
	return foundNode;
}

template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
bool HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager> :: Add(const Key& key , const T & value)
{
	hq_uint32 index = this->FindSlot(key) ;

	if (this->Find(index , key) != NULL)//this key already exists in this table
		return false;

	HQPair<Key , T > newItem(key , value);
	if (!m_buckets[index].PushBack(newItem))
		return false;
	this->m_numItems ++;

	if (m_buckets[index].GetSize() == 1)
		this->m_numSlots ++;

	return true;
}


template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
void HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager> :: Remove(const Key & key)
{
	hq_uint32 index = this->FindSlot(key);
	HQLinkedListNode<HQPair<Key , T> >* foundNode = this->Find(index , key);
	if (foundNode != NULL)
	{
		m_buckets[index].RemoveAt(foundNode);
		this->m_numItems --;
		if(m_buckets[index].GetSize() == 0)
			this->m_numSlots--;
	}
}

template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
void HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager> :: RemoveAll()
{
	for (hq_uint32 i = 0; i < this->m_numBuckets ; ++i)
	{
		m_buckets[i].RemoveAll();
	}

	this->m_numSlots = 0;
	this->m_numItems = 0;
}


template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
T* HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager> :: GetItemPointer(const Key& key)
{
	HQLinkedListNode<HQPair<Key , T> >* foundNode = this->Find(key);
	if (foundNode != NULL)
	{
		return &foundNode->m_element.m_second;
	}

	return NULL;
}
template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
const T* HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager> :: GetItemPointer(const Key& key) const
{
	HQLinkedListNode<HQPair<Key , T> >* foundNode = this->Find(key);
	if (foundNode != NULL)
	{
		return &foundNode->m_element.m_second;
	}

	return NULL;
}

template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
T& HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager> :: GetItem(const Key& key , bool &isFound)
{
	HQLinkedListNode<HQPair<Key , T> >* foundNode = this->Find(key);
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
template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
const T& HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager> :: GetItem(const Key& key , bool &isFound) const
{
	HQLinkedListNode<HQPair<Key , T> >* foundNode = this->Find(key);
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


template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
void HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager> ::
	GetIterator(typename HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager> :: Iterator & iterator)
{
	iterator.m_buckets = this->m_buckets;
	iterator.currentPos = 0;
	iterator.maxPos = this->m_numBuckets;
	if(this->m_buckets != NULL )
	{
		if (m_buckets[0].GetSize() == 0)
		{
			iterator.m_listIte.Invalid();
			++iterator;
		}
		else
			m_buckets[0].GetIterator(iterator.m_listIte);
	}

}


/*-----------iterator class------------*/

template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
Key HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager>::Iterator::m_invalidKey;


template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
typename HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager>::Iterator
	HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager>::Iterator::operator ++(hq_int32 i)
{
	Iterator preAdd = *this;
	++this->m_listIte;
	if (this->m_listIte.IsAtEnd())
	{
		while(currentPos < maxPos )
		{
			++currentPos;
			if(currentPos!=maxPos && m_buckets[currentPos].GetSize() > 0)
			{
				break;
			}
		}
		if (currentPos < maxPos)
			m_buckets[currentPos].GetIterator(this->m_listIte);
		else
			this->m_listIte.Invalid();
	}

	return preAdd;
}

template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
typename HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager>::Iterator&
	HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager>::Iterator::operator ++()
{
	++this->m_listIte;
	if (this->m_listIte.IsAtEnd())
	{
		while(currentPos < maxPos )
		{
			++currentPos;
			if(currentPos!=maxPos && m_buckets[currentPos].GetSize() > 0)
			{
				break;
			}
		}
		if (currentPos < maxPos)
			m_buckets[currentPos].GetIterator(this->m_listIte);
		else
			this->m_listIte.Invalid();
	}
	return *this;
}

template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
typename HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager>::Iterator
	HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager>::Iterator::operator --(hq_int32 i)
{
	Iterator preSub = *this;
	if (this->m_listIte.IsAtBegin())
	{
		while(currentPos > 0 )
		{
			if(m_buckets[--currentPos].GetSize() > 0)
			{
				break;
			}
		}
		m_buckets[currentPos].GetIteratorFromLastItem(this->m_listIte);
	}
	else
		--this->m_listIte;
	return preSub;
}

template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
typename HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager>::Iterator&
	HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager>::Iterator::operator --()
{
	if (this->m_listIte.IsAtBegin())
	{
		while(currentPos > 0 )
		{
			if(m_buckets[--currentPos].GetSize() > 0)
			{
				break;
			}
		}
		m_buckets[currentPos].GetIteratorFromLastItem(this->m_listIte);
	}
	else
		--this->m_listIte;
	return *this;
}

template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
inline T* HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager>::Iterator::operator->()
{
	HQ_ASSERT(!this->m_listIte.IsAtEnd());

	return &this->m_listIte.GetPointerNonCheck()->m_second;
}

template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
inline T& HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager>::Iterator::operator *()
{
	HQ_ASSERT(!this->m_listIte.IsAtEnd());

	return this->m_listIte.GetPointerNonCheck()->m_second;
}

template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
inline T* HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager>::Iterator::GetItemPointer()
{
	if(this->m_listIte.IsAtEnd())
		return NULL;
	return &this->m_listIte.GetPointerNonCheck()->m_second;
}
template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
inline T* HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager>::Iterator::GetItemPointerNonCheck()
{
	return &this->m_listIte.GetPointerNonCheck()->m_second;
}

template <class Key ,class T ,class HashFunction , class KeyEqual ,class MemoryManager >
const Key& HQHashTable<Key , T , HashFunction , KeyEqual , MemoryManager>::Iterator::GetKey()
{
	if(this->m_listIte.IsAtEnd())
	{
		return m_invalidKey;
	}
	return this->m_listIte.GetPointerNonCheck()->m_first;
}




#include "HQStringHashTable.h"

#endif
