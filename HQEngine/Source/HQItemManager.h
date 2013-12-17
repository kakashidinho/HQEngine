/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _HQ_ITEM_MAN_H_
#define _HQ_ITEM_MAN_H_
#include "HQSharedPointer.h"

#include <new>
#if defined DEBUG || defined _DEBUG
#include <assert.h>
#endif

#define MAX_ID 0xcdcdcdcd
#define INVALID_ID 0xcdcdcdcd
#define ID_NOT_FOUND 0xffffffff

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

/*-------------------------------------------------------------
template class HQItemManager - manages unordered list of items
--------------------------------------------------------------*/
template <class T>
class HQItemManager
{
private:
	struct SlotType
	{
		SlotType(SlotType * _nextSlot) : nextSlot(_nextSlot) , sharedPtr(HQSharedPtr<T>::null) {}
		SlotType * nextSlot;
		HQSharedPtr<T> sharedPtr;
	};
public:
	/*------iterator class - for traversing through list of items in this manager---*/
	class Iterator
	{
		friend class HQItemManager;
	private:
		SlotType* m_slots;
		hq_uint32 currentPos;
		hq_uint32 maxPos;
	public:
		Iterator& operator ++();//prefix addition . shift to next item
		Iterator operator ++(hq_int32);//suffic addition  . shift to next item

		Iterator& operator --();//prefix subtraction  . shift to prev item
		Iterator operator --(hq_int32);//suffix subtraction  . shift to prev item
		
		bool IsAtBegin(){return (currentPos == 0);};//is at first slot
		bool IsAtEnd(){return (currentPos == maxPos);};//is at invalid slot (ie. slot after last slot)
		
		void Rewind(){currentPos = 0;};//go to first slot

		HQSharedPtr<T> operator->();//unsafe
		T& operator *();//unsafe
		HQSharedPtr<T> GetItemPointer();
		HQSharedPtr<T> GetItemPointerNonCheck();//unsafe
		hq_uint32 GetID() {return currentPos;}
	};

	HQItemManager(hq_uint32 growQuantity = 20);
	~HQItemManager();
	/*
	add new item to list , pItem is pointer to item object that going to be added.pItem must not be an array
	Note: <pItem> must points to dynamc allocated memmory block,or else it will cause undefined behavior
	and that memory block must not be deallocated outside HQItemManager class objects,because it will be
	dealloc inside itemManager object that contains it.
	*/
	bool AddItem(T* pItem,hq_uint32 *itemID);
	bool AddItem(const HQSharedPtr<T> & pItem,hq_uint32 *itemID);

	hq_int32 Remove(hq_uint32 ID);//remove item that has id <ID>

	void RemoveAll();
	
	HQSharedPtr<T> GetItemPointer(hq_uint32 ID);//get pointer to item which has id <ID>
	T* GetItemRawPointer(hq_uint32 ID);//get raw pointer to item which has id <ID> . use with caution , don't assign this pointer to another shared pointer and don't delete this pointer
	HQSharedPtr<T> operator[] (hq_uint32 ID) {return this->GetItemPointer(ID);}
	HQSharedPtr<T> GetItemPointerNonCheck(hq_uint32 ID);//get pointer to item which has id <ID>.Doesn't check invalid ID.So application need to be careful when use this function
	//get raw pointer to item which has id <ID>.Doesn't check invalid ID.So application need to be careful when use this function. use with caution , don't assign this pointer to another shared pointer and don't delete this pointer
	T* GetItemRawPointerNonCheck(hq_uint32 ID);
	hq_uint32 GetNumItems() {return m_numItems;}
	void GetIterator(Iterator& iterator);//get iterator starting at the first item slot
protected:
	SlotType * GetFreeSlot();

	SlotType * m_slots;//item slots
	SlotType * m_pFirstFreeSlot;  
	hq_uint32 m_numSlots;//number of slots
	hq_uint32 m_numItems;//number of items
	hq_uint32 m_growQuantity;
};


template <class T>
HQItemManager<T>::HQItemManager(hq_uint32 growQuantity)
{
	this->m_numItems = 0;
	this->m_numSlots = 0;
	this->m_growQuantity = growQuantity;
	this->m_slots=NULL;
	this->m_pFirstFreeSlot = NULL;
}

template <class T>
HQItemManager<T>::~HQItemManager()
{
	RemoveAll();
}

template <class T>
void HQItemManager<T>::RemoveAll()
{
	if(m_slots != NULL)
	{
		for (hq_uint32 i=0;i<m_numSlots;++i)
		{
			m_slots[i].sharedPtr.~HQSharedPtr<T>();
		}
		free(m_slots);
		m_slots=NULL;
	}
	m_numItems=0;
	m_numSlots=0;
	m_pFirstFreeSlot = NULL;
}

template <class T>
hq_int32 HQItemManager<T>::Remove(hq_uint32 ID)
{
	if(ID >=m_numSlots || m_slots[ID].sharedPtr==NULL)
		return ID_NOT_FOUND;//invalid ID
	
	typename HQItemManager<T>::SlotType * slot = m_slots + ID;

	//free this slot
	slot->sharedPtr.ToNull();

	slot->nextSlot = m_pFirstFreeSlot;//store address of next free slot
	
	//mark this slot as the first free slot
	m_pFirstFreeSlot = slot;
	
	m_numItems--;
	return 1;
}
template <class T>
typename HQItemManager<T>::SlotType * HQItemManager<T>::GetFreeSlot()
{
	typename HQItemManager<T>::SlotType * pfreeSlot=NULL;

	if(m_pFirstFreeSlot != NULL)//we have free slot(s)
	{
		pfreeSlot = m_pFirstFreeSlot;

	}
	else
	{
		if(this->m_numSlots > MAX_ID)//exceed max allowed number of m_slots
			return NULL;
		hq_uint32 newNumSlots = this->m_numSlots + m_growQuantity;
		if (newNumSlots - 1 > MAX_ID)
			newNumSlots = MAX_ID + 1;

		typename HQItemManager<T>::SlotType *newPtr=(typename HQItemManager<T>::SlotType*)realloc(m_slots,newNumSlots * sizeof(typename HQItemManager<T>::SlotType));
		if(newPtr==NULL)
			return NULL;
		m_slots=newPtr;
		
		pfreeSlot= &m_slots[m_numSlots];
		
		hq_uint32 lastSlotID = newNumSlots - 1;
		new (&m_slots[lastSlotID]) typename HQItemManager<T>::SlotType(NULL);//initially,next free slot after last slot is NULL

		for (hq_uint32 i = lastSlotID ; i > m_numSlots; --i)
		{
			new (&m_slots[i - 1])	typename HQItemManager<T>::SlotType(m_slots + i);//initially,next free slot after slot (i - 1) simply is slot (i)
		}
			
		this->m_numSlots = newNumSlots;
	}

	return pfreeSlot;
}

template <class T>
bool HQItemManager<T>::AddItem(T* pItem,hq_uint32 *itemID)
{
	typename HQItemManager<T>::SlotType* pfreeSlot = this->GetFreeSlot();

	if (pfreeSlot == NULL)
		return false;

	pfreeSlot->sharedPtr = HQSharedPtr<T>(pItem);
	
	m_numItems ++;

	m_pFirstFreeSlot = pfreeSlot->nextSlot;//assign next free slot address

	if(itemID)
	{
		*itemID = (hq_uint32) (pfreeSlot - m_slots);
	}

	return true;
}

template <class T>
bool HQItemManager<T>::AddItem(const HQSharedPtr<T>& pItem,hq_uint32 *itemID)
{
	typename HQItemManager<T>::SlotType* pfreeSlot = this->GetFreeSlot();

	if (pfreeSlot == NULL)
		return false;

	pfreeSlot->sharedPtr = pItem;
	
	m_numItems ++;

	m_pFirstFreeSlot = pfreeSlot->nextSlot;//assign next free slot address

	if(itemID)
	{
		*itemID = (hq_uint32) (pfreeSlot - m_slots);
	}

	return true;
}

template <class T>
HQ_FORCE_INLINE HQSharedPtr<T> HQItemManager<T>::GetItemPointer(hq_uint32 ID)
{
#if 0
	HQ_ASSERT(ID < m_numSlots);
#else
	if (ID >= m_numSlots)
		return HQSharedPtr<T>::null;
#endif

	return m_slots[ID].sharedPtr;
}

template <class T>
HQ_FORCE_INLINE T* HQItemManager<T>::GetItemRawPointer(hq_uint32 ID)
{
#if 0
	HQ_ASSERT(ID < m_numSlots);
#else
	if (ID >= m_numSlots)
		return NULL;
#endif

	return m_slots[ID].sharedPtr.GetRawPointer();
}

template <class T>
HQ_FORCE_INLINE HQSharedPtr<T> HQItemManager<T>::GetItemPointerNonCheck(hq_uint32 ID)
{
	return m_slots[ID].sharedPtr;
}

template <class T>
HQ_FORCE_INLINE T* HQItemManager<T>::GetItemRawPointerNonCheck(hq_uint32 ID)
{
	return m_slots[ID].sharedPtr.GetRawPointer();
}

template <class T>
void HQItemManager<T>::GetIterator(typename HQItemManager<T>::Iterator & iterator)
{
	iterator.m_slots = m_slots;
	iterator.currentPos = 0;
	iterator.maxPos = m_numSlots;
	if(m_slots != NULL && m_slots[0].sharedPtr == NULL)
		++iterator;
}

/*-----------iterator class------------*/
template <class T>
typename HQItemManager<T>::Iterator HQItemManager<T>::Iterator::operator ++(hq_int32 i)
{
	Iterator preAdd = *this;
	while(currentPos < maxPos )
	{
		++currentPos;
		if(currentPos!=maxPos && m_slots[currentPos].sharedPtr!=NULL)
			break;
	}
	return preAdd;
}

template <class T>
typename HQItemManager<T>::Iterator& HQItemManager<T>::Iterator::operator ++()
{
	while(currentPos < maxPos )
	{
		++currentPos;
		if(currentPos!=maxPos && m_slots[currentPos].sharedPtr!=NULL)
			break;
	}
	return *this;
}

template <class T>
typename HQItemManager<T>::Iterator HQItemManager<T>::Iterator::operator --(hq_int32 i)
{
	Iterator preSub = *this;
	while(currentPos > 0 )
	{
		if(m_slots[--currentPos].sharedPtr!=NULL)
			break;
	}
	return preSub;
}

template <class T>
typename HQItemManager<T>::Iterator& HQItemManager<T>::Iterator::operator --()
{
	while(currentPos > 0 )
	{
		if(m_slots[--currentPos].sharedPtr!=NULL)
			break;
	}
	return *this;
}

template<class T>
HQ_FORCE_INLINE HQSharedPtr<T> HQItemManager<T>::Iterator::operator->()
{
	return m_slots[currentPos].sharedPtr;
}

template<class T>
HQ_FORCE_INLINE HQSharedPtr<T> HQItemManager<T>::Iterator::GetItemPointer()
{
	if(currentPos == maxPos || m_slots[currentPos].sharedPtr == NULL)
	{
		return HQSharedPtr<T> ::null;
	}
	return m_slots[currentPos].sharedPtr;
}
template<class T>
HQ_FORCE_INLINE HQSharedPtr<T> HQItemManager<T>::Iterator::GetItemPointerNonCheck()
{
	return m_slots[currentPos].sharedPtr;
}

template<class T>
HQ_FORCE_INLINE T& HQItemManager<T>::Iterator::operator *()
{
	return *m_slots[currentPos].sharedPtr;
}
#endif
