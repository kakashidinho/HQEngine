/*********************************************************************
*Copyright 2010 Le Hoang Quyen. All rights reserved.
*********************************************************************/
#ifndef _GROWABLE_ARRAY_
#define _GROWABLE_ARRAY_
#ifndef WIN32
#include <cstdlib>
#endif
#include "HQPrimitiveDataType.h"
#include "HQAssert.h"
/*-------------------------------------------------
template class HQGrowableArray - implements an array
that has growable size
------------------------------------------------*/
template <class T>
class HQGrowableArray
{
public:
	
	HQGrowableArray(hq_uint32 growQuantity = 20);
	HQGrowableArray(hq_uint32 initCapacity , hq_uint32 growQuantity = 20);
	~HQGrowableArray();

	bool Add(const T& item);//add a copy of item to the end of array using copy constuctor

	T& operator[](size_t index);//member access
	T& operator[](hq_ubyte8 index);//member access
	T& operator[](hq_int32 index);//member access
	T& operator[](hq_ushort16 index);//member access
	
	const T& operator[](size_t index) const;//member access
	const T& operator[](hq_ubyte8 index) const;//member access
	const T& operator[](hq_int32 index) const;//member access
	const T& operator[](hq_ushort16 index) const;//member access

	operator T*() {return elements;}//casting operator
	operator const T*() const {return elements;}//casting operator
	size_t Size(){return size;}//current size
	void Clear();
protected:
	T* elements;
	size_t size;
	size_t capacity;
	hq_uint32 growQuantity;
};


template <class T>
HQGrowableArray<T>::HQGrowableArray(hq_uint32 _growQuantity)
:elements(NULL) , size(0) , capacity(0), growQuantity(_growQuantity)
{
}

template <class T>
HQGrowableArray<T>::HQGrowableArray(hq_uint32 initCapacity , hq_uint32 growQuantity)
: size(0), capacity(initCapacity) , growQuantity(_growQuantity)
{
	this->elements = (T*) malloc(capacity * sizeof(T));
	if (this->elements == NULL)
		throw std::bad_alloc();
}

template <class T>
HQGrowableArray<T>::~HQGrowableArray()
{
	Clear();
}

template <class T>
void HQGrowableArray<T>::Clear()
{
	if(elements)
	{
		free(elements);
		elements = NULL;
	}
	size = 0;
	capacity = 0;
}


template <class T>
bool HQGrowableArray<T>::Add(const T& item)
{

	if(this->size >= this->capacity)//we use all available slots,so alloc more slots for additional items
	{
		size_t newMaxSize=this->capacity + growQuantity;//addition <growQuantity> slots for future use

		T* newPtr=(T*)realloc(elements,newMaxSize * sizeof(T));
		if(newPtr==NULL)
			return false;

		elements=newPtr;
		this->capacity = newMaxSize;

	}
	new (elements + size) T(item);

	size++;

	return true;
}

template <class T>
inline T& HQGrowableArray<T>::operator [](size_t index)
{
	HQ_ASSERT(index < this->size);

	return elements[index];
}

template <class T>
inline T& HQGrowableArray<T>::operator [](hqint32 index)
{
	HQ_ASSERT(index < this->size && index >= 0);

	return elements[index];
}

template <class T>
inline T& HQGrowableArray<T>::operator [](hq_ushort16 index)
{
	HQ_ASSERT(index < this->size);

	return elements[index];
}


template <class T>
inline T& HQGrowableArray<T>::operator [](hq_ubyte8 index)
{
	HQ_ASSERT(index < this->size);

	return elements[index];
}

template <class T>
inline const T& HQGrowableArray<T>::operator [](size_t index) const
{
	HQ_ASSERT(index < this->size);

	return elements[index];
}

template <class T>
inline const T& HQGrowableArray<T>::operator [](hq_ushort16 index) const
{
	HQ_ASSERT(index < this->size);

	return elements[index];
}


template <class T>
inline const T& HQGrowableArray<T>::operator [](hq_ubyte8 index) const
{
	HQ_ASSERT(index < this->size);

	return elements[index];
}

template <class T>
inline const T& HQGrowableArray<T>::operator [](hqint32 index) const
{
	HQ_ASSERT(index < this->size && index >= 0);

	return elements[index];
}

#endif
