/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _HQ_SHARED_PTR_
#define _HQ_SHARED_PTR_
#include <cstdlib>
#include "HQPrimitiveDataType.h"
#include "HQAssert.h"


class HQReference
{
private:
	hq_uint32 refCount;//reference count
public:
	inline HQReference(){ refCount = 0;};
	inline hq_uint32 AddRef() {return ++refCount;};//increase reference count
	inline hq_uint32 GetRefCount() {return refCount;}
	inline hq_uint32 Release() //decrease reference count
	{
		if(refCount == 0) 
			return 0 ;
		return --refCount;
	};
};

template <class T>
struct HQPointerRelease{
	static inline void Release (T *&ptr) {if (ptr != NULL) delete ptr ; ptr = NULL;}
};

template <class T>
struct HQArrayPointerRelease{
	static inline void Release  (T *&ptr) {if (ptr != NULL) delete[] ptr ; ptr = NULL;}
};

/*---------BaseSharedPointer-------------*/
template<class T , class PointerRelease>
class HQBaseSharedPtr
{
public:
	HQBaseSharedPtr();
	HQBaseSharedPtr(T *rawptr);
	HQBaseSharedPtr(const HQBaseSharedPtr<T , PointerRelease>& sptr2);//copy constructor

	~HQBaseSharedPtr();

	bool operator!() const {return m_ptr == NULL;}
	T& operator *();
	T* operator ->();
	const T* operator ->()const;
	inline T* GetRawPointer() {return m_ptr;}
	inline const T* GetRawPointer()const {return m_ptr;}
	inline HQBaseSharedPtr<T , PointerRelease>& operator = (const HQBaseSharedPtr<T , PointerRelease>& sptr2)  {this->Copy(sptr2) ; return *this;}

	inline bool operator == (const HQBaseSharedPtr<T , PointerRelease>& sptr2)const {return m_ptr == sptr2.m_ptr;}
	inline bool operator != (const HQBaseSharedPtr<T , PointerRelease>& sptr2)const {return m_ptr != sptr2.m_ptr;};
	inline bool operator == (const T* ptr2)const {return m_ptr == ptr2;}
	inline bool operator != (const T* ptr2)const {return m_ptr != ptr2;};

	inline hq_uint32 GetRefCount() {return m_ref->GetRefCount();}
	hq_uint32 AddRef() {return m_ref->AddRef();}//increase reference count
	hq_uint32 Release() {return m_ref->Release();}//decrease reference count
	HQBaseSharedPtr<T , PointerRelease>& ToNull();

	//null pointer
	static const HQBaseSharedPtr<T , PointerRelease> null;

protected:
	T * m_ptr;//the real pointer
	mutable HQReference* m_ref;

	void Copy(const HQBaseSharedPtr<T , PointerRelease>& sptr2);

};

template <class T, class PointerRelease>
const HQBaseSharedPtr<T , PointerRelease> HQBaseSharedPtr<T , PointerRelease>::null;

template <class T, class PointerRelease>
inline HQBaseSharedPtr<T , PointerRelease>::HQBaseSharedPtr()
{
	m_ptr = NULL;
	m_ref = HQ_NEW HQReference();
	m_ref->AddRef();
}

template <class T, class PointerRelease>
inline HQBaseSharedPtr<T , PointerRelease>::HQBaseSharedPtr(T *rawptr)
: m_ptr(rawptr)
{
	if (rawptr == NULL)
	{
		m_ref = null.m_ref;//point to null pointer reference
	}
	else
	{
		m_ref = HQ_NEW HQReference();
	}

	m_ref->AddRef();
}


template <class T, class PointerRelease>
inline HQBaseSharedPtr<T , PointerRelease>::HQBaseSharedPtr(const HQBaseSharedPtr<T , PointerRelease> &sptr2)
	:m_ptr (sptr2.m_ptr) , m_ref (sptr2.m_ref)
{
	m_ref->AddRef();
}


template <class T, class PointerRelease>
inline HQBaseSharedPtr<T , PointerRelease>::~HQBaseSharedPtr()
{
	if(m_ref->Release() == 0)
	{
		PointerRelease::Release(m_ptr);
		delete m_ref;
	}
}


template <class T, class PointerRelease>
inline T& HQBaseSharedPtr<T , PointerRelease>::operator *()
{
	HQ_ASSERT(m_ptr != NULL);

	return *m_ptr;
}


template <class T, class PointerRelease>
inline T* HQBaseSharedPtr<T , PointerRelease>::operator ->()
{
	HQ_ASSERT(m_ptr != NULL);

	return m_ptr;
}

template <class T, class PointerRelease>
inline const T* HQBaseSharedPtr<T , PointerRelease>::operator ->() const
{
	HQ_ASSERT(m_ptr != NULL);

	return m_ptr;
}

template <class T, class PointerRelease>
void HQBaseSharedPtr<T , PointerRelease>::Copy(const HQBaseSharedPtr<T , PointerRelease> &sptr2)
{
	if(this!=&sptr2)//not self assign
	{
		if(m_ref->Release() == 0)
		{
			PointerRelease::Release(m_ptr);
			delete m_ref;
		}
		this->m_ptr = sptr2.m_ptr;
		this->m_ref = sptr2.m_ref;
		this->m_ref->AddRef();
	}
}

template <class T, class PointerRelease>
inline HQBaseSharedPtr<T , PointerRelease>& HQBaseSharedPtr<T , PointerRelease>::ToNull()
{
	return ((*this) = HQBaseSharedPtr<T , PointerRelease>::null);
}

/*-------------specialized shared pointer------------------*/

//Shared normal pointer
template <class T>
class HQSharedPtr : public HQBaseSharedPtr<T , HQPointerRelease<T> > {
private:
	typedef HQBaseSharedPtr<T , HQPointerRelease<T> > ParentType;
public:
	inline HQSharedPtr() : ParentType(ParentType::null) {}
	inline HQSharedPtr(T *rawptr) : ParentType (rawptr) {}
	inline HQSharedPtr(const HQSharedPtr & sptr2):ParentType(sptr2) {}//copy constructor
	inline HQSharedPtr(const ParentType & sptr2):ParentType(sptr2) {}//copy constructor
	
};

//Shared array pointer
template <class T>
class HQSharedArrayPtr : public HQBaseSharedPtr<T , HQArrayPointerRelease<T> > {
private:
	typedef HQBaseSharedPtr<T , HQArrayPointerRelease<T> > ParentType;
public:

	inline HQSharedArrayPtr() : ParentType(ParentType::null) {}
	inline HQSharedArrayPtr(T *rawptr) : ParentType  (rawptr) {}
	inline HQSharedArrayPtr(const HQSharedArrayPtr & sptr2):ParentType(sptr2) {}//copy constructor
	inline HQSharedArrayPtr(const ParentType & sptr2):ParentType(sptr2) {}//copy constructor
	inline T& operator [](size_t index) {
		return this->m_ptr[index];
	}
	inline const T& operator [](size_t index)const {
		return this->m_ptr[index];
	}
};

#endif
