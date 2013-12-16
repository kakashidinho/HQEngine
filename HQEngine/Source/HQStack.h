/*********************************************************************
*Copyright 2010 Le Hoang Quyen. All rights reserved.
*********************************************************************/
#ifndef HQ_STACK_H
#define HQ_STACK_H

#include "HQPrimitiveDataType.h"

/*-------unresizable stack-----------*/
template <class T>
class HQStack
{
public:
	HQStack(hq_uint32 maxSize) : m_numElems(0) , m_maxSize(maxSize) 
	{m_elements = HQ_NEW T[maxSize];}
	~HQStack()
	{delete[] m_elements;}


	bool Push(const T& val);//data is copy using copy operator (=)
	void Pop();
	void RemoveAll() { m_numElems = 0; }
	
	T* GetTop() {if (m_numElems > 0) return &m_elements[m_numElems - 1]; return NULL;}
	T* operator [] (hq_uint index){if (index < m_numElems) return &m_elements[index] ; return NULL}//item in the bottom of stack has index 0

	hq_uint32 GetNumElements() {return m_numElems;}
	hq_uint32 GetMaxSize() {return m_maxSize;}
private:
	T *m_elements;
	hq_uint32 m_numElems;
	hq_uint32 m_maxSize;
};

template <class T>
inline bool HQStack<T>::Push(const T &val)
{
	if (m_numElems >= m_maxSize)
		return false;
	m_elements[m_numElems ++] = val;
	return true;
}

template <class T>
inline void HQStack<T>::Pop()
{
	if (m_numElems != 0)
		--m_numElems ;
}

#endif