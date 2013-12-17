/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_REF_COUNT_OBJECT_H
#define HQ_REF_COUNT_OBJECT_H

#include "HQPrimitiveDataType.h"

class HQReferenceCountObj {
public:
	HQReferenceCountObj();

	hquint32 GetRefCount() const {return m_refCount;}
	void AddRef();
	hquint32 Release();
protected:
	virtual ~HQReferenceCountObj();
private:
	hquint32 m_refCount;///reference count
};


inline HQReferenceCountObj::HQReferenceCountObj()
:m_refCount(1)
{
}

inline HQReferenceCountObj::~HQReferenceCountObj()
{
}

inline void HQReferenceCountObj::AddRef()
{
	m_refCount ++;
}

inline hquint32 HQReferenceCountObj::Release()
{
	m_refCount --;
	if (m_refCount == 0)
	{
		delete this;

		return 0;
	}

	return m_refCount;
}


#endif
