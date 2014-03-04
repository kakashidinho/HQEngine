/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef _2DMATH_
#define _2DMATH_
#include "HQUtilMathCommon.h"

template <typename T = hqint32>
class HQPoint
{
public:
	T x;
	T y;
};


template <typename T = hqint32>
class HQRect
{
public:
	T x;
	T y;
	T w;
	T h;
	bool IsPointInside(T x,T y) const;
	bool IsPointInside(const HQPoint<T> &point) const;
};

#include "HQ2DMathInline.h"

typedef HQPoint<hqint32> HQPointi;
typedef HQPoint<hqfloat32> HQPointf;
typedef HQRect<hqint32> HQRecti;
typedef HQRect<hqfloat32> HQRectf;

#endif
