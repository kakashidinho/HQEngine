/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_MISC_DATA_TYPE_H
#define HQ_MISC_DATA_TYPE_H

#include <string.h>
#include "HQPrimitiveDataType.h"

typedef hqfloat32 (&hq3FloatArrayRef) [3];
typedef const hqfloat32 (&hqConst3FloatArrayRef) [3];

struct HQFloat3
{
	void Duplicate(hq_float32 duplicatedVal)
	{
		x = y = z = duplicatedVal;
	}

	void Set(hq_float32 _x , hq_float32 _y , hq_float32 _z)
	{
		x = _x ; 
		y = _y ; 
		z = _z ;
	}
	
	operator hq_float32 * () {return f;}
	operator const hq_float32 * () const {return f;}
	operator hq3FloatArrayRef () {return f;}
	operator hqConst3FloatArrayRef () const {return f;}
	
	hqfloat32& operator[] (size_t index) {return f[index];}
	const hqfloat32& operator[] (size_t index) const{return f[index];}

	union{
		struct{
			hq_float32 x , y , z;
		};
		hq_float32 f[3];
	};
};

typedef hqfloat32 (&hq4FloatArrayRef) [4];
typedef const hqfloat32 (&hqConst4FloatArrayRef) [4];

struct HQFloat4 
{
	void Duplicate(hq_float32 duplicatedVal)
	{
		x = y = z = w = duplicatedVal;
	}

	void Set(hq_float32 _x , hq_float32 _y , hq_float32 _z)
	{
		x = _x ; 
		y = _y ; 
		z = _z ;
	}

	void Set(hq_float32 _x , hq_float32 _y , hq_float32 _z, hq_float32 _w)
	{
		x = _x ; 
		y = _y ; 
		z = _z ;
		w = _w;
	}
	
	operator hq_float32 * () {return f;}
	operator const hq_float32 * () const {return f;}
	operator hq4FloatArrayRef () {return f;}
	operator hqConst4FloatArrayRef () const {return f;}
	
	hqfloat32& operator[] (size_t index) {return f[index];}
	const hqfloat32& operator[] (size_t index) const{return f[index];}

	union{
		struct{
			hq_float32 x , y , z, w;
		};
		hq_float32 f[4];
	};

};

#endif
