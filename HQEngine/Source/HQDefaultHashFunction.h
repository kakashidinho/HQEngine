/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_DEFAULT_HASH_FUNC_H
#define HQ_DEFAULT_HASH_FUNC_H

#include "HQPrimitiveDataType.h"
#include <string>

template <class T> struct HQDefaultHashFunc
{
	hq_uint32 operator() (const T& x) const {return 0x0;}//default just return 0
};
/*------numeric data type---------*/

template <class T> struct HQDefaultNumericHashFunc
{
	hq_uint32 operator() (const T& x) const {return (hq_uint32)x;}
};

template <> struct HQDefaultHashFunc<hq_int64> : public HQDefaultNumericHashFunc<hq_int64> {};
template <> struct HQDefaultHashFunc<hq_uint64> : public HQDefaultNumericHashFunc<hq_uint64> {};
template <> struct HQDefaultHashFunc<long> : public HQDefaultNumericHashFunc<long> {};
template <> struct HQDefaultHashFunc<unsigned long> : public HQDefaultNumericHashFunc<unsigned long> {};
template <> struct HQDefaultHashFunc<hq_int32> : public HQDefaultNumericHashFunc<hq_int32> {};
template <> struct HQDefaultHashFunc<hq_uint32> : public HQDefaultNumericHashFunc<hq_uint32> {};
template <> struct HQDefaultHashFunc<hq_short16> : public HQDefaultNumericHashFunc<hq_short16> {};
template <> struct HQDefaultHashFunc<hq_ushort16> : public HQDefaultNumericHashFunc<hq_ushort16> {};
template <> struct HQDefaultHashFunc<char> : public HQDefaultNumericHashFunc<char> {};
template <> struct HQDefaultHashFunc<unsigned char> : public HQDefaultNumericHashFunc<unsigned char> {};

/*--------boolean---------------*/
template <> struct HQDefaultHashFunc<bool>
{
	hq_uint32 operator() (const bool& x) const {return x ? 1 : 0;}
};

/*-------floating point---------*/
template <> struct HQDefaultHashFunc<hq_float32>
{
	hq_uint32 operator() (const hq_float32& x) const {return *((hq_uint32*)&x);}
};
template <> struct HQDefaultHashFunc<hq_float64>
{
	hq_uint32 operator() (const hq_float64& x) const {return (hq_uint32)(*((hq_uint64*)&x));}
};

/*--------string----------------*/
template <class charType , class stringClass> struct HQDefaultStringHashFunc
{
	hq_uint32 operator() (const charType* x) const 
	{
		if (x == NULL || x[0] == 0)
			return 0;
		hq_uint32 hashCode = (hq_uint32)x[0];

		for (size_t i = 1 ; x[i] != 0 ; ++i)
			hashCode = 31 * hashCode + (hq_uint32)x[i];
		return hashCode;
	}
	hq_uint32 operator() (const stringClass& x) const 
	{
		size_t length = x.size();
		if (length == 0)
			return 0x0;
		hq_uint32 hashCode = (hq_uint32)x[0];
		for (size_t i = 1 ; i < length ; ++i)
			hashCode = 31 * hashCode + (hq_uint32)x[i];
		return hashCode;
	}
};

template <> struct HQDefaultHashFunc<char*> : public HQDefaultStringHashFunc<char , std::string> {};
template <> struct HQDefaultHashFunc<unsigned char*> : public HQDefaultStringHashFunc<unsigned char, std::string> {};
#ifndef ANDROID
template <> struct HQDefaultHashFunc<wchar_t*> : public HQDefaultStringHashFunc<wchar_t , std::wstring> {};
#endif
template <> struct HQDefaultHashFunc<std::string> : public HQDefaultStringHashFunc<char , std::string> {};
#ifndef ANDROID
template <> struct HQDefaultHashFunc<std::wstring> : public HQDefaultStringHashFunc<wchar_t , std::wstring> {};
#endif
#endif
