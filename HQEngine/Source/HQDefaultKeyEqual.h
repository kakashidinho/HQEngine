/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_DEFAUL_KEY_EQUAL_H
#define HQ_DEFAUL_KEY_EQUAL_H
#include <string>
#include "HQPrimitiveDataType.h"

template <class T> 
struct HQDefaultKeyEqual
{
	bool operator () (const T & val1 , const T &val2)const  {return val1 == val2;}
};

/*-------string---------------*/
template <class T>
struct HQDefaultStringKeyEqual
{
	bool operator () (const T & val1 , const T &val2)const  {return !val1.compare(val2);}
};

template <>
struct HQDefaultKeyEqual<std::string> : public HQDefaultStringKeyEqual<std::string> {};

#ifndef ANDROID
template <>
struct HQDefaultKeyEqual<std::wstring> : public HQDefaultStringKeyEqual<std::wstring> {};
#endif
#endif
