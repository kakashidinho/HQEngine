/*********************************************************************
*Copyright 2011 Le Hoang Quyen. All rights reserved.
*********************************************************************/
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