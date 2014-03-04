/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_PAIR_H
#define HQ_PAIR_H


template <class T1 , class T2>
struct HQPair
{
	HQPair() {}
	HQPair(T1 first , T2 second ) : m_first(first) , m_second(second) {}
	T1 m_first;
	T2 m_second;
};

#endif
