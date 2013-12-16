/*********************************************************************
*Copyright 2011 Le Hoang Quyen. All rights reserved.
*********************************************************************/
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