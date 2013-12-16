/*********************************************************************
*Copyright 2010 Le Hoang Quyen. All rights reserved.
*********************************************************************/
#ifndef _2DMATH_INL_
#define _2DMATH_INL_

template <typename T>
inline bool HQRect<T>::IsPointInside(T _x, T _y) const
{
	if(_x < this->x || _x > (this->x + this->w))
		return false;
	if(_y < this->y || _y > (this->y + this->h))
		return false;
	return true;
}

template <typename T>
inline bool HQRect<T>::IsPointInside(const HQPoint<T> &point) const
{
	if(point.x < this->x || point.x > (this->x + this->w))
		return false;
	if(point.y < this->y || point.y > (this->y + this->h))
		return false;
	return true;
}
#endif