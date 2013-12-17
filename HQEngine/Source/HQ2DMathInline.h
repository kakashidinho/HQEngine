/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

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
