/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_ENGINEAPP_DELEGATE_H
#define HQ_ENGINEAPP_DELEGATE_H
#include "HQTimer.h"
#include "HQKeyCode.h"
#include "HQUtilMath.h"
///
///application rendering delegate
///
class HQEngineRenderDelegate
{
public:
	///
	///render current frame. 
	///{dt} is elapsed time in seconds since last frame. 
	///note: HQRenderDevice::DisplayBackBuffer() will be called after this method is called.
	///
	virtual void Render(HQTime dt) = 0;

	///
	///called when render device restore from "lost" state. All vertex & index buffers, render targets' datas need to be reloaded. 
	///In android, shaders and textures need to be recreated too
	///
	virtual void OnResetDevice() {}
	///
	///called when render device begins to be in "lost" state. All vertex & index buffers, render targets' datas need to be reloaded 
	///after device restore. 
	///In android, shaders and textures need to be recreated after device restore too
	///
	virtual void OnLostDevice() {}
};

///
///window listener
///
class HQEngineWindowListener
{
public:
	///
	///application's window is about to be closed
	///and application's loop is about to be stopped. 
	///if this method return false, window will not be closed 
	///and application's loop also will not be stopped. this method is called on main thread
	///
	virtual bool WindowClosing() { return true; };
	///
	///application 's window is closed. this method is called on main thread
	///
	virtual void WindowClosed() {}
};

typedef HQKeyCode::MouseButtonEnum HQMouseKeyCodeType;

class HQEngineMouseListener
{
public:
	///
	///If mouse cursor is disable, {point} indicates position relative to the last position. 
	///else, it indicates position relative to upper left corner of window's rendering area
	///
	virtual void MousePressed( HQMouseKeyCodeType button , const HQPointi &point) {}
	///
	///If mouse cursor is disable, {point} indicates position relative to the last position. 
	///else, it indicates position relative to upper left corner of window's rendering area
	///
	virtual void MouseReleased( HQMouseKeyCodeType button , const HQPointi &point) {}
	///
	///If mouse cursor is disable, {point} indicates position relative to the last position. 
	///else, it indicates position relative to upper left corner of window's rendering area
	///
	virtual void MouseMove( const HQPointi &point) {}
	///
	///{delta} > 0 if wheel was rotated up
	///If mouse cursor is disable, {point} indicates position relative to the last position. 
	///else, it indicates position relative to upper left corner of window's rendering area
	///
	virtual void MouseWheel( hq_float32 delta, const HQPointi &point) {}
};


class HQEngineAppListener
{
public:
	///application will be terminated. This method is called in main thread instead of game thread like other delegate's methods
	virtual void OnDestroy() {}
	///application is paused. This method is called in main thread instead of game thread like other delegate's methods
	virtual void OnPause() {}
	///application is resumed. This method is called in main thread instead of game thread like other delegate's methods
	virtual void OnResume() {}

	///
	///back button is pressed. This method is called in main thread instead of game thread like other delegate's methods. 
	///return false if you don't want app to exit
	///
	virtual bool BackButtonPressed() {return true;}
};

class HQTouchEvent
{
public:
	virtual hquint32 GetNumTouches() const = 0;
	virtual hqint32 GetTouchID(hquint32 index) const = 0;
	virtual const HQPointf & GetPosition(hquint32 index) const = 0;
	virtual const HQPointf & GetPrevPosition(hquint32 index) const = 0;
};


class HQEngineMotionListener{
public:
	virtual void TouchBegan(const HQTouchEvent &event) {}
	virtual void TouchMoved(const HQTouchEvent &event) {}
	virtual void TouchEnded(const HQTouchEvent &event) {}
	virtual void TouchCancelled(const HQTouchEvent &event) {}
};

class HQEngineOrientationListener
{
public:
	virtual void ChangedToPortrait() {}
	virtual void ChangedToPortraitUpsideDown() {}
	///right side is at top
	virtual void ChangedToLandscapeLeft() {}
	///left side is at top
	virtual void ChangedToLandscapeRight() {}
};

///
///keyboard listener
///
typedef hq_uint32 HQKeyCodeType;

class HQEngineKeyListener
{
public:

	///
	///event when a key is pressed. 
	///{keyCode} can be one of values in HQKeyCode:KeyCodeEnum
	///
	virtual void KeyPressed(HQKeyCodeType keyCode) {}
	///
	///event when a key is released
	///{keyCode} can be one of values in HQKeyCode:KeyCodeEnum
	///
	virtual void KeyReleased(HQKeyCodeType keyCode) {}
};

#endif
