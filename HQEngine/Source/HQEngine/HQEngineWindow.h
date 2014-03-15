/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_ENGINE_WINDOW_H
#define HQ_ENGINE_WINDOW_H
#include "../HQEngineApp.h"


class HQEngineBaseWindow
{
public:
	HQEngineBaseWindow(const char* settingFileDir);
	
	///get initial width
	hquint32 GetWidth() {return m_width;}
	///get initial height
	hquint32 GetHeight() {return m_height;}
	///get initial windowed status
	bool IsWindowed() {return m_windowed;}

private :
	hquint32 m_width , m_height;
	bool m_windowed;
};

/*-----win32------*/
#if defined HQ_WIN_DESKTOP_PLATFORM
class HQEngineWindow : public HQEngineBaseWindow
{
public:
	HQEngineWindow(const char *title, const char* settingFileDir , HQWIPPlatformSpecificType* icons);
	~HQEngineWindow();
	
	HQRenderDeviceInitInput GetRenderDeviceInitInput() {return m_window;}
	HQReturnVal Show();

private:
	HWND m_window;
	wchar_t *m_title;
};

/*-----win rt-----------*/
#elif defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM
#include "HQEventSeparateThread.h"

#include "../HQPair.h"
#include "../HQLinkedList.h"
#include "../HQMemoryManager.h"

using namespace Windows::UI::Core;
using namespace Windows::Devices::Input;

/*-------window event handlers--------------*/
ref class HQWindowsEventHandler sealed
{
public:
	HQWindowsEventHandler(Windows::UI::Core::CoreCursor ^ cursor);
	
	void AttachToWindow();
	void DetachFromWindow();

	void HideMouse(bool hide);

	void OnKeyPressed(CoreWindow^ sender, KeyEventArgs^ args);

	void OnKeyReleased(CoreWindow^ sender, KeyEventArgs^ args);

	void OnPointerPressed(CoreWindow^ sender, PointerEventArgs^ args);

	void OnPointerReleased(CoreWindow^ sender, PointerEventArgs^ args);

	void OnPointerMoved(CoreWindow^ sender, PointerEventArgs^ args);

	void OnPointerWheelChanged(CoreWindow^ sender, PointerEventArgs^ args);

	void OnMouseMoved(MouseDevice^ device, MouseEventArgs^ args);

	void OnClosed(CoreWindow^ sender, CoreWindowEventArgs^ args);

private:
	typedef HQLinkedList<HQPair<unsigned int, HQPointf>, HQPoolMemoryManager> TouchHistoryTableType;

	void OnTouchEvent(HQEventType type, Windows::UI::Input::PointerPoint^ pointer);
	void AttachToWindowOnUIThread();
	void DetachFromWindowOnUIThread();
	void HideMouseOnUITHread(bool hide);

	bool m_mouseHide;
	Platform::Agile<Windows::UI::Core::CoreCursor> m_currentCursor;

	bool m_leftMousePressed;
	bool m_rightMousePressed;
	bool m_middleMousePressed;

	TouchHistoryTableType m_prevTouchPos;//holding the previous positions of the known touches

	Windows::Foundation::EventRegistrationToken m_OnKeyPressedToken;
	Windows::Foundation::EventRegistrationToken m_OnKeyReleasedToken;
	Windows::Foundation::EventRegistrationToken m_OnPointerPressedToken;
	Windows::Foundation::EventRegistrationToken m_OnPointerReleasedToken;
	Windows::Foundation::EventRegistrationToken m_OnPointerMovedToken;
	Windows::Foundation::EventRegistrationToken m_OnPointerWheelChangedToken;
	Windows::Foundation::EventRegistrationToken m_OnMouseMovedToken;
	Windows::Foundation::EventRegistrationToken m_OnClosedToken;
};

class HQEngineWindow : public HQEngineBaseWindow
{
public:
	HQEngineWindow(const char *title, const char* settingFileDir , HQWIPPlatformSpecificType* arg);
	~HQEngineWindow();
	
	HQRenderDeviceInitInput GetRenderDeviceInitInput() {return m_window;}
	HQWindowsEventHandler ^ GetWindowEventHandler() {return m_windowEventHandler;}
	HQReturnVal Show();

private:
	void ChangeCursorOnUIThread(Platform::Agile<Windows::UI::Core::CoreCursor> cursor);

	HQRenderDeviceInitInput m_window;
	HQWindowsEventHandler ^ m_windowEventHandler;
};

/*----mac osx----*/
#elif defined HQ_MAC_PLATFORM
class HQEngineWindow : public HQEngineBaseWindow
{
public:
	HQEngineWindow(const char *title, const char* settingFileDir , HQWIPPlatformSpecificType* icon);
	~HQEngineWindow();
	
	HQRenderDeviceInitInput GetRenderDeviceInitInput() {return &m_viewInfo;}
	HQReturnVal Show();
	
private:
	NSWindow *m_window;
	
	HQOSXRenderDeviceInitInput m_viewInfo;
	
};


/*------IOS-----*/
#elif defined HQ_IPHONE_PLATFORM
class HQEngineWindow : public HQEngineBaseWindow
{
public:
	HQEngineWindow(const char *title, const char* settingFileDir , HQWIPPlatformSpecificType* additionalSettings);
	~HQEngineWindow();
	
	HQRenderDeviceInitInput GetRenderDeviceInitInput();
	HQReturnVal Show();
	
private:
	UIViewController *m_viewController;
	UIView *m_view;
	HQIOSRenderDeviceInitInput m_renderDeviceInitInfo;
	
};

/*---------android----------*/
#elif defined HQ_ANDROID_PLATFORM

#include <j2cpp/j2cpp.hpp>
#include <android/view/View.hpp>

using namespace j2cpp;

class HQEngineWindow : public HQEngineBaseWindow
{
public:
	HQEngineWindow(const char *title, const char* settingFileDir , HQWIPPlatformSpecificType* additionalSetting);
	~HQEngineWindow();
	
	HQRenderDeviceInitInput GetRenderDeviceInitInput();
	HQReturnVal Show();

private:
	android::view::View m_view;
	HQAndroidRenderDeviceInitInput m_renderDeviceInitInfo;
};

/*---------------linux----------------------*/
#elif defined HQ_LINUX_PLATFORM

class HQEngineWindow : public HQEngineBaseWindow
{
public:
	HQEngineWindow(const char *title, const char* settingFileDir , HQWIPPlatformSpecificType* args);
	~HQEngineWindow();
	
	HQRenderDeviceInitInput GetRenderDeviceInitInput() {return &m_windowInfo;}
	Window GetRawWindow() {return m_windowInfo.window;}
	Display * GetDisplay() {return m_display;}
	HQReturnVal Show();
	
	bool EnableCursor(bool enable);
private:
	HQXWindowInfo m_windowInfo;
	Display * m_display;
	bool m_ownDisplay;
	Cursor m_inviCursor;
	char *m_title;
};

#else
#	error need implement
#endif

#endif
