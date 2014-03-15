/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQEnginePCH.h"
#include "../HQEngineWindow.h"
#include "../HQEngineCommonInternal.h"
#include "string.h"


struct KeyInfo{
	KeyInfo():
		pressed(false)
	{
	}
	
	KeyInfo(const KeyInfo& src)
		: pressed(src.pressed),
		  unmodSym(src.unmodSym)
	{
	}

	KeyInfo & operator = (const KeyInfo& src){
		this->pressed = src.pressed;
		this->unmodSym = src.unmodSym;
		return *this;
	}


	bool pressed;
	KeySym unmodSym;//unmodified keysym
};
//for detecting repeated key presses
static HQEnginePrimeHashTable<KeyCode, KeyInfo> g_keyTable;

//for storing old mouse position
static HQPointi g_oldMousePos = {-90000, -90000};

static const int g_wheelDelta = 120;

static HQEngineWindow* g_window = NULL;
/*-------prototypes------------*/
static void KeyDownMessage(XKeyEvent *keyEvent);
static void KeyUpMessage(XKeyEvent *keyEvent);
static void GetMousePoint(int motionEventX, int motionEventY, HQPointi& pointOut); 
static KeyInfo & GetKeyInfo(XKeyEvent* keyEvent);

/*-------window event handler--------------*/
void HQEngineWndEventHandler(XEvent * event)
{
	switch (event->type) 
	{
	case MotionNotify://mouse move
		{
			HQPointi point;
			GetMousePoint(event->xmotion.x, event->xmotion.y, point);			

			HQEngineApp::GetInstance()->GetMouseListener()->MouseMove(point);
		}
		break;
	case ButtonPress:
		{
			HQPointi point;	
			GetMousePoint(event->xbutton.x, event->xbutton.y, point);
			
			switch(event->xbutton.button)
			{
			case Button1://left mouse
				HQEngineApp::GetInstance()->GetMouseListener()->MousePressed(HQKeyCode::LBUTTON, point);
				break;
			case Button2://middle mouse
				HQEngineApp::GetInstance()->GetMouseListener()->MousePressed(HQKeyCode::MBUTTON, point);
				break;
			case Button3://right mouse
				HQEngineApp::GetInstance()->GetMouseListener()->MousePressed(HQKeyCode::RBUTTON, point);
				break;
			case Button4://wheel up
				HQEngineApp::GetInstance()->GetMouseListener()->MouseWheel(
					g_wheelDelta , point
					);
				break;
			case Button5://wheel down
				HQEngineApp::GetInstance()->GetMouseListener()->MouseWheel(
					-g_wheelDelta , point
					);
				break;
			}//switch(event->xbutton.button)
		}
		break;
	case ButtonRelease:
		{
			HQPointi point;	
			GetMousePoint(event->xbutton.x, event->xbutton.y, point);
			switch(event->xbutton.button)
			{
			case Button1://left mouse
				HQEngineApp::GetInstance()->GetMouseListener()->MouseReleased(HQKeyCode::LBUTTON, point);
				break;
			case Button2://middle mouse
				HQEngineApp::GetInstance()->GetMouseListener()->MouseReleased(HQKeyCode::MBUTTON, point);
				break;
			case Button3://right mouse
				HQEngineApp::GetInstance()->GetMouseListener()->MouseReleased(HQKeyCode::RBUTTON, point);
				break;
			}//switch(event->xbutton.button)
		}
		break;
	case KeyPress: 
		KeyDownMessage(&event->xkey);
		break;
	case KeyRelease: 
		KeyUpMessage(&event->xkey);
		break;
	/* TO DO:
	case WM_CLOSE:
		if(HQEngineApp::GetInstance()->GetWindowListener()->WindowClosing() == false)
			return 0;
		HQEngineApp::GetInstance()->Stop();
		break;
	*/
	case ResizeRequest:
		//invalidate old pointer position
		g_oldMousePos.x = g_oldMousePos.y = -90000;
		break;
	case DestroyNotify:
		HQEngineApp::GetInstance()->GetWindowListener()->WindowClosed();
		break;

	}
}

/*-----------get mouse point------------*/
void GetMousePoint(int motionEventX, int motionEventY, HQPointi& pointOut)
{
	if (HQEngineApp::GetInstance()->IsMouseCursorEnabled())
	{
		pointOut.x = motionEventX;
		pointOut.y = motionEventY;
	}
	else {//getting delta info
		if (g_oldMousePos.x == -90000)
		{
			pointOut.x = pointOut.y = 0;
		}
		else
		{
			pointOut.x = motionEventX - g_oldMousePos.x;
			pointOut.y = motionEventY - g_oldMousePos.y;
		}
	}
	
	g_oldMousePos.x = motionEventX;
	g_oldMousePos.y = motionEventY;
}

KeyInfo & GetKeyInfo(XKeyEvent* keyEvent)
{
	bool found = false;
	KeyInfo& info = g_keyTable.GetItem(keyEvent->keycode, found);
	if (!found)
	{
		KeyInfo newInfo;
		newInfo.unmodSym = XLookupKeysym(keyEvent, 0);

		g_keyTable.Add(keyEvent->keycode, newInfo);

		return g_keyTable.GetItem(keyEvent->keycode, found);
	}
	else
		return info;
}

void KeyDownMessage(XKeyEvent *event)
{
	KeyInfo & kinfo = GetKeyInfo(event);
	if (kinfo.pressed)
		return;//ignore repeated key press event

	kinfo.pressed = true;//mark this key as pressed

	HQEngineApp::GetInstance()->GetKeyListener()->KeyPressed(kinfo.unmodSym);
}

void KeyUpMessage(XKeyEvent *event)
{
	KeyInfo & kinfo = GetKeyInfo(event);
	kinfo.pressed = false;//mark this key as released
	
	HQEngineApp::GetInstance()->GetKeyListener()->KeyReleased(kinfo.unmodSym);
}

/*-------engine 's window class--------*/
HQEngineWindow::HQEngineWindow(const char *title, const char *settingFileDir ,  HQWIPPlatformSpecificType* args)
: HQEngineBaseWindow(settingFileDir),
  m_inviCursor(0)
{
	if (args == NULL || args->display == NULL)
	{
		//create default display
		this->m_display = XOpenDisplay(NULL);
		this->m_ownDisplay = true;
	}
	else
	{
		this->m_display = args->display;
		this->m_ownDisplay = false;
	}	

	if (this->m_display == NULL)
		throw std::bad_alloc();

	/*---------default settings ---*/
	m_windowInfo.x = m_windowInfo.y = 0;
	m_windowInfo.parent = DefaultRootWindow(m_display);//no parent
	m_windowInfo.window = 0;

	/*--------copy title----------*/
	size_t len = strlen(title);
	m_title = HQ_NEW char [len + 1];
	m_title[len] = '\0';
	for (size_t i = 0; i < len ; ++i)
		m_title[i] = title[i];

	m_windowInfo.title = m_title;	

	g_keyTable.RemoveAll();//reset key table

	g_window = this;//store global reference to this window


	//disable mouse cursor if needed
	if (HQEngineApp::GetInstance()->IsMouseCursorEnabled() == false)
		this->EnableCursor(false);
}

HQEngineWindow::~HQEngineWindow()
{
	delete[] m_title;
	if (m_inviCursor)
	{
		XFreeCursor(m_display, m_inviCursor);
	}

	if (m_ownDisplay)
		XCloseDisplay(m_display);

	g_window  = NULL;
}

HQReturnVal HQEngineWindow::Show()
{
	XMapWindow(GetDisplay(), GetRawWindow());
	return HQ_OK;
}

bool HQEngineWindow::EnableCursor(bool enable){
	
	if (GetRawWindow() == 0)
		return false; 
	if (!enable)
	{
		if (m_inviCursor == 0)
		{
			/*-------create invisible cursor ----*/
			XColor black;
			const char noData[] = { 0,0,0,0,0,0,0,0 };
			black.red = black.green = black.blue = 0;

			Pixmap bitmapNoData = XCreateBitmapFromData(GetDisplay(), GetRawWindow(), noData, 8, 8);
			m_inviCursor = XCreatePixmapCursor(GetDisplay(), bitmapNoData, bitmapNoData, 
						             &black, &black, 0, 0);
		}
		
		XDefineCursor(GetDisplay(), GetRawWindow(), m_inviCursor);
	}
	else
	{
		XUndefineCursor(GetDisplay(), GetRawWindow());
	}

	return true;//TO DO: error handling
}
