/*
Copyright (C) 2010-2014  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQEnginePCH.h"
#include "../HQEngineWindow.h"
#include "../HQEngineCommonInternal.h"

#include <string.h>
#include <X11/extensions/XInput2.h>


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

//mouse wheel delta
static const int g_wheelDelta = 120;

/*-------prototypes------------*/
static void ParseRawEvent(const double *input_values, unsigned char *mask, int mask_len,
                            double *output_values, int output_values_len);
static void KeyDownMessage(XKeyEvent *keyEvent);
static void KeyUpMessage(XKeyEvent *keyEvent);
static void RawMotionMessage(XGenericEventCookie* cookie);
static void GetMousePoint(int motionEventX, int motionEventY, HQPointi& pointOut); 
static KeyInfo & GetKeyInfo(XKeyEvent* keyEvent);


/*-----------get mouse point------------*/
void GetMousePoint(int motionEventX, int motionEventY, HQPointi& pointOut)
{
	if (HQEngineApp::GetInstance()->IsMouseCursorEnabled())
	{
		pointOut.x = motionEventX;
		pointOut.y = motionEventY;
	}
	else {//we are hiding the cursor, so reported position should be zero
		pointOut.x = pointOut.y = 0;
	}
}

/*---------------code from SDL---------------*/
static void ParseRawEvent(const double *input_values, unsigned char *mask, int mask_len,
                            double *output_values, int output_values_len) 
{
	const int MAX_AXIS = 16;
	int i = 0,z = 0;
	int top = mask_len * 8;
	if (top > MAX_AXIS)
	top = MAX_AXIS;

	memset(output_values, 0, output_values_len * sizeof(double));
	for (; i < top && z < output_values_len; i++) {
		if (XIMaskIsSet(mask, i)) {
		    const int value = (int) *input_values;
		    output_values[z] = value;
		    input_values++;
		}
		z++;
	}
}

void RawMotionMessage(XGenericEventCookie* cookie)
{
	const XIRawEvent *rawev = (const XIRawEvent*)cookie->data;
	double relative_cords[2];
	
	//get relative mouse motion
	ParseRawEvent(rawev->raw_values, rawev->valuators.mask,
                      rawev->valuators.mask_len, relative_cords, 2);	

	
	//now notify listener
	HQPointi point = {(int)relative_cords[0], (int)relative_cords[1]};
	HQEngineApp::GetInstance()->GetMouseListener()->MouseMove(point);
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
  m_inviCursor(0),
  m_xinputSupported(false),
  m_needKeyboardGrabbed(false),
  m_needMouseGrabbed(false),
  m_windowCloseMsg(None)
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

	//disable mouse cursor if needed
	if (HQEngineApp::GetInstance()->IsMouseCursorEnabled() == false)
		this->EnableCursor(false);
	
	//init xinput
	this->InitXinput();
}

HQEngineWindow::~HQEngineWindow()
{
	delete[] m_title;
	if (m_inviCursor)
	{
		XFreeCursor(m_display, m_inviCursor);
	}

	if (m_ownDisplay)//close display if we open it ourself
		XCloseDisplay(m_display);
}

void HQEngineWindow::InitXinput(){
	/*
	Note: code from SDL
	*/
	int event, err;
	int major = 2, minor = 0;
	int outmajor,outminor;

	if (!XQueryExtension(GetDisplay(), "XInputExtension", &m_xinput2Opcode, &event, &err))
		return;
	outmajor = major;
	outminor = minor;
	if (XIQueryVersion(GetDisplay(), &outmajor, &outminor) != Success) {
		return;
	}

	/*Check supported version*/
	if(outmajor * 1000 + outminor < major * 1000 + minor) {
		/*X server does not support the version we want*/
		return;
	}
	

	m_xinputSupported = true;
}

HQReturnVal HQEngineWindow::Show()
{	
	//create "window close" atom
	if (m_windowCloseMsg == None)
	{
		m_windowCloseMsg = XInternAtom(GetDisplay(), "WM_DELETE_WINDOW", False);
   		XSetWMProtocols(GetDisplay(), GetRawWindow(), &m_windowCloseMsg, 1);
	}
	
	//map window
	if (HQEngineApp::GetInstance()->GetRenderDevice()->IsWindowed())
		XMapWindow(GetDisplay(), GetRawWindow());
	else
	{
		XMapRaised(GetDisplay(), GetRawWindow());
		
		XGrabKeyboard(GetDisplay(), GetRawWindow(), True, GrabModeAsync,
		    GrabModeAsync, CurrentTime);
		XGrabPointer(GetDisplay(), GetRawWindow(), True, ButtonPressMask,
		    GrabModeAsync, GrabModeAsync, GetRawWindow(), None, CurrentTime);

		m_needKeyboardGrabbed = true;
		m_needMouseGrabbed = true;
	}
	return HQ_OK;
}

bool HQEngineWindow::EnableCursor(bool enable){
	
	if (enable == HQEngineApp::GetInstance()->IsMouseCursorEnabled())
		return true;//nothing to do

	return EnableCursorNonCheck(enable);
}

bool HQEngineWindow::EnableCursorNonCheck(bool enable)	{
	if (GetRawWindow() == 0 || !m_xinputSupported)
		return false;
	if (!enable)
	{
		/*--------------------these are SDL code----------------------------------*/
		{
			/*Enable Raw motion events for this display*/
			XIEventMask eventmask;
			unsigned char mask[3] = { 0,0,0 };
			eventmask.deviceid = XIAllMasterDevices;
			eventmask.mask_len = sizeof(mask);
			eventmask.mask = mask;

			XISetMask(mask, XI_RawMotion);
		
			//register raw motion event 
			if (XISelectEvents(GetDisplay(), DefaultRootWindow(GetDisplay()), &eventmask, 1) != Success) {
				return false;
			}
		}
		/*--------------------end SDL code----------------------------------*/

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
		if (HQEngineApp::GetInstance()->GetRenderDevice()->IsWindowed())//fullscreen window already grabs the pointer
		{
			XGrabPointer(GetDisplay(), GetRawWindow(), True, ButtonPressMask,
					GrabModeAsync, GrabModeAsync, GetRawWindow(), None, CurrentTime);//prevent cursor from going outside
			m_needMouseGrabbed = true;
		}
	}
	else
	{
		/*--------------------these are SDL code----------------------------------*/
		{
			/*Disable Raw motion events for this display*/
			XIEventMask eventmask;
			unsigned char mask[3] = { 0,0,0 };
			eventmask.deviceid = XIAllMasterDevices;
			eventmask.mask_len = sizeof(mask);
			eventmask.mask = mask;
		
			//unregister raw motion event 
			if (XISelectEvents(GetDisplay(), DefaultRootWindow(GetDisplay()), &eventmask, 1) != Success) {
				return false;
			}
		}
		/*--------------------end SDL code----------------------------------*/

		XUndefineCursor(GetDisplay(), GetRawWindow());
		if (HQEngineApp::GetInstance()->GetRenderDevice()->IsWindowed())//fullscreen window need mouse grabbed
		{
			XUngrabPointer(GetDisplay(), CurrentTime);
			m_needMouseGrabbed = false;
		}
	}

	return true;//TO DO: error handling
}


/*-------this's event handler--------------*/
void HQEngineWindow::HandleEvent(XEvent * event)
{
	switch (event->type) 
	{
	case GenericEvent://may be mouse's raw motion
		{
			XGenericEventCookie *cookie = (XGenericEventCookie*)event;
			if (XGetEventData(this->GetDisplay(), cookie)) {
				if(cookie->extension == m_xinput2Opcode)
				{
					switch (cookie->evtype)
					{
						case XI_RawMotion:
							RawMotionMessage(cookie);
						break;
					}//switch(cookie->evtype)
				}//if(cookie->extension != xinput2_opcode)
				XFreeEventData(this->GetDisplay(), cookie);
			}
		}
		break;
	case MotionNotify://mouse's absolute movement
		if (HQEngineApp::GetInstance()->IsMouseCursorEnabled())//only report absolute position if mouse's cursor is enabled
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
	case ConfigureNotify:
		HQEngineApp::GetInstance()->GetRenderDevice()->OnWindowSizeChanged(event->xconfigure.width, event->xconfigure.height);
		break;
	case FocusIn:
		//printf("focus in\n");
		if (this->NeedKeyboardGrabbed())
			XGrabKeyboard(this->GetDisplay(), this->GetRawWindow(), True, GrabModeAsync,
				    GrabModeAsync, CurrentTime);
		if (this->NeedMouseGrabbed())
			XGrabPointer(this->GetDisplay(), this->GetRawWindow(), True, ButtonPressMask,
				    GrabModeAsync, GrabModeAsync, this->GetRawWindow(), None, CurrentTime);
		if (!HQEngineApp::GetInstance()->IsMouseCursorEnabled())
		{
			//re-disable cursor
			this->EnableCursorNonCheck(false);
		}
		break;
	case FocusOut:
		//printf("focus out\n");
		if (this->NeedKeyboardGrabbed())
			XUngrabKeyboard(this->GetDisplay(), CurrentTime);
		if (this->NeedMouseGrabbed())
			XUngrabPointer(this->GetDisplay(), CurrentTime);
		if (!HQEngineApp::GetInstance()->IsMouseCursorEnabled())
		{
			//temporarily re-enable cursor
			/*--------------------these are SDL code----------------------------------*/
			{
				/*Disable Raw motion events for this display*/
				XIEventMask eventmask;
				unsigned char mask[3] = { 0,0,0 };
				eventmask.deviceid = XIAllMasterDevices;
				eventmask.mask_len = sizeof(mask);
				eventmask.mask = mask;

				//unregister raw motion event
				if (XISelectEvents(this->GetDisplay(), DefaultRootWindow(this->GetDisplay()), &eventmask, 1) != Success) {
					//what to do
				}
			}
			/*--------------------end SDL code----------------------------------*/

			XUndefineCursor(this->GetDisplay(), this->GetRawWindow());
		}
		break;
	case ClientMessage:
		if (event->xclient.data.l[0] == m_windowCloseMsg)//close button is clicked
		{
			if(HQEngineApp::GetInstance()->GetWindowListener()->WindowClosing() == true)//if listener agree to close this this
			{
				HQEngineApp::GetInstance()->Stop();
				HQEngineApp::GetInstance()->DestroyWindow();
				HQEngineApp::GetInstance()->GetWindowListener()->WindowClosed();
				HQEngineApp::GetInstance()->GetAppListener()->OnDestroy();
				exit(0);
			}
		}
		break;
	case DestroyNotify:
		HQEngineApp::GetInstance()->GetWindowListener()->WindowClosed();
		break;

	}
}
