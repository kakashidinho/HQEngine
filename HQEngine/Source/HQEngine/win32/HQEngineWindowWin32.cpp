/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQEnginePCH.h"
#include "../HQEngineWindow.h"
#include "string.h"

#include <iostream>
#if _MSC_VER < 1700
#include <hash_map>
#define hash_map_type stdext::hash_map
#else
#include <unordered_map>
#define hash_map_type std::unordered_map
#endif

#define USE_KEY_HOOK 0
#define HQ_WINDOW_STYLE WS_OVERLAPPEDWINDOW & (~(WS_MAXIMIZEBOX | WS_THICKFRAME))

extern HINSTANCE ge_module;

struct KeyPressedFlag{
	KeyPressedFlag():
		pressed(false)
	{
	}
	KeyPressedFlag(bool _pressed)
		: pressed(_pressed)
	{
	}

	operator bool() const {return pressed;}
	KeyPressedFlag & operator = (KeyPressedFlag& _flag){
		this->pressed = _flag.pressed;
		return *this;
	}

	KeyPressedFlag & operator = (bool _pressed){
		this->pressed = _pressed;
		return *this;
	}

	bool pressed;
};
//for detecting repeat key press
static hash_map_type<UINT, KeyPressedFlag> g_keyPressed;
static unsigned char g_mouseInputBuffer[40];

#if USE_KEY_HOOK
static HHOOK g_keyboardHook = NULL;//for disable window key
static bool g_windowActive;//for disable window key
#endif
static STICKYKEYS g_oldStickyKeys = {sizeof(STICKYKEYS), 0};//for disable sticky shortcuy key
static TOGGLEKEYS g_oldToggleKeys = {sizeof(TOGGLEKEYS), 0};//for disable toggle shortcuy key
static FILTERKEYS g_oldFilterKeys = {sizeof(FILTERKEYS), 0};//for disable filter shortcuy key  

/*-------function prototypes-----------*/
void RawInputMessage(WPARAM wParam, LPARAM lParam);
bool KeyDownMessage(WPARAM wParam, LPARAM lParam);
bool KeyUpMessage(WPARAM wParam, LPARAM lParam);
LRESULT CALLBACK LowLevelKeyboardProc( int nCode, WPARAM wParam, LPARAM lParam );//low level keyboard hook


/*-------window procedure--------------*/
LRESULT CALLBACK WndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	switch (message) 
	{
	case WM_ACTIVATEAPP:
        // g_windowActive is used to control if the Windows key is filtered by the keyboard hook or not.
        if( wParam == TRUE )
		{
#if USE_KEY_HOOK
			//re-register keyboard hook
			if (g_keyboardHook == NULL)
				g_keyboardHook = SetWindowsHookEx( WH_KEYBOARD_LL,  LowLevelKeyboardProc, ge_module, 0 );
			

			g_windowActive  = true; 
#endif
		}
        else 
		{
#if USE_KEY_HOOK
			if (g_keyboardHook != NULL)
			{
				UnhookWindowsHookEx(g_keyboardHook);//unregister keyboard hook
				g_keyboardHook = NULL;
			}
			g_windowActive  = false;   
#endif
		}
        break;
	case WM_INPUT://raw input
		RawInputMessage(wParam , lParam);
		break;
	case WM_MOUSEMOVE://mouse move
		{
			HQPointi point = {
				lParam & 0xffff,
				(lParam & 0xffff0000) >> 16
			};
			
			HQEngineApp::GetInstance()->GetMouseListener()->MouseMove(point);
		}
		break;
	case WM_MOUSEWHEEL:
		{
			HQPointi point = {
				lParam & 0xffff,
				(lParam & 0xffff0000) >> 16
			};
			ScreenToClient(hwnd, (POINT*) &point);
			HQEngineApp::GetInstance()->GetMouseListener()->MouseWheel(
				(hqfloat32)GET_WHEEL_DELTA_WPARAM(wParam) , point
					);
		}
		break;
	/*------left button--------*/
	case WM_LBUTTONDOWN://pressed
		{
			HQPointi point = {
				lParam & 0xffff,
				(lParam & 0xffff0000) >> 16
			};
			HQEngineApp::GetInstance()->GetMouseListener()->MousePressed(HQKeyCode::LBUTTON, point);
		}
		break;
	case WM_LBUTTONUP://released
		{
			HQPointi point = {
				lParam & 0xffff,
				(lParam & 0xffff0000) >> 16
			};
			HQEngineApp::GetInstance()->GetMouseListener()->MouseReleased(HQKeyCode::LBUTTON, point);
		}

		break;
	/*------right button--------*/
	case WM_RBUTTONDOWN://pressed
		{
			HQPointi point = {
				lParam & 0xffff,
				(lParam & 0xffff0000) >> 16
			};
			HQEngineApp::GetInstance()->GetMouseListener()->MousePressed(HQKeyCode::RBUTTON, point);
		}
		break;
	case WM_RBUTTONUP://released
		{
			HQPointi point = {
				lParam & 0xffff,
				(lParam & 0xffff0000) >> 16
			};
		
			HQEngineApp::GetInstance()->GetMouseListener()->MouseReleased(HQKeyCode::RBUTTON, point);
		}
		break;
	/*------middle button--------*/
	case WM_MBUTTONDOWN://pressed
		{
			HQPointi point = {
				lParam & 0xffff,
				(lParam & 0xffff0000) >> 16
			};
			HQEngineApp::GetInstance()->GetMouseListener()->MousePressed(HQKeyCode::MBUTTON, point);
		}
		break;
	case WM_MBUTTONUP://released
		{
			HQPointi point = {
				lParam & 0xffff,
				(lParam & 0xffff0000) >> 16
			};
			HQEngineApp::GetInstance()->GetMouseListener()->MouseReleased(HQKeyCode::MBUTTON, point);
		}
		break;
	case WM_MOVE://window move
		if (!HQEngineApp::GetInstance()->IsMouseCursorEnabled())
		{
			RECT rect;
			GetWindowRect(hwnd , &rect);
			ClipCursor(&rect);
		}
		break;
	case WM_ACTIVATE:
		break;
	case WM_SYSKEYDOWN: 
	case WM_KEYDOWN: 
		if(KeyDownMessage(wParam , lParam))
			return 0;
		break;
	case WM_SYSKEYUP: 
	case WM_KEYUP: 
		if (KeyUpMessage(wParam , lParam))
			return 0;
		break;
	case WM_CLOSE:
		if(HQEngineApp::GetInstance()->GetWindowListener()->WindowClosing() == false)
			return 0;
		HQEngineApp::GetInstance()->Stop();
		break;
	case WM_DESTROY:
		HQEngineApp::GetInstance()->GetWindowListener()->WindowClosed();
		break;

	}


	return DefWindowProc(hwnd, message, wParam, lParam);
}

/*--------handle raw input message------------------*/
void RawInputMessage(WPARAM wParam, LPARAM lParam)
{
	if (GET_RAWINPUT_CODE_WPARAM(wParam) == RIM_INPUT)
	{
		unsigned int bufferSize;
		//get buffer size
		GetRawInputData((HRAWINPUT)lParam, RID_INPUT, NULL, &bufferSize, sizeof (RAWINPUTHEADER));

		if (bufferSize > 40)//invalid
			return;//message handling chain will continue
		else
		{
			GetRawInputData((HRAWINPUT)lParam, RID_INPUT, (void*)g_mouseInputBuffer, &bufferSize, sizeof (RAWINPUTHEADER));
			RAWINPUT *rawInput = (RAWINPUT*) g_mouseInputBuffer;
			if (rawInput->header.dwType == RIM_TYPEMOUSE)
			{
				RAWMOUSE & mouseData = rawInput->data.mouse;
				//get mouse listener
				HQEngineMouseListener *listener = HQEngineApp::GetInstance()->GetMouseListener();
				
				HQPointi point = {
					( hq_int32) mouseData.lLastX ,
					( hq_int32) mouseData.lLastY
				};
				
				if (mouseData.usFlags & MOUSE_MOVE_ABSOLUTE)//mouse movement data is based on absolute position
				{
				}
				else//mouse movement data is relative to last point
				{
					if (0 != point.x || 0 != point.y)
						listener->MouseMove(point);
				}
				//left button
				if (mouseData.usButtonFlags & RI_MOUSE_LEFT_BUTTON_DOWN)
					listener->MousePressed(HQKeyCode::LBUTTON, point);
				else if (mouseData.usButtonFlags & RI_MOUSE_LEFT_BUTTON_UP)
					listener->MouseReleased(HQKeyCode::LBUTTON, point);
				
				//right button
				if (mouseData.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_DOWN)
					listener->MousePressed(HQKeyCode::RBUTTON, point);
				else if (mouseData.usButtonFlags & RI_MOUSE_RIGHT_BUTTON_UP)
					listener->MouseReleased(HQKeyCode::RBUTTON, point);
				
				//middle button
				if (mouseData.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_DOWN)
					listener->MousePressed(HQKeyCode::MBUTTON, point);
				else if (mouseData.usButtonFlags & RI_MOUSE_MIDDLE_BUTTON_UP)
					listener->MouseReleased(HQKeyCode::MBUTTON, point);

				//wheel
				if (mouseData.usButtonFlags & RI_MOUSE_WHEEL)
					listener->MouseWheel((hqfloat32)(SHORT)mouseData.usButtonData, point);

			}//if (rawInput->header.dwType == RIM_TYPEMOUSE)
			else if (rawInput->header.dwType == RIM_TYPEKEYBOARD)//keyboard
			{
				RAWKEYBOARD &keyData = rawInput->data.keyboard;
				USHORT scanCode = keyData.MakeCode;
				HQKeyCodeType keyCode = keyData.VKey;

				switch (keyData.VKey)
				{
				case VK_CONTROL:
					keyCode = (scanCode & 0xe000) != 0 ? HQKeyCode::RCONTROL : HQKeyCode::LCONTROL;
					break;
				case VK_MENU:
					keyCode = (scanCode & 0xe000) != 0 ? HQKeyCode::RALT : HQKeyCode::LALT;
					break;
				case VK_SHIFT:
					keyCode = MapVirtualKey(scanCode, MAPVK_VSC_TO_VK_EX);
					break;
				}//switch (keyData.VKey)

				if (keyData.Flags & RI_KEY_BREAK)//key up
				{
					HQEngineApp::GetInstance()->GetKeyListener()->KeyReleased(keyCode);
					g_keyPressed[keyCode] = false;
				}
				else
				{
					if (g_keyPressed[keyCode] == false)
					{
						g_keyPressed[keyCode] = true;
						HQEngineApp::GetInstance()->GetKeyListener()->KeyPressed(keyCode);
					}
				}

			}//else if (rawInput->header.dwType == RIM_TYPEKEYBOARD)
		}//else
	}//if (GET_RAWINPUT_CODE_WPARAM(wParam) == RIM_INPUT)

}
/*--------handle legacy key down message------------------*/
bool KeyDownMessage(WPARAM wParam, LPARAM lParam)
{
	if (lParam & 0x40000000)//this message is repeated
		return false;//message handling chain will continue
	switch (wParam)
	{
	case VK_CONTROL:
		if ((lParam & (0x1 << 24)) == 0)//left
			HQEngineApp::GetInstance()->GetKeyListener()->KeyPressed(HQKeyCode::LCONTROL);
		else//right
			HQEngineApp::GetInstance()->GetKeyListener()->KeyPressed(HQKeyCode::RCONTROL);
		break;
	case VK_MENU:
		if ((lParam & (0x1 << 24)) == 0)//left
			HQEngineApp::GetInstance()->GetKeyListener()->KeyPressed(HQKeyCode::LALT);
		else//right
			HQEngineApp::GetInstance()->GetKeyListener()->KeyPressed(HQKeyCode::RALT);
		return 0;
	case VK_SHIFT:
		{
			UINT vkeyCode = MapVirtualKey((lParam & (0xff << 16)) >> 16 , MAPVK_VSC_TO_VK_EX);
			HQEngineApp::GetInstance()->GetKeyListener()->KeyPressed(vkeyCode);
		}
		break;
	default:
		HQEngineApp::GetInstance()->GetKeyListener()->KeyPressed(wParam);
		if(wParam == VK_F10)
			return true;//message handling chain break at this point
		break;
	}

	return false;//message handling chain will continue
}
/*--------handle legacy key up message------------------*/
bool KeyUpMessage(WPARAM wParam, LPARAM lParam)
{
	switch (wParam)
	{
	case VK_CONTROL:
		if ((lParam & (0x1 << 24)) == 0)//left
			HQEngineApp::GetInstance()->GetKeyListener()->KeyReleased(HQKeyCode::LCONTROL);
		else//right
			HQEngineApp::GetInstance()->GetKeyListener()->KeyReleased(HQKeyCode::RCONTROL);
		break;
	case VK_MENU:
		if ((lParam & (0x1 << 24)) == 0)//left
			HQEngineApp::GetInstance()->GetKeyListener()->KeyReleased(HQKeyCode::LALT);
		else//right
			HQEngineApp::GetInstance()->GetKeyListener()->KeyReleased(HQKeyCode::RALT);
		break;
	case VK_SHIFT:
		{
			UINT vkeyCode = MapVirtualKey((lParam & (0xff << 16)) >> 16 , MAPVK_VSC_TO_VK_EX);
			HQEngineApp::GetInstance()->GetKeyListener()->KeyReleased(vkeyCode);
		}
		break;
	default:
		HQEngineApp::GetInstance()->GetKeyListener()->KeyReleased(wParam);
		break;
	}

	return false;//message handling chain will continue
}

#if USE_KEY_HOOK
/*-----low level keyboard hook----------------------*/
LRESULT CALLBACK LowLevelKeyboardProc( int nCode, WPARAM wParam, LPARAM lParam )
{
    if (nCode < 0 || nCode != HC_ACTION )  // do not process message 
        return CallNextHookEx( g_keyboardHook, nCode, wParam, lParam); 
 
    bool eatKeystroke = false;
    KBDLLHOOKSTRUCT* p = (KBDLLHOOKSTRUCT*)lParam;
    switch (wParam) 
    {
        case WM_KEYDOWN:  
        case WM_KEYUP:    
        {
            eatKeystroke = (g_windowActive && ((p->vkCode == VK_LWIN) || (p->vkCode == VK_RWIN)));
            break;
        }
    }
 
    if( eatKeystroke )
        return 1;
    else
        return CallNextHookEx( g_keyboardHook, nCode, wParam, lParam );
}
#endif//#if USE_KEY_HOOK

/*----------disable window accessibility keys------*/
void AllowAccessibilityShortcutKeys( bool allowKeys )
{
    if( allowKeys )
    {
        //restore stickyKeys/etc to original state  
        SystemParametersInfo(SPI_SETSTICKYKEYS, sizeof(STICKYKEYS), &g_oldStickyKeys, 0);
        SystemParametersInfo(SPI_SETTOGGLEKEYS, sizeof(TOGGLEKEYS), &g_oldToggleKeys, 0);
        SystemParametersInfo(SPI_SETFILTERKEYS, sizeof(FILTERKEYS), &g_oldFilterKeys, 0);
    }
    else
    {
        STICKYKEYS skOff = g_oldStickyKeys;
        if( (skOff.dwFlags & SKF_STICKYKEYSON) == 0 )
        {
            //disable the hotkey and the confirmation
            skOff.dwFlags &= ~SKF_HOTKEYACTIVE;
            skOff.dwFlags &= ~SKF_CONFIRMHOTKEY;
 
            SystemParametersInfo(SPI_SETSTICKYKEYS, sizeof(STICKYKEYS), &skOff, 0);
        }
 
        TOGGLEKEYS tkOff = g_oldToggleKeys;
        if( (tkOff.dwFlags & TKF_TOGGLEKEYSON) == 0 )
        {
            //disable the hotkey and the confirmation
            tkOff.dwFlags &= ~TKF_HOTKEYACTIVE;
            tkOff.dwFlags &= ~TKF_CONFIRMHOTKEY;
 
            SystemParametersInfo(SPI_SETTOGGLEKEYS, sizeof(TOGGLEKEYS), &tkOff, 0);
        }
 
        FILTERKEYS fkOff = g_oldFilterKeys;
        if( (fkOff.dwFlags & FKF_FILTERKEYSON) == 0 )
        {
            //disable the hotkey and the confirmation
            fkOff.dwFlags &= ~FKF_HOTKEYACTIVE;
            fkOff.dwFlags &= ~FKF_CONFIRMHOTKEY;
 
            SystemParametersInfo(SPI_SETFILTERKEYS, sizeof(FILTERKEYS), &fkOff, 0);
        }
    }
}

/*-------engine 's window class--------*/
HQEngineWindow::HQEngineWindow(const char *title, const char *settingFileDir ,  HQWIPPlatformSpecificType* icons)
: HQEngineBaseWindow(settingFileDir)
{
	/*--------copy title----------*/
	size_t len = strlen(title);
	m_title = HQ_NEW wchar_t [len + 1];
	m_title[len] = '\0';
	for (size_t i = 0; i < len ; ++i)
		m_title[i] = title[i];
	
	/*-------create window class---------*/
	WNDCLASSEX wndclass ;

	if (icons == NULL)
	{
		wndclass.hIconSm       = LoadIcon(NULL,IDI_APPLICATION);
		wndclass.hIcon         = LoadIcon(NULL,IDI_APPLICATION);
	}
	else
	{
		wndclass.hIconSm       = icons->sicon;
		wndclass.hIcon         = icons->icon;
	}
	wndclass.cbSize        = sizeof(wndclass);
	wndclass.lpfnWndProc   = WndProc;
	wndclass.cbClsExtra    = 0;
	wndclass.cbWndExtra    = 0;
	wndclass.hInstance     = ge_module;
	wndclass.hCursor       = LoadCursor(NULL, IDC_ARROW);
	wndclass.hbrBackground = (HBRUSH)(COLOR_WINDOW);
	wndclass.lpszMenuName  = NULL;
	wndclass.lpszClassName = m_title;
	wndclass.style         = CS_HREDRAW | CS_VREDRAW ;
	
	if (RegisterClassEx(&wndclass) == 0)
		throw std::bad_alloc();

	RECT winRect = {0 , 0 , this->GetWidth() , this->GetHeight()};
	AdjustWindowRect(&winRect,HQ_WINDOW_STYLE,FALSE);
	
	/*---------create window----------------*/
	if (!(m_window = CreateWindowEx( NULL, m_title,
		m_title,
		HQ_WINDOW_STYLE,
		0,
		0,
		winRect.right - winRect.left, winRect.bottom - winRect.top, 
		NULL, NULL, ge_module , NULL)))
	{
		UnregisterClass(m_title , ge_module);//unregister window class
		throw std::bad_alloc();
	}

#if USE_KEY_HOOK
	//performance problem
	/*-----disable window key--------------*/
	if (g_keyboardHook == NULL)
		g_keyboardHook = SetWindowsHookEx( WH_KEYBOARD_LL,  LowLevelKeyboardProc, ge_module, 0 );
#else
	RAWINPUTDEVICE rid;
	rid.usUsagePage = 1; 
	rid.usUsage = 6;//keyboard 
	rid.dwFlags = RIDEV_NOLEGACY | RIDEV_NOHOTKEYS | RIDEV_APPKEYS;
	rid.hwndTarget = m_window;

	if (RegisterRawInputDevices(&rid, 1, sizeof(RAWINPUTDEVICE)) == FALSE) 
	{
		//do something?
	}

	g_keyPressed.clear();//reset hash table
#endif
	/*--- save the current sticky/toggle/filter key settings so they can be restored later--*/
    SystemParametersInfo(SPI_GETSTICKYKEYS, sizeof(STICKYKEYS), &g_oldStickyKeys, 0);
    SystemParametersInfo(SPI_GETTOGGLEKEYS, sizeof(TOGGLEKEYS), &g_oldToggleKeys, 0);
    SystemParametersInfo(SPI_GETFILTERKEYS, sizeof(FILTERKEYS), &g_oldFilterKeys, 0);

	AllowAccessibilityShortcutKeys( false );
}

HQEngineWindow::~HQEngineWindow()
{
#if USE_KEY_HOOK
	if (g_keyboardHook != NULL)
	{
		UnhookWindowsHookEx(g_keyboardHook);
		g_keyboardHook = NULL;
	}
#else
	RAWINPUTDEVICE rid;
	rid.usUsagePage = 1; 
	rid.usUsage = 6;//keyboard 
	rid.dwFlags = RIDEV_REMOVE;
	rid.hwndTarget = m_window;

	//remove raw keyboard
	if (RegisterRawInputDevices(&rid, 1, sizeof(RAWINPUTDEVICE)) == FALSE) 
	{
		//do something?
	}
#endif
	//restore shortcut keys setting
	AllowAccessibilityShortcutKeys(true);

	DestroyWindow(m_window);//destroy window
	UnregisterClass(m_title , ge_module);//unregister window class

	delete[] m_title;
}

HQReturnVal HQEngineWindow::Show()
{
	if (ShowWindow(m_window, SW_SHOW))
		return HQ_OK;
	else
		return HQ_FAILED;
}
