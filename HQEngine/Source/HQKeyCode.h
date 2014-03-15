/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#ifndef HQ_ENGINE_KEY_CODE_H
#define HQ_ENGINE_KEY_CODE_H

#include "HQPlatformDef.h"

#if defined HQ_WIN_DESKTOP_PLATFORM

#include <winuser.h>

#elif (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)

#elif defined HQ_MAC_PLATFORM

#include <Carbon/Carbon.h>

#elif defined HQ_IPHONE_PLATFORM

#elif defined HQ_ANDROID_PLATFORM

#elif defined HQ_LINUX_PLATFORM

#include <X11/keysym.h>

#else

#error need implement

#endif

namespace HQKeyCode
{
#if (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
	using namespace Windows::System;
#endif

	///mouse button key code
	enum MouseButtonEnum
	{
		LBUTTON = 1,//left button
		RBUTTON = 2,//right button
		MBUTTON = 3//middle button
	};
	
	///keyboard key code
	enum KeyCodeEnum
	{
#if defined HQ_WIN_DESKTOP_PLATFORM
		BACKSPACE = VK_BACK,
		TAB = VK_TAB ,
		ENTER = VK_RETURN,

		LSHIFT = VK_LSHIFT,
		RSHIFT = VK_RSHIFT,
		LCONTROL = VK_LCONTROL,
		RCONTROL = VK_RCONTROL,
		LALT = VK_LMENU,
		RALT = VK_RMENU,
		PAUSE = VK_PAUSE,
		CAP_SLOCK = VK_CAPITAL,
		ESCAPE = VK_ESCAPE ,

		SPACE = VK_SPACE,
		PAGE_UP = VK_PRIOR,
		PAGE_DOWN = VK_NEXT,
		END = VK_END ,
		HOME = VK_HOME,
		LEFT = VK_LEFT,
		UP = VK_UP,
		RIGHT = VK_RIGHT ,
		DOWN = VK_DOWN ,
		SELECT = VK_SELECT ,
		PRINT_SCREEN = VK_SNAPSHOT,
		INSERT = VK_INSERT,
		DEL = VK_DELETE ,
		
		NUM0 = 0x30,
		NUM1 = 0x31,
		NUM2 = 0x32,
		NUM3 = 0x33,
		NUM4 = 0x34,
		NUM5 = 0x35,
		NUM6 = 0x36,
		NUM7 = 0x37,
		NUM8 = 0x38,
		NUM9 = 0x39,
		
		A = 0x41,
		B = 0x42,
		C = 0x43,
		D = 0x44,
		E = 0x45,
		F = 0x46,
		G = 0x47,
		H = 0x48,
		I = 0x49,
		J = 0x4A,
		K = 0x4B,
		L = 0x4C,
		M = 0x4D,
		N = 0x4E,
		O = 0x4F,
		P = 0x50,
		Q = 0x51,
		R = 0x52,
		S = 0x53,
		T = 0x54,
		U = 0x55,
		V = 0x56,
		W = 0x57,
		X = 0x58,
		Y = 0x59,
		Z = 0x5A,

		//LWIN = VK_LWIN,//left window key
		//RWIN = VK_RWIN,//right window key

		SLEEP = VK_SLEEP,

		NUMPAD0 = VK_NUMPAD0,
		NUMPAD1 = VK_NUMPAD1,
		NUMPAD2 = VK_NUMPAD2,
		NUMPAD3 = VK_NUMPAD3,
		NUMPAD4 = VK_NUMPAD4,
		NUMPAD5 = VK_NUMPAD5,
		NUMPAD6 = VK_NUMPAD6,
		NUMPAD7 = VK_NUMPAD7,
		NUMPAD8 = VK_NUMPAD8,
		NUMPAD9 = VK_NUMPAD9,
		MULTIPLY = VK_MULTIPLY,
		ADD = VK_ADD,
		SEPARATOR = VK_SEPARATOR,
		SUBTRACT = VK_SUBTRACT,
		DECIMAL = VK_DECIMAL,
		DIVIDE = VK_DIVIDE , //"/"
		F1 = VK_F1,
		F2 = VK_F2,
		F3 = VK_F3,
		F4 = VK_F4,
		F5 = VK_F5,
		F6 = VK_F6,
		F7 = VK_F7,
		F8 = VK_F8,
		F9 = VK_F9,
		F10 = VK_F10,
		F11 = VK_F11,
		F12 = VK_F12,

		NUM_LOCK = VK_NUMLOCK,
		SCROLL_LOCK = VK_SCROLL,

		PLUS = VK_OEM_PLUS,// '+' any country
		COMMA = VK_OEM_COMMA,// ',' any country
		MINUS = VK_OEM_MINUS,// '-' any country
		PERIOD = VK_OEM_PERIOD,// '.' any country
		
		SPECIAL1 = VK_OEM_1,// ';:' for US
		SPECIAL2 = VK_OEM_2,// '/?' for US
		SPECIAL3 = VK_OEM_3,// '`~' for US

		SPECIAL4 = VK_OEM_4,//  '[{' for US
		SPECIAL5 = VK_OEM_5,//  '\|' for US
		SPECIAL6 = VK_OEM_6,//  ']}' for US
		SPECIAL7 = VK_OEM_7,//  ''"' for US
		
#elif (defined HQ_WIN_PHONE_PLATFORM || defined HQ_WIN_STORE_PLATFORM)
		BACKSPACE = VirtualKey::Back,
		TAB = VirtualKey::Tab ,
		ENTER = VirtualKey::Enter,

		LSHIFT = VirtualKey::LeftShift,
		RSHIFT = VirtualKey::RightShift,
		LCONTROL = VirtualKey::LeftControl,
		RCONTROL = VirtualKey::RightControl,
		LALT = VirtualKey::LeftMenu,
		RALT = VirtualKey::RightMenu,
		PAUSE = VirtualKey::Pause,
		CAP_SLOCK = VirtualKey::CapitalLock,
		ESCAPE = VirtualKey::Escape ,

		SPACE = VirtualKey::Space,
		PAGE_UP = VirtualKey::PageUp,
		PAGE_DOWN = VirtualKey::PageDown,
		END = VirtualKey::End ,
		HOME = VirtualKey::Home,
		LEFT = VirtualKey::Left,
		UP = VirtualKey::Up,
		RIGHT = VirtualKey::Right ,
		DOWN = VirtualKey::Down ,
		SELECT = VirtualKey::Select ,
		PRINT_SCREEN = VirtualKey::Snapshot,
		INSERT = VirtualKey::Insert,
		DEL = VirtualKey::Delete ,
		
		NUM0 = VirtualKey::Number0,
		NUM1 = VirtualKey::Number1,
		NUM2 = VirtualKey::Number2,
		NUM3 = VirtualKey::Number3,
		NUM4 = VirtualKey::Number4,
		NUM5 = VirtualKey::Number5,
		NUM6 = VirtualKey::Number6,
		NUM7 = VirtualKey::Number7,
		NUM8 = VirtualKey::Number8,
		NUM9 = VirtualKey::Number9,
		
		A = VirtualKey::A,
		B = VirtualKey::B,
		C = VirtualKey::C,
		D = VirtualKey::D,
		E = VirtualKey::E,
		F = VirtualKey::F,
		G = VirtualKey::G,
		H = VirtualKey::H,
		I = VirtualKey::I,
		J = VirtualKey::J,
		K = VirtualKey::K,
		L = VirtualKey::L,
		M = VirtualKey::M,
		N = VirtualKey::N,
		O = VirtualKey::O,
		P = VirtualKey::P,
		Q = VirtualKey::Q,
		R = VirtualKey::R,
		S = VirtualKey::S,
		T = VirtualKey::T,
		U = VirtualKey::U,
		V = VirtualKey::V,
		W = VirtualKey::W,
		X = VirtualKey::X,
		Y = VirtualKey::Y,
		Z = VirtualKey::Z,

		//LWIN = VK_LWIN,//left window key
		//RWIN = VK_RWIN,//right window key

		SLEEP = VirtualKey::Sleep,

		NUMPAD0 = VirtualKey::NumberPad0,
		NUMPAD1 = VirtualKey::NumberPad1,
		NUMPAD2 = VirtualKey::NumberPad2,
		NUMPAD3 = VirtualKey::NumberPad3,
		NUMPAD4 = VirtualKey::NumberPad4,
		NUMPAD5 = VirtualKey::NumberPad5,
		NUMPAD6 = VirtualKey::NumberPad6,
		NUMPAD7 = VirtualKey::NumberPad7,
		NUMPAD8 = VirtualKey::NumberPad8,
		NUMPAD9 = VirtualKey::NumberPad9,
		MULTIPLY = VirtualKey::Multiply,
		ADD = VirtualKey::Add,
		SEPARATOR = VirtualKey::Separator,
		SUBTRACT = VirtualKey::Subtract,
		DECIMAL = VirtualKey::Decimal,
		DIVIDE = VirtualKey::Divide , //"/"
		F1 = VirtualKey::F1,
		F2 = VirtualKey::F2,
		F3 = VirtualKey::F3,
		F4 = VirtualKey::F4,
		F5 = VirtualKey::F5,
		F6 = VirtualKey::F6,
		F7 = VirtualKey::F7,
		F8 = VirtualKey::F8,
		F9 = VirtualKey::F9,
		F10 = VirtualKey::F10,
		F11 = VirtualKey::F11,
		F12 = VirtualKey::F12,

		NUM_LOCK = VirtualKey::NumberKeyLock,
		SCROLL_LOCK = VirtualKey::Scroll,

#elif defined HQ_MAC_PLATFORM
		BACKSPACE = kVK_Delete,
		TAB = kVK_Tab ,
		ENTER = kVK_Return,
		
		LSHIFT = kVK_Shift,
		RSHIFT = kVK_RightShift,
		LCONTROL = kVK_Control,
		RCONTROL = kVK_RightControl,
		COMMAND = kVK_Command,
		CAPS_LOCK = kVK_CapsLock,
		ESCAPE = kVK_Escape ,
		
		SPACE = kVK_Space,
		PAGE_UP = kVK_PageUp,
		PAGE_DOWN = kVK_PageDown,
		END = kVK_End ,
		HOME = kVK_Home,
		LEFT = kVK_LeftArrow,
		UP = kVK_UpArrow,
		RIGHT = kVK_RightArrow ,
		DOWN = kVK_DownArrow ,
		DEL = kVK_ForwardDelete ,
		
		NUM0 = kVK_ANSI_0,
		NUM1 = kVK_ANSI_1,
		NUM2 = kVK_ANSI_2,
		NUM3 = kVK_ANSI_3,
		NUM4 = kVK_ANSI_4,
		NUM5 = kVK_ANSI_5,
		NUM6 = kVK_ANSI_6,
		NUM7 = kVK_ANSI_7,
		NUM8 = kVK_ANSI_8,
		NUM9 = kVK_ANSI_9,
		
		A = kVK_ANSI_A,
		B = kVK_ANSI_B,
		C = kVK_ANSI_C,
		D = kVK_ANSI_D,
		E = kVK_ANSI_E,
		F = kVK_ANSI_F,
		G = kVK_ANSI_G,
		H = kVK_ANSI_H,
		I = kVK_ANSI_I,
		J = kVK_ANSI_J,
		K = kVK_ANSI_K,
		L = kVK_ANSI_L,
		M = kVK_ANSI_M,
		N = kVK_ANSI_N,
		O = kVK_ANSI_O,
		P = kVK_ANSI_P,
		Q = kVK_ANSI_Q,
		R = kVK_ANSI_R,
		S = kVK_ANSI_S,
		T = kVK_ANSI_T,
		U = kVK_ANSI_U,
		V = kVK_ANSI_V,
		W = kVK_ANSI_W,
		X = kVK_ANSI_X,
		Y = kVK_ANSI_Y,
		Z = kVK_ANSI_Z,
		
		NUMPAD0 = kVK_ANSI_Keypad0,
		NUMPAD1 = kVK_ANSI_Keypad1,
		NUMPAD2 = kVK_ANSI_Keypad2,
		NUMPAD3 = kVK_ANSI_Keypad3,
		NUMPAD4 = kVK_ANSI_Keypad4,
		NUMPAD5 = kVK_ANSI_Keypad5,
		NUMPAD6 = kVK_ANSI_Keypad6,
		NUMPAD7 = kVK_ANSI_Keypad7,
		NUMPAD8 = kVK_ANSI_Keypad8,
		NUMPAD9 = kVK_ANSI_Keypad9,
		MULTIPLY = kVK_ANSI_KeypadMultiply,
		ADD = kVK_ANSI_KeypadPlus,
		SUBTRACT = kVK_ANSI_KeypadMinus,
		DECIMAL = kVK_ANSI_KeypadDecimal,
		DIVIDE = kVK_ANSI_KeypadDivide , //"/"
		NUMPAD_ENTER = kVK_ANSI_KeypadEnter,
		F1 = kVK_F1,
		F2 = kVK_F2,
		F3 = kVK_F3,
		F4 = kVK_F4,
		F5 = kVK_F5,
		F6 = kVK_F6,
		F7 = kVK_F7,
		F8 = kVK_F8,
		F9 = kVK_F9,
		F10 = kVK_F10,
		F11 = kVK_F11,
		F12 = kVK_F12,
		
		PLUS = kVK_ANSI_Equal,// '+' any country
		COMMA = kVK_ANSI_Comma,// ',' any country
		MINUS = kVK_ANSI_Minus,// '-' any country
		PERIOD = kVK_ANSI_Period,// '.' any country
		
		SPECIAL1 = kVK_ANSI_Semicolon,// ';:' for US
		SPECIAL2 = kVK_ANSI_Slash,// '/?' for US
		SPECIAL3 = kVK_ANSI_Grave,// '`~' for US
		
		SPECIAL4 = kVK_ANSI_LeftBracket,//  '[{' for US
		SPECIAL5 = kVK_ANSI_Backslash,//  '\|' for US
		SPECIAL6 = kVK_ANSI_RightBracket,//  ']}' for US
		SPECIAL7 = kVK_ANSI_Quote,//  ''"' for US
#elif defined HQ_IPHONE_PLATFORM

#elif defined HQ_ANDROID_PLATFORM

#elif defined HQ_LINUX_PLATFORM
			
		BACKSPACE = XK_BackSpace,
		TAB = XK_Tab ,
		ENTER = XK_Return,

		LSHIFT = XK_Shift_L,
		RSHIFT = XK_Hyper_R,
		LCONTROL = XK_Control_L,
		RCONTROL = XK_Control_R,
		LALT = XK_Alt_L,
		RALT = XK_Alt_R,
		PAUSE = XK_Pause,
		CAP_SLOCK = XK_Caps_Lock,
		ESCAPE = XK_Escape ,

		SPACE = XK_space,
		PAGE_UP = XK_Page_Up,
		PAGE_DOWN = XK_Page_Down,
		END = XK_End ,
		HOME = XK_Home,
		LEFT = XK_Left,
		UP = XK_Up,
		RIGHT = XK_Right ,
		DOWN = XK_Down ,
		SELECT = XK_Select ,
		PRINT_SCREEN = XK_Print,
		INSERT = XK_Insert,
		DEL = XK_Delete ,
		
		NUM0 = 0x30,
		NUM1 = 0x31,
		NUM2 = 0x32,
		NUM3 = 0x33,
		NUM4 = 0x34,
		NUM5 = 0x35,
		NUM6 = 0x36,
		NUM7 = 0x37,
		NUM8 = 0x38,
		NUM9 = 0x39,
		
		A = 0x41,
		B = 0x42,
		C = 0x43,
		D = 0x44,
		E = 0x45,
		F = 0x46,
		G = 0x47,
		H = 0x48,
		I = 0x49,
		J = 0x4A,
		K = 0x4B,
		L = 0x4C,
		M = 0x4D,
		N = 0x4E,
		O = 0x4F,
		P = 0x50,
		Q = 0x51,
		R = 0x52,
		S = 0x53,
		T = 0x54,
		U = 0x55,
		V = 0x56,
		W = 0x57,
		X = 0x58,
		Y = 0x59,
		Z = 0x5A,

		NUMPAD0 = XK_KP_0,
		NUMPAD1 = XK_KP_1,
		NUMPAD2 = XK_KP_2,
		NUMPAD3 = XK_KP_3,
		NUMPAD4 = XK_KP_4,
		NUMPAD5 = XK_KP_5,
		NUMPAD6 = XK_KP_6,
		NUMPAD7 = XK_KP_7,
		NUMPAD8 = XK_KP_8,
		NUMPAD9 = XK_KP_9,
		MULTIPLY = XK_KP_Multiply,
		ADD = XK_KP_Add,
		SEPARATOR = XK_KP_Separator,
		SUBTRACT = XK_KP_Subtract,
		DECIMAL = XK_KP_Decimal,
		DIVIDE = XK_KP_Divide , //"/"
		F1 = XK_F1,
		F2 = XK_F2,
		F3 = XK_F3,
		F4 = XK_F4,
		F5 = XK_F5,
		F6 = XK_F6,
		F7 = XK_F7,
		F8 = XK_F8,
		F9 = XK_F9,
		F10 = XK_F10,
		F11 = XK_F11,
		F12 = XK_F12,

		NUM_LOCK = XK_Num_Lock,
		SCROLL_LOCK = XK_Scroll_Lock,

		PLUS = XK_equal,// '=+' any country
		COMMA = XK_comma,// ',' any country
		MINUS = XK_minus,// '-' any country
		PERIOD = XK_period,// '.' any country
		
		SPECIAL1 = XK_semicolon,// ';:' for US
		SPECIAL2 = XK_slash,// '/?' for US
		SPECIAL3 = XK_grave,// '`~' for US

		SPECIAL4 = XK_bracketleft,//  '[{' for US
		SPECIAL5 = XK_backslash,//  '\|' for US
		SPECIAL6 = XK_bracketright,//  ']}' for US
		SPECIAL7 = XK_apostrophe,//  ''"' for US
#else
#	error need implement
#endif

		UNDEFINED = 0xffffffff,
		

	};
}

#endif
