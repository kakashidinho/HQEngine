#include "../HQEnginePCH.h"
#include "../HQEngineWindow.h"
#include "../HQEventSeparateThread.h"
#include "HQWinStoreUtil.h"
#include "string.h"
#include <iostream>
#include <agile.h>

using namespace Windows::UI::Core;
using namespace Windows::Devices::Input;
using namespace Windows::Foundation;

extern HQEventQueue* hq_engine_eventQueue_internal;
extern Platform::Agile<CoreWindow> hq_engine_coreWindow_internal;
#ifdef HQ_WIN_STORE_PLATFORM
extern Platform::Agile <CoreCursor> hq_engine_defaultCursor_internal;
#endif

//application status flag
extern int hq_internalAppExit;
extern void HQAppInternalSetStatusFlag(int flag);
extern int HQAppInternalGetStatusFlag();

static float ConvertDipsToPixels(float dips)
{
	static const float dipsPerInch = 96.0f;
	return floor(dips * Windows::Graphics::Display:: DisplayProperties::LogicalDpi / dipsPerInch + 0.5f); // Round to nearest integer.
}

/*-------window event handlers--------------*/
HQWindowsEventHandler::HQWindowsEventHandler(Windows::UI::Core::CoreCursor ^ cursor)
	:m_mouseHide(false), m_currentCursor(cursor),
	m_leftMousePressed(false), m_rightMousePressed(false), m_middleMousePressed(false),
	m_prevTouchPos(new HQPoolMemoryManager(sizeof(TouchHistoryTableType::LinkedListNodeType), MAX_MULTITOUCHES, 1, false))
{
	AttachToWindow();
}

void HQWindowsEventHandler::AttachToWindow()
{
	//run on UI thread

	HQWinStoreUtil::RunOnUIThread(
		hq_engine_coreWindow_internal->Dispatcher,
		[this]()
		{
			this->AttachToWindowOnUIThread();
		});
}

void HQWindowsEventHandler::DetachFromWindow()
{
	//run on UI thread

	HQWinStoreUtil::RunOnUIThreadAndWait( 
		hq_engine_coreWindow_internal->Dispatcher,
		[this]()
		{
			this->DetachFromWindowOnUIThread();
		});
}

void HQWindowsEventHandler::AttachToWindowOnUIThread()
{
	auto window = CoreWindow::GetForCurrentThread();

	m_OnClosedToken = window->Closed += 
		ref new TypedEventHandler<CoreWindow^, CoreWindowEventArgs^>(this, &HQWindowsEventHandler::OnClosed);

	m_OnKeyPressedToken = window->KeyDown +=
		ref new TypedEventHandler<CoreWindow^, KeyEventArgs^>(this, &HQWindowsEventHandler::OnKeyPressed);

	m_OnKeyReleasedToken = window->KeyUp +=
		ref new TypedEventHandler<CoreWindow^, KeyEventArgs^>(this, &HQWindowsEventHandler::OnKeyReleased);

	m_OnPointerPressedToken = window->PointerPressed +=
		ref new TypedEventHandler<CoreWindow^, PointerEventArgs^>(this, &HQWindowsEventHandler::OnPointerPressed);

	m_OnPointerReleasedToken = window->PointerReleased +=
		ref new TypedEventHandler<CoreWindow^, PointerEventArgs^>(this, &HQWindowsEventHandler::OnPointerReleased);

	m_OnPointerMovedToken = window->PointerMoved +=
		ref new TypedEventHandler<CoreWindow^, PointerEventArgs^>(this, &HQWindowsEventHandler::OnPointerMoved);

	m_OnPointerWheelChangedToken = window->PointerWheelChanged +=
		ref new TypedEventHandler<CoreWindow^, PointerEventArgs^>(this, &HQWindowsEventHandler::OnPointerWheelChanged);

#ifdef HQ_WIN_STORE_PLATFORM
	m_OnMouseMovedToken = MouseDevice::GetForCurrentView()->MouseMoved += 
		ref new TypedEventHandler<MouseDevice^, MouseEventArgs^>(this, &HQWindowsEventHandler::OnMouseMoved);
#endif
}

void HQWindowsEventHandler::DetachFromWindowOnUIThread()
{
	auto window = CoreWindow::GetForCurrentThread();

#ifdef HQ_WIN_STORE_PLATFORM
	window->PointerCursor = m_currentCursor.Get();
#endif

	window->Closed -= m_OnClosedToken;

	window->KeyDown -= m_OnKeyPressedToken;

	window->KeyUp -= m_OnKeyReleasedToken;

	window->PointerPressed -= m_OnPointerPressedToken;

	window->PointerReleased -= m_OnPointerReleasedToken;

	window->PointerMoved -= m_OnPointerMovedToken;

	window->PointerWheelChanged -= m_OnPointerWheelChangedToken;
#ifdef HQ_WIN_STORE_PLATFORM
	MouseDevice::GetForCurrentView()->MouseMoved -=  m_OnMouseMovedToken;
#endif
}

void HQWindowsEventHandler::HideMouseOnUITHread(bool hide)
{
#ifdef HQ_WIN_STORE_PLATFORM
	auto window = CoreWindow::GetForCurrentThread();

	m_mouseHide = hide;

	if (m_mouseHide)
	{
		window->PointerCursor = nullptr;
	}
	else
	{
		window->PointerCursor = m_currentCursor.Get();
	}
#endif
}

void HQWindowsEventHandler:: HideMouse(bool hide) {
	//run on UI thread
	HQWinStoreUtil::RunOnUIThread(
		hq_engine_coreWindow_internal->Dispatcher,
		[this, hide]()
		{	
			this->HideMouseOnUITHread(hide);
		});
}

void HQWindowsEventHandler::OnClosed(CoreWindow^ sender, CoreWindowEventArgs^ args)
{
	HQAppInternalSetStatusFlag(hq_internalAppExit);//set application exit flag

	HQEngineApp::GetInstance()->GetWindowListener()->WindowClosed();
	HQEngineApp::GetInstance()->GetAppListener()->OnDestroy();//app will be destroyed after this method
}

void HQWindowsEventHandler::OnKeyPressed(CoreWindow^ sender, KeyEventArgs^ args)
{
	HQEventQueue::ScopeLock sl(hq_engine_eventQueue_internal);
	HQEvent &newEvent = hq_engine_eventQueue_internal->BeginAddEvent();
	
	newEvent.type = HQ_KEY_PRESSED;
	newEvent.keyData.keyCode = (HQKeyCodeType)args->VirtualKey;
	
	hq_engine_eventQueue_internal->EndAddEvent();
}

void HQWindowsEventHandler::OnKeyReleased(CoreWindow^ sender, KeyEventArgs^ args)
{
	HQEventQueue::ScopeLock sl(hq_engine_eventQueue_internal);
	HQEvent &newEvent = hq_engine_eventQueue_internal->BeginAddEvent();
	
	newEvent.type = HQ_KEY_RELEASED;
	newEvent.keyData.keyCode = (HQKeyCodeType)args->VirtualKey;
	
	hq_engine_eventQueue_internal->EndAddEvent();
}

void HQWindowsEventHandler::OnPointerPressed(CoreWindow^ sender, PointerEventArgs^ args)
{
	HQEventQueue::ScopeLock sl(hq_engine_eventQueue_internal);

	auto pointer = args->CurrentPoint;
#ifdef HQ_WIN_STORE_PLATFORM
	switch (pointer->PointerDevice->PointerDeviceType)
	{
	case PointerDeviceType::Mouse:
		{
			//begin add event to game thread's event queue
			HQEvent &newEvent = hq_engine_eventQueue_internal->BeginAddEvent();

			newEvent.type = HQ_MOUSE_PRESSED;
			//position
			newEvent.mouseData.position.x =  (hqint32)pointer->Position.X;
			newEvent.mouseData.position.y =  (hqint32)pointer->Position.Y;

			//button
			if (pointer->Properties->IsLeftButtonPressed)
			{
				m_leftMousePressed = true;
				newEvent.mouseData.keyCode = HQKeyCode::LBUTTON;
			}
			if (pointer->Properties->IsMiddleButtonPressed)
			{
				m_middleMousePressed = true;
				newEvent.mouseData.keyCode = HQKeyCode::MBUTTON;
			}
			if (pointer->Properties->IsRightButtonPressed)
			{
				m_rightMousePressed = true;
				newEvent.mouseData.keyCode = HQKeyCode::RBUTTON;
			}
	
			//finish add event to game thread's event queue
			hq_engine_eventQueue_internal->EndAddEvent();
		}
		break;
	case PointerDeviceType::Touch:
#endif//#ifdef HQ_WIN_STORE_PLATFORM
		OnTouchEvent(HQ_TOUCH_BEGAN, pointer);
#ifdef HQ_WIN_STORE_PLATFORM
		break;
	}//switch (pointer->PointerDevice->PointerDeviceType)
#endif//#ifdef HQ_WIN_STORE_PLATFORM
}

void HQWindowsEventHandler::OnPointerReleased(CoreWindow^ sender, PointerEventArgs^ args)
{
	HQEventQueue::ScopeLock sl(hq_engine_eventQueue_internal);

	auto pointer = args->CurrentPoint;
#ifdef HQ_WIN_STORE_PLATFORM
	switch (pointer->PointerDevice->PointerDeviceType)
	{
	case PointerDeviceType::Mouse:
		{
			//begin add event to game thread's event queue
			HQEvent &newEvent = hq_engine_eventQueue_internal->BeginAddEvent();

			newEvent.type = HQ_MOUSE_RELEASED;
			//position
			newEvent.mouseData.position.x =  (hqint32)pointer->Position.X;
			newEvent.mouseData.position.y =  (hqint32)pointer->Position.Y;

			//button
			if (m_leftMousePressed && !pointer->Properties->IsLeftButtonPressed)
			{
				m_leftMousePressed = false;
				newEvent.mouseData.keyCode = HQKeyCode::LBUTTON;
			}
			if (m_middleMousePressed && !pointer->Properties->IsMiddleButtonPressed)
			{
				m_middleMousePressed = false;
				newEvent.mouseData.keyCode = HQKeyCode::MBUTTON;
			}
			if (m_rightMousePressed && !pointer->Properties->IsRightButtonPressed)
			{
				m_rightMousePressed = false;
				newEvent.mouseData.keyCode = HQKeyCode::RBUTTON;
			}

			//finish add event to game thread's event queue
			hq_engine_eventQueue_internal->EndAddEvent();
		}
		break;
	case PointerDeviceType::Touch:
#endif//#ifdef HQ_WIN_STORE_PLATFORM
		OnTouchEvent(HQ_TOUCH_ENDED, pointer);
#ifdef HQ_WIN_STORE_PLATFORM
		break;
	}//switch (pointer->PointerDevice->PointerDeviceType)
#endif//#ifdef HQ_WIN_STORE_PLATFORM
}

void HQWindowsEventHandler::OnPointerMoved(CoreWindow^ sender, PointerEventArgs^ args)
{
	HQEventQueue::ScopeLock sl(hq_engine_eventQueue_internal);

	auto pointer = args->CurrentPoint;
#ifdef HQ_WIN_STORE_PLATFORM
	switch (pointer->PointerDevice->PointerDeviceType)
	{
	case PointerDeviceType::Mouse:
		{
			if (!m_mouseHide)
			{
				//begin add event to game thread's event queue
				HQEvent &newEvent = hq_engine_eventQueue_internal->BeginAddEvent();

				newEvent.type = HQ_MOUSE_MOVED;
				//position
				newEvent.mouseData.position.x =  (hqint32)pointer->Position.X;
				newEvent.mouseData.position.y =  (hqint32)pointer->Position.Y;

				//finish add event to game thread's event queue
				hq_engine_eventQueue_internal->EndAddEvent();
			}
		}
		break;
	case PointerDeviceType::Touch:
#endif//#ifdef HQ_WIN_STORE_PLATFORM
		OnTouchEvent(HQ_TOUCH_MOVED, pointer);
#ifdef HQ_WIN_STORE_PLATFORM
		break;
	}//switch (pointer->PointerDevice->PointerDeviceType)
#endif//#ifdef HQ_WIN_STORE_PLATFORM
}

void HQWindowsEventHandler::OnPointerWheelChanged(CoreWindow^ sender, PointerEventArgs^ args)
{
	HQEventQueue::ScopeLock sl(hq_engine_eventQueue_internal);

	auto pointer = args->CurrentPoint;
#ifdef HQ_WIN_STORE_PLATFORM
	switch (pointer->PointerDevice->PointerDeviceType)
	{
	case PointerDeviceType::Mouse:
		{
			//begin add event to game thread's event queue
			HQEvent &newEvent = hq_engine_eventQueue_internal->BeginAddEvent();

			newEvent.type = HQ_MOUSE_WHEEL;
			//position
			newEvent.mouseData.position.x =  (hqint32)pointer->Position.X;
			newEvent.mouseData.position.y =  (hqint32)pointer->Position.Y;
			
			//wheel delta
			newEvent.mouseData.wheelDelta = (hqfloat32)pointer->Properties->MouseWheelDelta;

			//finish add event to game thread's event queue
			hq_engine_eventQueue_internal->EndAddEvent();
		}
		break;
	}//switch (pointer->PointerDevice->PointerDeviceType)
#endif//#ifdef HQ_WIN_STORE_PLATFORM
}

void HQWindowsEventHandler::OnMouseMoved(MouseDevice^ device, MouseEventArgs^ args)
{
	//relative mouse move
	if (m_mouseHide == false)
		return;

	HQEventQueue::ScopeLock sl(hq_engine_eventQueue_internal);

	//begin add event to game thread's event queue
	HQEvent &newEvent = hq_engine_eventQueue_internal->BeginAddEvent();

	newEvent.type = HQ_MOUSE_MOVED;

	newEvent.mouseData.position.x = (hqint32)args->MouseDelta.X;
	newEvent.mouseData.position.y = (hqint32)args->MouseDelta.Y;

	//finish add event to game thread's event queue
	hq_engine_eventQueue_internal->EndAddEvent();
}

void HQWindowsEventHandler::OnTouchEvent(HQEventType type, Windows::UI::Input::PointerPoint^ pointer)
{
	HQEventQueue::ScopeLock sl(hq_engine_eventQueue_internal);

	//begin add event to game thread's event queue
	HQEvent &newEvent = hq_engine_eventQueue_internal->BeginAddEvent();

	newEvent.type = type;

	newEvent.touchData.numTouches = 1;
	HQTouch& newTouch = newEvent.touchData.touches[0];
	newTouch.touchID = (hqint32)pointer->PointerId;
	newTouch.position.x = pointer->Position.X;
	newTouch.position.y = pointer->Position.Y;

	HQPointf * oldTouchPosition = NULL;

	//find the previous position of this touch
	TouchHistoryTableType::Iterator ite;
	m_prevTouchPos.GetIterator(ite);
	while (!ite.IsAtEnd())
	{
		if (ite->m_first == pointer->PointerId)
		{
			oldTouchPosition = &ite.GetPointerNonCheck()->m_second;
			break;
		}

		++ite;
	}

	switch (type)
	{
	case HQ_TOUCH_BEGAN:
		newTouch.prev_position = newTouch.position;
		if (oldTouchPosition != NULL)//remove the touch from history table
			m_prevTouchPos.RemoveAt(ite.GetNode());

		//store the current position to the history table
		m_prevTouchPos.PushBack(TouchHistoryTableType::LinkedListItemType(pointer->PointerId, newTouch.position));
		break;
	case HQ_TOUCH_MOVED:
		if (oldTouchPosition == NULL)//not found
		{
			newTouch.prev_position = newTouch.position;
			//store the current position to the history table
			m_prevTouchPos.PushBack(TouchHistoryTableType::LinkedListItemType(pointer->PointerId, newTouch.position));

			newEvent.type = HQ_TOUCH_BEGAN;//send touch began event first then touch moved event

			hq_engine_eventQueue_internal->EndAddEvent();

			//begin add touch moved event
			HQEvent &newEvent2 = hq_engine_eventQueue_internal->BeginAddEvent();
			newEvent2 = newEvent;//copy data
			newEvent2.type = HQ_TOUCH_MOVED;
		}
		else
		{
			newTouch.prev_position = *oldTouchPosition;
			//change the previois position
			ite->m_second = newTouch.position;
		}

		break;
	case HQ_TOUCH_ENDED:
	case HQ_TOUCH_CANCELLED:
		if (oldTouchPosition == NULL)//not found
		{
			newTouch.prev_position = newTouch.position;

			newEvent.type = HQ_TOUCH_BEGAN;//send touch began event first then send touch ended event

			hq_engine_eventQueue_internal->EndAddEvent();

			//begin add touch ended event
			HQEvent &newEvent2 = hq_engine_eventQueue_internal->BeginAddEvent();
			newEvent2 = newEvent;//copy data
			newEvent2.type = type;
		}
		else
			newTouch.prev_position = *oldTouchPosition;

		if (oldTouchPosition != NULL)//remove the touch from history table
			m_prevTouchPos.RemoveAt(ite.GetNode());
		break;
	}
	

	//finish add event to game thread's event queue
	hq_engine_eventQueue_internal->EndAddEvent();
}

/*-------engine 's window class--------*/
//base class
HQEngineBaseWindow::HQEngineBaseWindow(const char* settingFileDir)
{
	//get window size on ui thead
	HQWinStoreUtil::RunOnUIThreadAndWait(
		hq_engine_coreWindow_internal->Dispatcher,
		[this]()
		{
			m_width = (hquint32)ConvertDipsToPixels(hq_engine_coreWindow_internal->Bounds.Width);
			m_height = (hquint32)ConvertDipsToPixels(hq_engine_coreWindow_internal->Bounds.Height);
		});
	
	m_windowed = true;//not relevant
}



HQEngineWindow::HQEngineWindow(const char *title, const char *settingFileDir ,  HQWIPPlatformSpecificType* args)
: HQEngineBaseWindow(settingFileDir)
{
	m_window = hq_engine_coreWindow_internal;
	
	Platform::Agile<CoreCursor> cursor = 
#ifdef HQ_WIN_STORE_PLATFORM
		(args == NULL)? hq_engine_defaultCursor_internal: args->cursor;
#else
		nullptr;
#endif

#ifdef HQ_WIN_STORE_PLATFORM
	//change window cursor
	HQWinStoreUtil::RunOnUIThread(
		hq_engine_coreWindow_internal->Dispatcher,
		[this, cursor]()
		{
			this->ChangeCursorOnUIThread(cursor);
		});
#endif

	//create window events handler
	m_windowEventHandler = ref new HQWindowsEventHandler(cursor.Get());
}

HQEngineWindow::~HQEngineWindow()
{
	if (HQAppInternalGetStatusFlag() != hq_internalAppExit)//no need to do this if app is exiting
		m_windowEventHandler->DetachFromWindow();
	m_windowEventHandler = nullptr;
}

void HQEngineWindow::ChangeCursorOnUIThread(Platform::Agile<Windows::UI::Core::CoreCursor> cursor)
{
	auto window = CoreWindow::GetForCurrentThread();

	window->PointerCursor = cursor.Get();
}

HQReturnVal HQEngineWindow::Show()
{
	return HQ_OK;
}