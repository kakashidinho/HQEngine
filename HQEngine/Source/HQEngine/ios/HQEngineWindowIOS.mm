/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the MIT license.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../HQEnginePCH.h"
#include "../HQEngineWindow.h"
#include "../HQEventSeparateThread.h"

#include "string.h"
#include <iostream>

extern const int hq_internalAppExit ;

extern HQEventQueue* hq_engine_eventQueue_internal;
extern bool HQAppInternalGetStatusFlag();


/*---view controller class----*/
@interface HQIOSViewController : UIViewController
{
@private
	UIInterfaceOrientation m_supportOrientation;
	UIDeviceOrientation m_lastRequestOrientation;
}

- (id) initWithOrientation: (bool) landscapeMode;
- (void) initOnMainThread;

@end

@implementation HQIOSViewController

- (id) initWithOrientation: (bool) landscapeMode
{
	if ((self = [super init]) != nil)
	{
		if (landscapeMode)
		{
			self->m_supportOrientation = UIInterfaceOrientationLandscapeRight;
		}
		else
		{
			self->m_supportOrientation = UIInterfaceOrientationPortrait;
		}
		
		
		[self performSelectorOnMainThread:@selector(initOnMainThread) 
										   withObject:nil 
										waitUntilDone:TRUE];
		
		
	}
	
	return self;
}

- (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)_interfaceOrientation
{
	return _interfaceOrientation == self->m_supportOrientation;
}

- (void)deviceOrientationDidChange:(NSNotification *)notification 
{
	
	UIDeviceOrientation orientation = [[UIDevice currentDevice] orientation];
	
	if ((orientation == UIDeviceOrientationLandscapeLeft || 
		 orientation == UIDeviceOrientationLandscapeRight ||
		 orientation == UIDeviceOrientationPortrait ||
		 orientation == UIDeviceOrientationPortraitUpsideDown) &&
		self->m_lastRequestOrientation !=  orientation)
	{
		self->m_lastRequestOrientation =  orientation;
		
		//add orientation changed event
		HQEventQueue::ScopeLock sl(hq_engine_eventQueue_internal);
		HQEvent &newEvent = hq_engine_eventQueue_internal->BeginAddEvent();
		
		switch (orientation)
		{
			case UIDeviceOrientationPortrait:
				newEvent.type = HQ_ORIENTATION_PORTRAIT;
				break;
			case UIDeviceOrientationPortraitUpsideDown:
				newEvent.type = HQ_ORIENTATION_PORTRAIT_UPSIDE_DOWN;
				break;
			case UIDeviceOrientationLandscapeRight:
				newEvent.type = HQ_ORIENTATION_LANDSCAPE_RIGHT;
				break;
			case UIDeviceOrientationLandscapeLeft:
				newEvent.type = HQ_ORIENTATION_LANDSCAPE_LEFT;
				break;
		}
		hq_engine_eventQueue_internal->EndAddEvent();
	}
}

- (void) initOnMainThread
{
	//listen to device orientation change
	UIDeviceOrientation orientation = [[UIDevice currentDevice] orientation];
	[[NSNotificationCenter defaultCenter] addObserver: self 
											 selector: @selector(deviceOrientationDidChange:) 
												 name: UIDeviceOrientationDidChangeNotification 
											   object: nil];
	
	switch (orientation)
	{
		case UIDeviceOrientationPortraitUpsideDown:
		case UIDeviceOrientationLandscapeLeft:
		case UIDeviceOrientationLandscapeRight:
			self->m_lastRequestOrientation = orientation;
			break;
		default:
			self->m_lastRequestOrientation = UIDeviceOrientationPortrait;
			break;
	}
}

- (void) deallocOnMainThread
{
	[[NSNotificationCenter defaultCenter] removeObserver: self 
											name: UIDeviceOrientationDidChangeNotification 
											   object: nil];
}

- (void) dealloc
{
	if (HQAppInternalGetStatusFlag() != hq_internalAppExit)//only perform this task when application is not in exiting state
	{
		[self performSelectorOnMainThread:@selector(deallocOnMainThread) 
							   withObject:nil 
							waitUntilDone:TRUE];
	}
	
	[super dealloc];
}




@end


/*-----view class--------*/
@interface HQIOSView : UIView
{
}

- (void) addToMainWindow;
- (void) removeFromMainWindow;
- (void) makeVisible;

@end

@implementation HQIOSView

- (id)initWithFrame:(CGRect)frameRect 
{
	if ((self = [super initWithFrame:frameRect]) != nil) 
	{
		self.multipleTouchEnabled = YES;
	}
	return self;
}

+ (Class)layerClass {
	return [CAEAGLLayer class];
}

- (BOOL)canBecomeFirstResponder {
    return YES;
}

- (void)getTouchData: (HQTouchData*) touchDataOut fromUITouchArray:(NSSet *)touches
{
	touchDataOut->numTouches = 0;
	
	for ( UITouch * touch in touches )
	{
		touchDataOut->touches[touchDataOut->numTouches].touchID = (hqint32) touch;
		
		CGPoint point = [touch locationInView : self];
		touchDataOut->touches[touchDataOut->numTouches].position.x = point.x;
		touchDataOut->touches[touchDataOut->numTouches].position.y = point.y;
		
		point = [touch previousLocationInView : self];
		touchDataOut->touches[touchDataOut->numTouches].prev_position.x = point.x;
		touchDataOut->touches[touchDataOut->numTouches].prev_position.y = point.y;
		
		++touchDataOut->numTouches;
		
		if (touchDataOut->numTouches == MAX_MULTITOUCHES)
			break;
	}
}

- (void)touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event
{
	HQEventQueue::ScopeLock sl(hq_engine_eventQueue_internal);
	HQEvent &newEvent = hq_engine_eventQueue_internal->BeginAddEvent();
	
	newEvent.type = HQ_TOUCH_BEGAN;
	
	[self getTouchData: &newEvent.touchData fromUITouchArray: touches];
	
	hq_engine_eventQueue_internal->EndAddEvent();
}

- (void)touchesCancelled:(NSSet *)touches withEvent:(UIEvent *)event
{
	HQEventQueue::ScopeLock sl(hq_engine_eventQueue_internal);
	HQEvent &newEvent = hq_engine_eventQueue_internal->BeginAddEvent();
	
	newEvent.type = HQ_TOUCH_CANCELLED;
	
	[self getTouchData: &newEvent.touchData fromUITouchArray: touches];
	
	hq_engine_eventQueue_internal->EndAddEvent();
}

- (void)touchesMoved:(NSSet *)touches withEvent:(UIEvent *)event
{
	HQEventQueue::ScopeLock sl(hq_engine_eventQueue_internal);
	HQEvent &newEvent = hq_engine_eventQueue_internal->BeginAddEvent();
	
	newEvent.type = HQ_TOUCH_MOVED;
		
	[self getTouchData: &newEvent.touchData fromUITouchArray: touches];
		
	hq_engine_eventQueue_internal->EndAddEvent();
}

- (void)touchesEnded:(NSSet *)touches withEvent:(UIEvent *)event
{
	HQEventQueue::ScopeLock sl(hq_engine_eventQueue_internal);
	HQEvent &newEvent = hq_engine_eventQueue_internal->BeginAddEvent();
	
	newEvent.type = HQ_TOUCH_ENDED;
	
	[self getTouchData: &newEvent.touchData fromUITouchArray: touches];
	
	hq_engine_eventQueue_internal->EndAddEvent();
}

- (void) addToMainWindow
{
	UIWindow * l_mainWindow = [[UIApplication sharedApplication] keyWindow];
	[l_mainWindow addSubview: self];
	[super becomeFirstResponder];
}

- (void) removeFromMainWindow
{
	[self removeFromSuperview];
}

- (void) makeVisible
{
	UIWindow * l_mainWindow = [[UIApplication sharedApplication] keyWindow];
	[l_mainWindow makeKeyAndVisible];
}


@end



/*-----------------------engine's window class---------------------------*/

//base class


HQEngineBaseWindow::HQEngineBaseWindow(const char* settingFileDir)
{
	UIScreen * l_mainScreen = [UIScreen mainScreen];
	CGRect screenRect = [l_mainScreen bounds];

	m_width = screenRect.size.width;
	m_height = screenRect.size.height;
	
	m_windowed = false;
}

HQEngineWindow::HQEngineWindow(const char *title, const char *settingFileDir ,  HQWIPPlatformSpecificType* additionalSettings)
: HQEngineBaseWindow(settingFileDir)
{
	/*-------create view---------*/
	CGRect rect = CGRectMake(0 , 0 , this->GetWidth() , this->GetHeight() );
	
	bool landscapeMode = false;
	if (additionalSettings != NULL)
		landscapeMode = additionalSettings->landscapeMode;
		
	m_view = [[HQIOSView alloc] initWithFrame: rect];
	//view controller
	m_viewController = [[HQIOSViewController alloc] initWithOrientation:landscapeMode];
	m_viewController.view = m_view;
	
	//add view to main window
	[m_view performSelectorOnMainThread:@selector(addToMainWindow) withObject:nil waitUntilDone:YES];
	
	/*--------------*/
	m_renderDeviceInitInfo.eaglLayer = (CAEAGLLayer*)[m_view layer];
	m_renderDeviceInitInfo.landscapeMode = landscapeMode;
}

HQEngineWindow::~HQEngineWindow()
{
	if (HQAppInternalGetStatusFlag() != hq_internalAppExit)//only perform this task when application is not in exiting state
	{
		[m_view performSelectorOnMainThread:@selector(removeFromMainWindow) withObject:nil waitUntilDone:YES];
	}
	[m_view release];
	[m_viewController release];
}

HQRenderDeviceInitInput HQEngineWindow::GetRenderDeviceInitInput() 
{
	return &m_renderDeviceInitInfo;
}

HQReturnVal HQEngineWindow::Show()
{
	[m_view performSelectorOnMainThread:@selector(makeVisible) withObject:nil waitUntilDone:YES];

	return HQ_OK;
}
