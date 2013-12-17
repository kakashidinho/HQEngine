/*
Copyright (C) 2010-2013  Le Hoang Quyen (lehoangq@gmail.com)

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU General Public License as
published by the Free Software Foundation; either version 2 of the
License, or (at your option) any later version.  See the file
COPYING.txt included with this distribution for more information.


*/

#include "../stdafx.h"
#include "../HQEngineWindow.h"
#include "string.h"
#include <iostream>

#define HQ_WINDOW_STYLE (NSClosableWindowMask | NSMiniaturizableWindowMask | NSTitledWindowMask)

/*----------------window class--------------------*/
@interface HQMacWindow : NSWindow
{
@private
	BOOL m_destroyByClosing;//is window destroyed by closing
}

- (void) setDestroyByClosing;
- (BOOL) isDestroyByClosing;

@end

/*---------------window delegate class-----*/
@interface HQMacWindowDelegate : NSObject <NSWindowDelegate>
{
	
}

@end

@implementation HQMacWindowDelegate


- (BOOL)windowShouldClose:(id)sender
{
	if(HQEngineApp::GetInstance()->GetWindowListener()->WindowClosing())
		return YES;
	return NO;
}

- (void)windowWillClose:(NSNotification *)notification
{
	HQEngineApp::GetInstance()->Stop();
	
	[[notification object] setDestroyByClosing];//tell window object that it will be destroyed by closing
}

@end

/*--------------window class implementation------*/


@implementation HQMacWindow

- (void) setDestroyByClosing
{
	m_destroyByClosing = YES;
}

- (BOOL) isDestroyByClosing
{
	return m_destroyByClosing;
}

- (id)initWithContentRect:(NSRect)contentRect 
				styleMask:(NSUInteger)windowStyle 
				  backing:(NSBackingStoreType)bufferingType 
					defer:(BOOL)deferCreation
{
	if ((self = [super initWithContentRect: contentRect 
								 styleMask: windowStyle
								   backing: bufferingType
									 defer: deferCreation]) != nil) 
	{
		m_destroyByClosing = NO;
		HQMacWindowDelegate * l_delegate = [[HQMacWindowDelegate alloc] init];
		[self setDelegate: l_delegate];
	}
	return self;
}

- (BOOL) canBecomeKeyWindow
{
	return YES;
}

- (void)dealloc
{
	
    [self setDelegate: nil];
    [super dealloc];
}

@end

/*---------------view class-----------------------*/
static BOOL g_oldShiftFlag = NO;
static BOOL g_oldCtrlFlag = NO;

@interface HQMacView : NSView
{
}

@end

@implementation HQMacView

- (id)initWithFrame:(NSRect)frameRect 
{
	if ((self = [super initWithFrame:frameRect]) != nil) 
	{
	}
	return self;
}

- (BOOL)acceptsFirstResponder {
    [[self window] makeFirstResponder:self];
    return YES;
}

- (BOOL)isFlipped
{
	return YES;
}

-(void)getMousePoint: (HQPointi*)engine_point fromEvent:(NSEvent*) theEvent 
{
	if (HQEngineApp::GetInstance()->IsMouseCursorEnabled())
	{
		NSPoint event_location = [theEvent locationInWindow];
		
		NSPoint local_point = [self convertPoint:event_location fromView:nil];
		
		engine_point->x = (hq_int32)local_point.x ;
		engine_point->y = (hq_int32) local_point.y;
		
	}
	else {//getting delta info
		engine_point->x = (hq_int32)[theEvent deltaX] ;
		engine_point->y = (hq_int32)[theEvent deltaY];
	}
}

-(void)keyDown:(NSEvent *)theEvent {
	if ([theEvent isARepeat] == NO)
		HQEngineApp::GetInstance()->GetKeyListener()->KeyPressed((HQKeyCodeType)[theEvent keyCode]);
    [super keyDown:theEvent];
}

-(void)keyUp:(NSEvent *)theEvent {
	HQEngineApp::GetInstance()->GetKeyListener()->KeyReleased((HQKeyCodeType)[theEvent keyCode]);
    [super keyUp:theEvent];
}

/*----shift & control------*/
- (void)flagsChanged:(NSEvent *)theEvent {
	BOOL shiftFlag = ([NSEvent modifierFlags] & NSShiftKeyMask) != 0;
	BOOL ctrlFlag = ([NSEvent modifierFlags] & NSControlKeyMask) != 0;
	
	HQKeyCodeType keyCode = (HQKeyCodeType)[theEvent keyCode];
	
	if (shiftFlag != g_oldShiftFlag)//shift key state changed
	{
		if (g_oldShiftFlag == NO)//shift key pressed
		{
			HQEngineApp::GetInstance()->GetKeyListener()->KeyPressed(keyCode);
		}
		else {//released
			HQEngineApp::GetInstance()->GetKeyListener()->KeyReleased(keyCode);
		}
		
		g_oldShiftFlag = shiftFlag;

	}
	else if (ctrlFlag != g_oldCtrlFlag)//control key state changed
	{
		if (g_oldCtrlFlag == NO)//control key pressed
		{
			HQEngineApp::GetInstance()->GetKeyListener()->KeyPressed(keyCode);
		}
		else {//released
			HQEngineApp::GetInstance()->GetKeyListener()->KeyReleased(keyCode);
		}
		
		g_oldCtrlFlag = ctrlFlag;
	}
	
	
	[super flagsChanged:theEvent];
}

//mouse moved with/without press any button
- (void) uniMouseMoved : (NSEvent *)theEvent
{
	HQPointi point;
	[self getMousePoint: &point fromEvent: theEvent];
	
	HQEngineApp::GetInstance()->GetMouseListener()->MouseMove(point);
}


//left mouse down
- (void)mouseDown:(NSEvent *)theEvent
{
	HQPointi point;
	[self getMousePoint: &point fromEvent: theEvent];
	HQEngineApp::GetInstance()->GetMouseListener()->MousePressed(HQKeyCode::LBUTTON, point );
	[super mouseDown: theEvent];
}

//mouse moved when left clicked
- (void)mouseDragged:(NSEvent *)theEvent
{
	[self uniMouseMoved: theEvent];
	[super mouseDragged: theEvent];
}

//left mouse up
- (void)mouseUp:(NSEvent *)theEvent
{
	HQPointi point;
	[self getMousePoint: &point fromEvent: theEvent];
	HQEngineApp::GetInstance()->GetMouseListener()->MouseReleased(HQKeyCode::LBUTTON, point );
	[super mouseUp: theEvent];
}

//right mouse down
- (void)rightMouseDown:(NSEvent *)theEvent
{
	HQPointi point;
	[self getMousePoint: &point fromEvent: theEvent];
	HQEngineApp::GetInstance()->GetMouseListener()->MousePressed(HQKeyCode::RBUTTON, point );
	[super rightMouseDown: theEvent];
}

//mouse moved when right clicked
- (void)rightMouseDragged:(NSEvent *)theEvent
{
	[self uniMouseMoved: theEvent];
	[super rightMouseDragged: theEvent];
}

//right mouse up
- (void)rightMouseUp:(NSEvent *)theEvent
{
	HQPointi point;
	[self getMousePoint: &point fromEvent: theEvent];
	HQEngineApp::GetInstance()->GetMouseListener()->MouseReleased(HQKeyCode::RBUTTON, point );
	[super rightMouseUp: theEvent];
}

//middle mouse down
- (void)otherMouseDown:(NSEvent *)theEvent
{
	HQPointi point;
	[self getMousePoint: &point fromEvent: theEvent];
	HQEngineApp::GetInstance()->GetMouseListener()->MousePressed(HQKeyCode::MBUTTON, point );
	[super otherMouseDown: theEvent];
}
//mouse moved when middle clicked
- (void)otherMouseDragged:(NSEvent *)theEvent
{
	[self uniMouseMoved: theEvent];
	[super otherMouseDragged: theEvent];
}
//middle mouse up
- (void)otherMouseUp:(NSEvent *)theEvent
{
	HQPointi point;
	[self getMousePoint: &point fromEvent: theEvent];
	HQEngineApp::GetInstance()->GetMouseListener()->MouseReleased(HQKeyCode::MBUTTON, point );
	[super otherMouseUp: theEvent];
}

//scroll wheel
- (void)scrollWheel:(NSEvent *)theEvent
{
	HQPointi point;
	[self getMousePoint: &point fromEvent: theEvent];
	CGFloat scrollAmount = [theEvent deltaY];
	HQEngineApp::GetInstance()->GetMouseListener()->MouseWheel((hqfloat32) scrollAmount, point);
	
	[super scrollWheel: theEvent];
}

//mouse moved without press any button
- (void)mouseMoved:(NSEvent *)theEvent
{
	[self uniMouseMoved: theEvent];
	[super mouseMoved: theEvent];
}

@end


/*-----------------------engine's window class---------------------------*/
HQEngineWindow::HQEngineWindow(const char *title, const char *settingFileDir ,  HQWIPPlatformSpecificType* icon)
: HQEngineBaseWindow(settingFileDir)
{
	/*-------create window---------*/
	NSRect rect = NSMakeRect(0 , 0 , this->GetWidth() , this->GetHeight() );
	if (this->IsWindowed())
	{
		m_window = [[HQMacWindow alloc] initWithContentRect: rect 
										styleMask: HQ_WINDOW_STYLE
										backing:NSBackingStoreBuffered
										defer:NO];
	}
	else//fullscreen
	{
		m_window = [[HQMacWindow alloc] initWithContentRect: rect 
										styleMask:NSBorderlessWindowMask 
										backing:NSBackingStoreBuffered
										defer:NO];
		
		[m_window setLevel:NSMainMenuWindowLevel+1];
		[m_window setOpaque:YES]; 
		[m_window setHidesOnDeactivate:YES];
	}
	
	HQMacView * l_view = [[HQMacView alloc] initWithFrame: rect];
	[m_window setContentView: l_view];
	
	/*----input for render device Init() method----*/
	m_viewInfo.nsView = l_view;
	m_viewInfo.isWindowed = this->IsWindowed();
	
	/*-----accept mouse movement event----*/
	[m_window setAcceptsMouseMovedEvents : YES];
	
	/*-----title-------*/
	NSString * nsTitle =  [[NSString alloc] initWithCString: title encoding:NSASCIIStringEncoding];
	
	[m_window setTitle: nsTitle];
	
	[nsTitle release];
	
	if (icon != NULL)
		[NSApp setApplicationIconImage: icon->icon];
}

HQEngineWindow::~HQEngineWindow()
{
	if([(HQMacWindow*)m_window isDestroyByClosing] == YES)//window will be auto released
	{
		HQEngineApp::GetInstance()->GetWindowListener()->WindowClosed();
	}
	else
		[m_window release];
}

HQReturnVal HQEngineWindow::Show()
{
	[m_window makeKeyAndOrderFront: [m_window contentView] ];
	

	return HQ_OK;
}
