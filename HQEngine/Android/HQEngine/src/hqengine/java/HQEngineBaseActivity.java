package hqengine.java;

import java.io.File;
import java.nio.ByteBuffer;
import java.util.Timer;
import java.util.TimerTask;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLDisplay;

import android.app.Activity;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.View;
import android.view.View.OnTouchListener;

import android.view.OrientationEventListener;

public class HQEngineBaseActivity extends Activity implements OnTouchListener {
    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        //check sdcard
	  	String state = Environment.getExternalStorageState();
	
	  	if (Environment.MEDIA_MOUNTED.equals(state)) {
	  	    // We can read and write the media
	  	    externalStorageAvailable = externalStorageWriteable = true;
	  	} else if (Environment.MEDIA_MOUNTED_READ_ONLY.equals(state)) {
	  	    // We can only read the media
	  	    externalStorageAvailable = true;
	  	    externalStorageWriteable = false;
	  	} else {
	  	    // Something else is wrong. It may be one of many other states, but all we need
	  	    //  to know is we can neither read nor write
	  	    externalStorageAvailable = externalStorageWriteable = false;
	  	}
	  	
	  	if (externalStorageWriteable && externalStorageAvailable )
	  	{
	  		//create external resource directories
	  	
		  	File file = Environment.getExternalStorageDirectory();
		  	sdResourcePath = file.getAbsolutePath();
		  	sdResourcePath += "/Android/data/" + this.getPackageName() + "/files";
		  	File resourceDir = new File(sdResourcePath);
		  	@SuppressWarnings("unused")
			boolean _re;
		  	if (resourceDir.exists() == false)
		  	{
		  		_re = resourceDir.mkdirs();
		  	}
	  	}
        
        //create content view
        view = new HQEngineView(this);
        view.setOnTouchListener(this);
        
        setContentView(view);
        
        //create orientation event listener
        orientationListener = new OrientationEventListener(this, SensorManager.SENSOR_DELAY_GAME ) {
			
			@Override
			public void onOrientationChanged(int orientation) {
				
				if ((orientation >=0 && orientation < 45) || (orientation >= 315 && orientation < 360))
				{	
					if (currentOrient != down)
					{
						onChangedToPortraitNative();
						currentOrient = down;
					}
				}
				else if (orientation >= 45 && orientation < 135)
				{
					if (currentOrient != right)
					{
						onChangedToLandscapeRightNative();
						currentOrient = right;
					}
				}
				else if (orientation >= 135 && orientation < 225)
				{
					if (currentOrient != up)
					{
						onChangedToPortraitUpsideDownNative();
						currentOrient = up;
					}
				}
				else if (orientation >= 225 && orientation < 315)
				{
					if (currentOrient != left)
					{
						onChangedToLandscapeLeftNative();
						currentOrient = left;
					}
				}
				
			}
			
			private static final int unknown = -1;
			private static final int down = 0;
			private static final int up = 1;
			private static final int right = 2;
			private static final int left= 3;
			private int currentOrient = unknown;
			
		};

		if(sleepUIThread)
		{
			//create timer to sleep ui thread
			sleepUIThreadTimer = new Timer();
			
			sleepUIThreadTask = new TimerTask() {
				@Override
				public void run() {
					view.post(sleepUIThread);
				}
				
				private Runnable sleepUIThread = new Runnable() {
					
					@Override
					public void run() {
						Thread.yield();
					}
				};
			};
	        
			sleepUIThreadTimer.schedule(sleepUIThreadTask, 9, 9);
		}
		
		//create touch data native buffer
		touchDataNative = new HQEngineTouchDataNative();
		
        //create egl
		EGL10 egl = (EGL10)EGLContext.getEGL();
    	EGLDisplay glDisplay = egl.eglGetDisplay(EGL10.EGL_DEFAULT_DISPLAY);

    	int[] version = new int[2];
    	egl.eglInitialize(glDisplay, version);
    	
        //call native init function
        onCreateNative();//assign wrapper main
    	onCreateNativeInternal(view, egl, glDisplay);//start game thread
        
    }

	@Override
	public boolean onTouch(View _view, MotionEvent mt) {
		onTouchNative(touchDataNative.getTouchData(mt));
		return true;
	}
	
	@Override
	public boolean onKeyDown(int keyCode, KeyEvent event)
	{
		if (keyCode == KeyEvent.KEYCODE_BACK && event.getRepeatCount() == 0)
		{
			if (!onBackKeyPressedNative())
				return true;
		}
		return super.onKeyDown(keyCode, event);
	}
	
	@Override
	protected void onPause ()
	{
		orientationListener.disable();
		onPauseNative();
		super.onPause();
	}
	
	@Override
	protected void onResume ()
	{
		if(orientationListener.canDetectOrientation())
		{
			orientationListener.enable();
		}
		onResumeNative();
		super.onResume();
	}
	
	@Override
	protected void onDestroy ()
	{
		if(sleepUIThread)
		{
			orientationListener.disable();
			sleepUIThreadTask.cancel();
			sleepUIThreadTimer.cancel();
			sleepUIThreadTimer.purge();
			sleepUIThreadTimer = null;
		}
		onDestroyNative();
		
		touchDataNative = null;
		
		super.onDestroy();
		
		int pid = android.os.Process.myPid();

	    android.os.Process.killProcess(pid);
	}
	
	protected String getSDResourcePath()
	{
		return sdResourcePath;
	}
	
	protected boolean setSDPathCurrentDir()
	{
		if (externalStorageWriteable && externalStorageAvailable )
		{
			setCurrentDir(sdResourcePath);
			return true;
		}
		
		return false;
	}
	
	private native void setCurrentDir(String dir);
	
	private native void onTouchNative(ByteBuffer touchDataBuffer);
	private native boolean onBackKeyPressedNative();
	
	private native void onCreateNativeInternal(HQEngineView _view, EGL10 egl, EGLDisplay display);
	private native void onCreateNative();
	
	private native void onPauseNative();
	private native void onResumeNative();
	private native void onDestroyNative();
	private native void onChangedToPortraitNative();
	private native void onChangedToPortraitUpsideDownNative();
	private native void onChangedToLandscapeLeftNative();//right side is at top
	private native void onChangedToLandscapeRightNative();//left side is at top
	
	
	private HQEngineView view;
	private boolean externalStorageAvailable = false;
	private boolean externalStorageWriteable = false;
	private String sdResourcePath = "";
	private HQEngineTouchDataNative touchDataNative = null;
	private OrientationEventListener orientationListener = null;
	private TimerTask sleepUIThreadTask = null;
	private Timer sleepUIThreadTimer = null;
	private static final boolean sleepUIThread = false;
	
	static{
		System.loadLibrary("HQEngine");
	}

}
