package hqengine.java;

import android.os.Handler;
import android.os.Looper;

class HQEngineUIRunnable implements java.lang.Runnable {

	private HQEngineUIRunnable(long _nativeFuncAddress, boolean _block)
	{
		this.nativeFuncAddress = _nativeFuncAddress;
		this.block = _block;
	}
	
	@Override
	public void run() {
		runNative(nativeFuncAddress);
		
		this.block = false;
	}

	private static void registerNativeFunc(long funcAddress, boolean _block)
	{
		HQEngineUIRunnable runnable = new HQEngineUIRunnable(funcAddress, _block);
		
		//if this is ui thread
		if (Thread.currentThread() == Looper.getMainLooper().getThread())
		{
			//call immediately
			runnable.run();
		}
		else
		{
			handler.post(runnable);
			
			while (runnable.block)
			{
				Thread.yield();
			}
		}
	}
	
	private native void runNative(long nativeFuncAddress);
	private native static void initNative();
	
	private static Handler handler;
	private long nativeFuncAddress;
	private volatile boolean block;
	
	public static void init()
	{
        handler = new Handler(Looper.getMainLooper());
        
        initNative();
	}

}
