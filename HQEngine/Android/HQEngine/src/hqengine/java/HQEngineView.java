package hqengine.java;


import android.content.Context;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

public class HQEngineView extends SurfaceView implements SurfaceHolder.Callback{

	public HQEngineView(Context context) {
		super(context);
		
		this.getHolder().addCallback(this);
		
        
        this.setVisibility(INVISIBLE);
        
	}

	
	@Override
	public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
		
	}

	@Override
	public void surfaceCreated(SurfaceHolder holder) {
		surfaceCreatedNative();
	}

	@Override
	public void surfaceDestroyed(SurfaceHolder arg0) {
		surfaceDestroyedNative();
	}
	
	void isHQEngineViewClass()
	{
		
	}

	
	private native void surfaceCreatedNative();
	
	private native void surfaceDestroyedNative();
	
	//force HQEngineUIRunnable class to be loaded in virtual machine
	static {
		HQEngineUIRunnable.init();
	}
	
}
