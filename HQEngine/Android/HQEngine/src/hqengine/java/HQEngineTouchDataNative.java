package hqengine.java;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashMap;

import android.view.MotionEvent;

class HQEngineTouchDataNative {
	public HQEngineTouchDataNative()
	{
		nativeBuffer = ByteBuffer.allocateDirect(getBufferSize());
		if (isLittleEndian())
			nativeBuffer.order(ByteOrder.LITTLE_ENDIAN);
		else
			nativeBuffer.order(ByteOrder.BIG_ENDIAN);
		
	}
	
	
	public ByteBuffer getTouchData(MotionEvent mt)
	{
		
		int action = mt.getAction();
		int pointers = Math.min(mt.getPointerCount(), MAX_MULTITOUCHES);
		int actionIndex;
		int actionMask = action & MotionEvent.ACTION_MASK;
		
		switch(actionMask)
		{
		case MotionEvent.ACTION_DOWN:
		
			this.setType(HQ_TOUCH_BEGAN);
			this.setNumTouches(pointers);
			
			for (int i = 0; i < pointers; ++i)
			{
				historyTouches.put(mt.getPointerId(i), new PrevPosition(mt.getX(i), mt.getY(i)));
				this.setTouchData(i, mt.getPointerId(i), mt.getX(i), mt.getY(i));
			}
			
			break;
		case MotionEvent.ACTION_POINTER_DOWN:
			
			actionIndex = GetActionIndex(action);
			this.setType(HQ_TOUCH_BEGAN);
			this.setNumTouches(1);
			
			historyTouches.put(mt.getPointerId(actionIndex), new PrevPosition(mt.getX(actionIndex), mt.getY(actionIndex)));
			this.setTouchData(0, mt.getPointerId(actionIndex), mt.getX(actionIndex), mt.getY(actionIndex));
			break;
			
		case MotionEvent.ACTION_UP:
			this.setType(HQ_TOUCH_ENDED);
			this.setNumTouches(pointers);
			
			for (int i = 0; i < pointers; ++i)
			{
				this.setTouchData(i, mt.getPointerId(i), mt.getX(i), mt.getY(i));
				
				//get last position
				PrevPosition prevPos = historyTouches.get(mt.getPointerId(i));
				this.setTouchDataPrevPos(i, prevPos.x, prevPos.y);
				
				//remove history touch tracking
				historyTouches.remove(mt.getPointerId(i));
			}
			break;
		case MotionEvent.ACTION_POINTER_UP:
			{
				actionIndex = GetActionIndex(action);
				this.setType(HQ_TOUCH_ENDED);
				this.setNumTouches(1);
				
				this.setTouchData(0, mt.getPointerId(actionIndex), mt.getX(actionIndex), mt.getY(actionIndex));
				
				//get last position
				PrevPosition prevPos = historyTouches.get(mt.getPointerId(actionIndex));
				this.setTouchDataPrevPos(actionIndex, prevPos.x, prevPos.y);
				
				//remove history touch tracking
				historyTouches.remove(mt.getPointerId(actionIndex));
			}
			
			break;
		case MotionEvent.ACTION_MOVE:
			this.setType(HQ_TOUCH_MOVED);
			this.setNumTouches(pointers);
			
			for (int i = 0; i < pointers; ++i)
			{
				this.setTouchData(i, mt.getPointerId(i), mt.getX(i), mt.getY(i));
				
				//get last position
				PrevPosition prevPos = historyTouches.get(mt.getPointerId(i));
				this.setTouchDataPrevPos(i, prevPos.x, prevPos.y);
				
				//update last position
				prevPos.x = mt.getX(i);
				prevPos.y = mt.getY(i);
				
			}
			break;
		case MotionEvent.ACTION_CANCEL:
		
			this.setType(HQ_TOUCH_CANCELLED);
			this.setNumTouches(pointers);
			
			for (int i = 0; i < pointers; ++i)
			{
				this.setTouchData(i, mt.getPointerId(i), mt.getX(i), mt.getY(i));
				
				//get last position
				PrevPosition prevPos = historyTouches.get(mt.getPointerId(i));
				this.setTouchDataPrevPos(i, prevPos.x, prevPos.y);
				
				//remove history touch tracking
				historyTouches.remove(mt.getPointerId(i));
			}
			break;
		}
		return nativeBuffer;
	}
	
	private void setType(int type)
	{
		nativeBuffer.putInt(touchEventTypeOffset, type);
	}
	
	private void setNumTouches(int count)
	{
		nativeBuffer.putInt(MAX_MULTITOUCHES * singleTouchSize, count);
	}
	
	private void setTouchID(int index , int id)
	{
		nativeBuffer.putInt(singleTouchSize * index, id);
	}
	
	private void setTouchDataPos(int index, float x, float y)
	{
		int startIndex = singleTouchSize * index;
		nativeBuffer.putFloat(startIndex + 4, x);
		nativeBuffer.putFloat(startIndex + 8, y);
	}
	
	private void setTouchDataPrevPos(int index, float x, float y)
	{
		int startIndex = singleTouchSize * index;
		nativeBuffer.putFloat(startIndex + 12, x);
		nativeBuffer.putFloat(startIndex + 16, y);
	}
	
	private void setTouchData(int index, int id, float x, float y)
	{
		int startIndex = singleTouchSize * index;
		nativeBuffer.putInt(startIndex, id);
		
		nativeBuffer.putFloat(startIndex + 4, x);
		nativeBuffer.putFloat(startIndex + 8, y);
	}
	
	private void setTouchData(int index, int id, float x, float y, float prevX, float prevY)
	{
		int startIndex = singleTouchSize * index;
		nativeBuffer.putInt(startIndex, id);
		
		nativeBuffer.putFloat(startIndex + 4, x);
		nativeBuffer.putFloat(startIndex + 8, y);
		
		nativeBuffer.putFloat(startIndex + 12, prevX);
		nativeBuffer.putFloat(startIndex + 16, prevY);
	}
	
	
	private static int GetActionIndex(int action)
	{
		return (action & MotionEvent.ACTION_POINTER_ID_MASK) >>> MotionEvent.ACTION_POINTER_ID_SHIFT;
	}	

	private static native boolean isLittleEndian();
	private static native int getBufferSize();
	private static native int getTouchEventTypeOffset();
	private static native int getSingleTouchSize();
	private static native int getMaxMultiTouches();
	private static native int getTouchBeganTypeVal();
	private static native int getTouchMovedTypeVal();
	private static native int getTouchEndedTypeVal();
	private static native int getTouchCancelledTypeVal();
	
	public static final int HQ_TOUCH_BEGAN ;
	public static final int HQ_TOUCH_MOVED ;
	public static final int HQ_TOUCH_ENDED ;
	public static final int HQ_TOUCH_CANCELLED ;
	public static final int MAX_MULTITOUCHES;
	private static final int touchEventTypeOffset;
	private static final int singleTouchSize;
	
	private ByteBuffer nativeBuffer;
	private HashMap<Integer, PrevPosition> historyTouches = new HashMap<Integer, PrevPosition>();
	
	static
	{
		HQ_TOUCH_BEGAN = getTouchBeganTypeVal();
		HQ_TOUCH_MOVED = getTouchMovedTypeVal();
		HQ_TOUCH_ENDED = getTouchEndedTypeVal();
		HQ_TOUCH_CANCELLED = getTouchCancelledTypeVal();
		
		MAX_MULTITOUCHES = getMaxMultiTouches();
		
		touchEventTypeOffset = getTouchEventTypeOffset();
		singleTouchSize = getSingleTouchSize();
	}
	
	private class PrevPosition
	{
		public PrevPosition(float _x, float _y)
		{
			x = _x;
			y = _y;
		}
		
		public float x;
		public float y;
	}
}
