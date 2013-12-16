package hqengineTest.test;

import android.os.Bundle;
import android.os.Environment;
import hqengine.java.HQEngineBaseActivity;
import java.io.File;
import java.io.FileOutputStream;
import org.apache.commons.net.ftp.*;



public class TestActivity extends HQEngineBaseActivity {
	@Override
	public void onCreate(Bundle savedInstanceState) {
        //check sdcard
		boolean mExternalStorageAvailable = false;
    	boolean mExternalStorageWriteable = false;
    	String state = Environment.getExternalStorageState();

    	if (Environment.MEDIA_MOUNTED.equals(state)) {
    	    // We can read and write the media
    	    mExternalStorageAvailable = mExternalStorageWriteable = true;
    	} else if (Environment.MEDIA_MOUNTED_READ_ONLY.equals(state)) {
    	    // We can only read the media
    	    mExternalStorageAvailable = true;
    	    mExternalStorageWriteable = false;
    	} else {
    	    // Something else is wrong. It may be one of many other states, but all we need
    	    //  to know is we can neither read nor write
    	    mExternalStorageAvailable = mExternalStorageWriteable = false;
    	}
    	
    	if (mExternalStorageWriteable == false || mExternalStorageAvailable == false)
    		System.exit(-1);

    	//create resource directories
    	
    	File file = Environment.getExternalStorageDirectory();
    	String resourcePath = file.getAbsolutePath();
    	resourcePath += "/Android/data/hqengineTest.test/files";
    	File resourceDir = new File(resourcePath);
    	@SuppressWarnings("unused")
		boolean _re;
    	if (resourceDir.exists() == false)
    	{
    		_re = resourceDir.mkdirs();
    	}
    	
    	File resourceSubDir;
    	
    	resourceSubDir = new File(resourcePath + "/test/audio");
    	if (resourceSubDir.exists() == false)
    		resourceSubDir.mkdirs();
    	
    	resourceSubDir = new File(resourcePath + "/test/image");
    	if (resourceSubDir.exists() == false)
    		resourceSubDir.mkdirs();
    	
    	resourceSubDir = new File(resourcePath + "/test/Font");
    	if (resourceSubDir.exists() == false)
    		resourceSubDir.mkdirs();
    	
    	resourceSubDir = new File(resourcePath + "/test/shader");
    	if (resourceSubDir.exists() == false)
    		resourceSubDir.mkdirs();
    	
    	resourceSubDir = new File(resourcePath + "/test/setting");
    	if (resourceSubDir.exists() == false)
    		resourceSubDir.mkdirs();
    	
    	resourceSubDir = new File(resourcePath + "/test/log");
    	if (resourceSubDir.exists() == false)
    		resourceSubDir.mkdirs();
    	
        //download resources
    	/*
    	try{
        	ftpClient = new FTPClient();
        	ftpClient.connect("ftp.drivehq.com");
        	
        	if (FTPReply.isPositiveCompletion(ftpClient.getReplyCode())) {
        	
        		if (ftpClient.login("kakashidinho", "mtht621989"))
        		{
        			ftpClient.setFileType(FTP.BINARY_FILE_TYPE);
        			ftpClient.enterLocalPassiveMode();
                    
        			//download resources
        			String urlPath = "/HQEngine/testResources/";
        			//audios
        			String fileName = "audio/battletoads-double-dragons-2.ogg";
        			downloadFromURL(urlPath + fileName, resourcePath + "/test/" + fileName);
        			
        			//images
        			fileName = "image/Marine.dds";
        			downloadFromURL(urlPath + fileName, resourcePath + "/test/" + fileName);
        			
        			fileName = "image/MarineFlip.pvr";
        			downloadFromURL(urlPath + fileName, resourcePath + "/test/" + fileName);
        			
        			fileName = "image/pen2.png";
        			downloadFromURL(urlPath + fileName, resourcePath + "/test/" + fileName);
        			
        			fileName = "image/pen2.jpg";
        			downloadFromURL(urlPath + fileName, resourcePath + "/test/" + fileName);
        			
        			fileName = "image/metall16bit.bmp";
        			downloadFromURL(urlPath + fileName, resourcePath + "/test/" + fileName);
        			
        			//fileName = "image/skyboxPVRTC2.pvr";
        			//downloadFromURL(urlPath + fileName, resourcePath + "/test/" + fileName);
        			
        			//shaders
        			fileName = "shader/vs.txt";
        			downloadFromURL(urlPath + fileName, resourcePath + "/test/" + fileName, true);
        			
        			fileName = "shader/ps.txt";
        			downloadFromURL(urlPath + fileName, resourcePath + "/test/" + fileName);
  
        			//setting
        			fileName = "setting/Setting.txt";
        			downloadFromURL(urlPath + fileName, resourcePath + "/test/" + fileName);
        			
        			ftpClient.logout();
        		}
        		
        		ftpClient.disconnect();
        	}
        	
        	ftpClient = null;
    	}
    	catch (Exception e)
    	{
    		e.printStackTrace();
    	}
		*/
        //call native init function
        onCreateNative(resourcePath);
        //super method
        super.onCreate(savedInstanceState);

	}
	
	private void downloadFromURL(String urlString, String fileName, boolean redownload) throws Exception
	{
    	File file = new File(fileName);
    	
    	if (!file.exists() || redownload)
    	{
        	
        	FileOutputStream fo = new FileOutputStream(file);
        	
        	ftpClient.retrieveFile(urlString, fo);
        	
        	fo.close();
    	}
	}
	
	private void downloadFromURL(String urlString, String fileName) throws Exception
	{
		downloadFromURL(urlString, fileName, false);
	}
	
	@Override 
	protected void onSaveInstanceState(Bundle outState) 
	{
		
		int a = 0;
	};

	
	private native void onCreateNative(String resourcePath);
	
	private FTPClient ftpClient = null;
	
    static{
    	System.loadLibrary("HQAudio");
    	System.loadLibrary("HQSceneManagement");
    	System.loadLibrary("test");
    }
}