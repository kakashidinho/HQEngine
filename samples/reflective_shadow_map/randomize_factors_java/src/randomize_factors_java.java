import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.Random;


public class randomize_factors_java {

	/**
	 * @param args
	 */
	public static void main(String[] args) throws IOException {
		if (args.length < 4 ){
			System.out.println("usage: <random seed> <width> <height> <output file name> <optional output text file name>\n");
			return ;
		}
		//uniformly randomize 2 factors and encode in 32 bit TGA file
		int width, height, seed;
		String fileName = args[3];
		String textFile = null;
		
		seed = Integer.parseInt(args[0]);
		width = Integer.parseInt(args[1]);
		height = Integer.parseInt(args[2]);
		
		if (args.length >=5 )
			textFile = args[4];
		
		Random random = new Random(seed);

		//write to a TGA file and optional text file
		BufferedWriter bw = null;
		BufferedOutputStream bo = new BufferedOutputStream(new FileOutputStream(fileName));
		
		bo.write(0);/* id length  */
		bo.write(0);/* color map type */
		bo.write(2);/* uncompressed RGB */
		bo.write(0); bo.write(0);//color map table offset
		bo.write(0); bo.write(0);//number of color map entries
		bo.write(0);//number of bits per pixel in color map
		bo.write(0); bo.write(0);/* X origin */
		bo.write(0); bo.write(0);/* y origin */
		bo.write((width & 0x00FF));
		bo.write((width & 0xFF00) >>> 8);
		bo.write((height & 0x00FF));
		bo.write((height & 0xFF00) >>> 8);
		bo.write(32); /* 32 bit bitmap */
		bo.write(2 << 4); //image descriptor. top left origin
		   
		if (textFile != null)
			bw = new BufferedWriter(new OutputStreamWriter( new FileOutputStream( new File(textFile))));
		
		final int dS = 65535 / (width * height);
		for(int i = 0; i < height; i++)
		{
			for(int j = 0; j < width; j++)
			{
				int S1 = random.nextInt(width * height) * dS;
				int S2 = random.nextInt(width * height) * dS;
				
				if (S1 == S2 && S1 == 0){
					S1 = 1;
				}
				
				//pixel layout S2 = {A, R}, S1 = {G, B} 
				bo.write(S1 & 0x00ff);
				bo.write((S1 & 0xff00) >>> 8);
				
				bo.write(S2 & 0x00ff);
				bo.write((S2 & 0xff00) >>> 8);
				
				if (bw != null)
				{
					bw.write(S1 + "," + S2 + " ");
				}
			}
			if (bw != null)
			{
				bw.newLine();
			}
		}

		//write footer
		bo.write(0); bo.write(0); bo.write(0); bo.write(0); //extension offset
		bo.write(0); bo.write(0); bo.write(0); bo.write(0); //developer area offset
		
		String signature = "TRUEVISION-XFILE.";
		for (int i = 0; i < signature.length(); ++i)
			bo.write(signature.charAt(i));
		bo.write(0);
		
		bo.close();
		
		if (bw != null)
			bw.close();

	}

}
