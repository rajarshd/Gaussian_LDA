package data;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map.Entry;
import java.util.Random;
import java.util.Set;

import org.ejml.data.DenseMatrix64F;

//import edu.mit.jwi.Dictionary;
//import edu.mit.jwi.IDictionary;
//import edu.mit.jwi.item.IIndexWord;
//import edu.mit.jwi.item.ISynset;
//import edu.mit.jwi.item.IWord;
//import edu.mit.jwi.item.IWordID;
//import edu.mit.jwi.item.POS;

/**
 * Class for reading the data file. In my case, its a bunch of word embeddings (word vectors), where each line
 * is a word embedding.
 * @author rajarshd
 *
 */
public class Data {
	
	public static  String inputFileName = "";
	
	/**
	 * The dimension of the vectors.
	 */
	public static int D; 
	/**
	 * Number of data points
	 */
	public static  int numVectors;
	
	
	/**
	 * 
	 * @param args
	 */
	public static DenseMatrix64F readData() {
		// TODO Auto-generated method stub
		try
		{
			BufferedReader reader =
			    new BufferedReader(
			        new InputStreamReader(new FileInputStream(inputFileName), "UTF-8"));
			String line = "";
			ArrayList<String> lines = new ArrayList<String>(); //read each line and store here and when you have all then put into the matrix
			while((line = reader.readLine())!=null)			
				lines.add(line);							
			reader.close();
			numVectors = lines.size();
			DenseMatrix64F data = new DenseMatrix64F(numVectors,D); //initialize the data matrix
			//now populate the matrix
			for(int i=0;i<lines.size();i++)
			{
				String vector = lines.get(i);
				String[] vals  = vector.split(" ");
				assert vals.length == D;
				int j=0;
				for(String val: vals)
				{
					double dVal = Double.parseDouble(val);
					data.set(i,j,dVal);
					j++;
				}
			}
			return data;
		}
		catch(FileNotFoundException ex){
			ex.printStackTrace();
		}catch(IOException ex){
			ex.printStackTrace();
		}
		
		return null;
	}
	
	public static ArrayList<ArrayList<Integer>> readCorpus(String inputCorpusName)
	{
		try
		{
			BufferedReader reader =
			    new BufferedReader(
			        new InputStreamReader(new FileInputStream(inputCorpusName), "UTF-8"));
			ArrayList<ArrayList<Integer>> corpus = new ArrayList<ArrayList<Integer>>();
			
			String line = "";			
			while((line = reader.readLine())!=null)
			{
				ArrayList<Integer> doc = new ArrayList<Integer>();
				String[] words = line.split(" ");
				for(String word:words)
					doc.add(Integer.parseInt(word));
				corpus.add(doc);				
			}
			reader.close();
			return corpus;			
		}catch(FileNotFoundException ex){
			ex.printStackTrace();
		}catch(IOException ex){
			ex.printStackTrace();
		}
		return null;
	}
}
