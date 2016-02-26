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
	
	
	
	private static void readClusterPrintAsHTML()
	{
		//first read the all_word_langid_map_max.txt file to create a map of word -> maxLang
		HashMap<String,String> wordLangMap = new HashMap<String,String>();
		String wordLangFile = "data/test/all/all_word_langid_map_max.txt";
		//read cluster file
		String clusterFileName = "/Users/rajarshd/Dropbox/Research/c-lab/DPGMM/my_implementation/final_clusters/last_iteration_table_members.txt";
		String outputFileName = "/Users/rajarshd/Dropbox/Research/c-lab/DPGMM/my_implementation/final_clusters/last_iteration_table_members.html";
		DecimalFormat format = new DecimalFormat("0.00"); 
		
		try{
			
			PrintStream out = new PrintStream(outputFileName,"UTF-8");
			out.println("<html>");
			out.println("<meta http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\" />");
			out.println("<head>");
			out.println("<link rel=\"stylesheet\" href=\"https://code.jquery.com/ui/1.11.1/themes/smoothness/jquery-ui.css\">");
			out.println("<script src=\"https://code.jquery.com/jquery-1.10.2.js\"></script>");
			out.println("<script src=\"https://code.jquery.com/ui/1.11.1/jquery-ui.js\"></script>");			
			out.println("<script>");
			out.println("$(function() {");
			out.println("$( \"#accordion\" ).accordion({collapsible: true,heightStyle:\"content\",active:false});");
			out.println("});</script>");
			out.println("<style type=\"text/css\"></style>");
			out.println("</head>");
			out.println("<body>");
			out.println("<div id=\"accordion\" class=\"ui-accordion ui-widget ui-helper-reset\",role=\"tablist\">");
			BufferedReader reader = new BufferedReader(
			        new InputStreamReader(new FileInputStream(wordLangFile), "UTF-8"));
			String line;
			while((line = reader.readLine())!=null)
			{
				String[] split = line.split(" ");
				String word = split[0];
				String langsCounts = split[1];
				split = langsCounts.split(":");				
				String[] splits = langsCounts.split(":");
				String lang = splits[0];
				int count = Integer.parseInt(splits[1]);
				wordLangMap.put(word, lang);	
				
				//String maxLang = split[0];
								
			}
			reader.close();
			reader = new BufferedReader(new InputStreamReader(new FileInputStream(clusterFileName), "UTF-8"));
			int NumClusterGtThan10 = 0;
			Set<String> languages = new HashSet<String>();
			while((line = reader.readLine())!=null)//for each cluster
			{	HashMap<String,Double> langCountMap = new HashMap<String,Double>();
				String[] split = line.split(":");
				int clusterNum = Integer.parseInt(split[0]);
				String[] clusterWords = split[1].split(" ");
				int count = 0;				
				for(int m=1;m<clusterWords.length;m++) //starting from 1 because first word is space
				{
					String word = clusterWords[m];
					if(wordLangMap.containsKey(word))
					{
						String lang = wordLangMap.get(word);
						languages.add(lang);
						if(!langCountMap.containsKey(lang))						
							langCountMap.put(lang, 1.0);
						else
						{
							double langCount = langCountMap.get(lang);
							langCountMap.put(lang, langCount+1.0);
						}						
						count++;
					}
				}
				
				Set<Entry<String,Double>> set = langCountMap.entrySet();
				ArrayList<Entry<String,Double>> list = new ArrayList<Entry<String,Double>>();
				Iterator<Entry<String,Double>> iter = set.iterator();
				while(iter.hasNext())				
					list.add(iter.next());
				//sort in descending order
				Collections.sort(list, new Comparator<Entry<String,Double>>(){

					@Override
					public int compare(Entry<String, Double> o1,
							Entry<String, Double> o2) {
						// TODO Auto-generated method stub
						double count1 = o1.getValue();
						double count2 = o2.getValue();						
						if(count2 > count1)
							return 1;
						else if(count2 < count1)
							return -1;
						else						
							return 0;
					}
					
				});
				
				//normalize the list
				double sum = 0;
				for(Entry<String,Double> e:list)								
					sum = sum + e.getValue();
				
				for(Entry<String,Double> e:list)	
					e.setValue(e.getValue()/sum);
				if(count >=10)
				{
					NumClusterGtThan10++;
					out.println("<h3>");
					
					out.println(NumClusterGtThan10+": Number of Words "+count+"<br/>");
					System.out.println(NumClusterGtThan10+": Number of Words "+count);
					double sumProb = 0;
					for(int i=0;i<list.size();i++)
					{
						if(sumProb<0.95)
						{
							out.print(list.get(i).getKey()+" : "+format.format(list.get(i).getValue()*100)+"% ");
							System.out.print(list.get(i).getKey()+" : "+list.get(i).getValue()*100+" ");
						}
						else
						{
							double left = 1-sumProb;
							out.println("Others : "+format.format(left*100)+"%");
							System.out.println("Others : "+left);
							break;
						}						
						sumProb = sumProb + list.get(i).getValue();
					}
					out.println("");
					System.out.println();
					out.println("</h3>");
					//Now print the words
					out.println("<div> <p>");
					for(int m=1;m<clusterWords.length;m++) //starting from 1 because first word is space
					{
						String word = clusterWords[m];
						out.print(word+" ");
					}
					out.println("</p></div>");
				}				
			}
			out.println("</div>");
			out.println("</body>");
			out.println("</html>");
			out.close();
			reader.close();
			System.out.println("Total number of languages are "+languages.size());
			Iterator<String> iter = languages.iterator();
			while(iter.hasNext())
			{
				System.out.print(iter.next()+" ");
			}
		}catch(FileNotFoundException ex){
			ex.printStackTrace();
		}catch(IOException ex){
			ex.printStackTrace();
		}		
	}
	
	
	public static void create20NewsCorpus(String fileName, String blackListFile)
	{
		//read the black list file and read the wordid's
		try{
		BufferedReader reader = new BufferedReader(
		        new InputStreamReader(new FileInputStream(blackListFile), "UTF-8"));
		HashSet<String> blackList = new HashSet<String>();
		String line;
		while((line = reader.readLine())!=null)
			blackList.add(line);
		reader.close();
		
		//Now read the 20-news corpus, its of the form docId, wordId, count
		reader = new BufferedReader(
		        new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
		
		String prevDocNum = "1";
		ArrayList<ArrayList<String>> corpus = new ArrayList<ArrayList<String>>();
		ArrayList<String> doc = new ArrayList<String>();
		while((line = reader.readLine())!=null)
		{
			String[] split = line.split(" ");
			String docNum = split[0];
			if(!docNum.equals(prevDocNum))
			{
				corpus.add(doc); //adding the previous
				doc = new ArrayList<String>(); //next doc
				System.out.println("Finished document "+prevDocNum);
				//out.println();
			}
			String wordId = split[1];
			int count = Integer.parseInt(split[2]);
			if(!blackList.contains(wordId))
				for(int i=0;i<count;i++)
					doc.add(wordId);		
			prevDocNum = docNum;						
		}		
		//shuffling the array
		for(ArrayList<String> eachDoc:corpus)
  			Collections.shuffle(eachDoc);
		PrintStream out = new PrintStream("20_news/corpus.test","UTF-8");
		for(ArrayList<String> eachDoc:corpus)
		{
			for(String word:eachDoc)
				out.print(word+" ");
			out.println();
		}
		out.close();
		
		}catch(FileNotFoundException ex){
			ex.printStackTrace();
		}catch(IOException ex){
			ex.printStackTrace();
		}		
	}
	
	public static void createVectorsFor20News()
	{
	
			//the wordIndex of 20-news starts from 1 and not 0. Therefore the first row of the vector file has to be padded with zeros, just to make the indexing
			//correct. Also some indices are not present in 20 news, for example index 41 is not present. Therefor that row also have to be padded wih 0's. Its true
			//that the row will never be used, but just to make the indexing correct, we will have to do it.
	
		try{
		PrintStream out = new PrintStream("20_news/window5/20_news_vectors_30.txt","UTF-8");
		//first print the 0's in the 1st line
		for(int i=0;i<Data.D;i++)
			out.print("0.0"+" ");
		out.println();
		
		//read the 20 news dictionary file
		BufferedReader vocab20News = new BufferedReader(
		        new InputStreamReader(new FileInputStream("20_news/newDict.txt"), "UTF-8"));
		String word20News;
		int counter = 0;
		
		//read the blacklist file, so that we dont have to search for them
		BufferedReader reader = new BufferedReader(
		        new InputStreamReader(new FileInputStream("20_news/blacklist.txt"), "UTF-8"));
		HashSet<Integer> blackList = new HashSet<Integer>();
		String line;
		while((line = reader.readLine())!=null)
			blackList.add(Integer.parseInt(line));
		reader.close();
		
		while((word20News = vocab20News.readLine())!=null)
		{	
			//see if this word is blacklisted
			if(blackList.contains(counter+1))
			{
				for(int i=0;i<Data.D;i++)
					out.print("0.0"+" ");
				out.println();
				counter++;
				continue;
			}
			
			BufferedReader wikiVectors = new BufferedReader(
			        new InputStreamReader(new FileInputStream("wikipedia/new/wiki_vectors.txt.30"), "UTF-8"));
			BufferedReader wikiVocab = new BufferedReader(
			        new InputStreamReader(new FileInputStream("wikipedia/new/wiki.vocab"), "UTF-8"));
			String wikiWord, wikiVector;
			while((wikiWord = wikiVocab.readLine())!=null && (wikiVector = wikiVectors.readLine())!=null)
			{
				if(word20News.equals(wikiWord))
				{
					out.println(wikiVector);
					break;
				}
			}		
			wikiVectors.close();
			wikiVocab.close();
			System.out.println("Done for word no. "+counter);
			counter++;			
		}
		out.close();
		vocab20News.close();
		}catch(FileNotFoundException ex){
			ex.printStackTrace();
		}catch(IOException ex){
			ex.printStackTrace();
		}	
		
	}
	
	
	
	public static void createNIPSDataset(String inputFile, String outputFile)
	{
		try
		{
		//read the NIPS blacklist file
		BufferedReader reader = new BufferedReader(
				    new InputStreamReader(new FileInputStream("nips/blacklist.txt"), "UTF-8"));		
		String word = "";
		HashSet<String> blackWords = new HashSet<String>();
		while((word = reader.readLine())!=null)
			blackWords.add(word);
		reader.close();
		
		//read the NIPS vocab file
		reader = new BufferedReader(
			    new InputStreamReader(new FileInputStream("nips/nips_5000_dict.csv"), "UTF-8"));	
		HashMap<Integer,String> vocabMap = new HashMap<Integer, String>();
		int counter = 0;
		while((word = reader.readLine())!=null)
		{
			vocabMap.put(counter, word);
			counter++;
		}
		reader.close();
		
		//now read the train file, each document is a column vector. Number of rows of the matrix are V = 5000
		int V = 5000;
		//read the first line of the file and decide the number of documents
		reader = new BufferedReader(
			    new InputStreamReader(new FileInputStream(inputFile), "UTF-8"));
		String line = reader.readLine();
		reader.close();		
		String[] split = line.split(",");
		int numDocs = split.length;
		
		
		int[][] corpus = new int[V][numDocs];
		reader = new BufferedReader(
			    new InputStreamReader(new FileInputStream(inputFile), "UTF-8"));
		int rowCounter = 0;
		//read the training or test matrix
		while((line = reader.readLine())!=null)
		{
			split = line.split(",");
			for(int colCounter = 0;colCounter<split.length;colCounter++)			
				corpus[rowCounter][colCounter] = Integer.parseInt(split[colCounter]);			
			rowCounter++;
		}
		reader.close();

		//read the vocab file which is the intersection of wikipedia and nips vocab
		reader = new BufferedReader(
			    new InputStreamReader(new FileInputStream("nips/nips_wiki.vocab"), "UTF-8"));
		HashMap<String,Integer> word2IdMap = new HashMap<String, Integer>();
		counter = 0;
		while((line = reader.readLine())!=null)
		{
			word2IdMap.put(line, counter);
			counter++;
		}
		reader.close();
		PrintStream out = new PrintStream(outputFile,"UTF-8");
		//now go over the matrix column wise, get the word from the map and then		
		for(int j=0;j<numDocs;j++)
		{
			ArrayList<Integer> doc = new ArrayList<Integer>();			
			for(int v=0;v<V;v++)
			{
				word = vocabMap.get(v);
				if(!blackWords.contains(word))
				{
					//now get the corresponding id for the nips_wiki.vocab file.
					int id = word2IdMap.get(word);
					//repeat it corpus[v][j] number of times
					for(int r=0;r<corpus[v][j];r++)
						doc.add(id);
				}
			}
			Collections.shuffle(doc);
			for(int id:doc)
				out.print(id+" ");
			out.println();			
		}
		out.close();
		}catch(FileNotFoundException ex){
			ex.printStackTrace();
		}catch(IOException ex){
			ex.printStackTrace();
		}	
	}
	
	public static void createNIPSVectors()
	{
		try{
				PrintStream out = new PrintStream("nips/nips_vectors_50.txt","UTF-8");
				
				BufferedReader reader = new BufferedReader(
					    new InputStreamReader(new FileInputStream("nips/nips_wiki.vocab"), "UTF-8"));
				String line="";
				int counter = 0;
				while((line = reader.readLine())!=null)
				{
					BufferedReader wikiVectors = new BufferedReader(
					        new InputStreamReader(new FileInputStream("wikipedia/new/wiki_vectors.txt.50"), "UTF-8"));
					BufferedReader wikiVocab = new BufferedReader(
					        new InputStreamReader(new FileInputStream("wikipedia/new/wiki.vocab"), "UTF-8"));
					String wikiWord, wikiVector;
					while((wikiWord = wikiVocab.readLine())!=null && (wikiVector = wikiVectors.readLine())!=null)
					{
						if(line.equals(wikiWord))
						{
							out.println(wikiVector);
							break;
						}
					}
					wikiVocab.close();
					wikiVectors.close();
					System.out.println("Done for word no. "+counter);
					counter++;
				}
				reader.close();
				out.close();
			
			
		}catch(FileNotFoundException ex){
			ex.printStackTrace();
		}catch(IOException ex){
			ex.printStackTrace();
		}	
		
	}
	
}
