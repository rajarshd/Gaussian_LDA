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
	
	/**
	 * read the input tweet file and gather the word vector for each word from vectors
	 * trained by Manaal
	 */
	private static void gatherVectors()
	{
		//first read the wordvector file and make a map of words to line numbers
		HashMap<String,Integer> vector2lineNum = new HashMap<String,Integer>();
		
		try
		{
			BufferedReader reader =
			    new BufferedReader(
			        new InputStreamReader(new FileInputStream("data/train/vec_all_tweets.txt"), "UTF-8"));
			String line = "";
			int counter = 0;
			while((line = reader.readLine())!=null)
			{
				String[] split = line.split(" ");
				String word = split[0];
				//System.out.println(word);
				if(!vector2lineNum.containsKey(word))				
					vector2lineNum.put(word, counter);
				counter++;
			}
			reader.close();
			System.out.println("Training file read");
			//now read the vocab file of test tweets and get the vectors
			reader =
				    new BufferedReader(
				        new InputStreamReader(new FileInputStream("data/test/bergsma_all_tweets.txt.vocab"), "UTF-8"));
			counter = 0;
			BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
				    new FileOutputStream("data/test/bergsma_all_tweets.vocab.line_num"), "UTF-8"));
//				
			while((line = reader.readLine())!=null)
			{
				if(vector2lineNum.containsKey(line))				
					out.write(line+" "+vector2lineNum.get(line)+"\n");
				else				
					counter++;				
			}
			reader.close();
			out.close();
			System.out.println("There are "+counter+" OOV words");
			
		}catch(FileNotFoundException ex){
			ex.printStackTrace();
		}catch(IOException ex){
			ex.printStackTrace();
		}		
	}
	
	private static void getVectorsFromLineNumbers()
	{
		try
		{
			BufferedReader reader =
			    new BufferedReader(
			        new InputStreamReader(new FileInputStream("data/test/bergsma_all_tweets.vocab.line_num.sorted"), "UTF-8"));
			
			BufferedReader reader1 =
				    new BufferedReader(
				        new InputStreamReader(new FileInputStream("data/train/vec_all_tweets.txt"), "UTF-8"));
			
			BufferedWriter out = new BufferedWriter(new OutputStreamWriter(
				    new FileOutputStream("data/test/bergsma_vectors.txt"), "UTF-8"));
			
			
			String line;
			int counter = 0;
			while((line = reader.readLine())!=null)
			{
				String[] split = line.split(" ");
				int lineNum = Integer.parseInt(split[1]);
				String line1;
				while(counter<lineNum) //reading the vector file
				{
					line1 = reader1.readLine();
					counter++;
				}
				if(counter == lineNum)
				{
					line1 = reader1.readLine();;
					counter++;
					String[] wordVectors =  line1.split(" ");
					if(!wordVectors[0].equals(split[0]))
					{
						System.out.println("The words dont match!!");
						System.exit(1);
					}
					else
					{
						String vector=wordVectors[1];
						for(int i=2;i<wordVectors.length;i++)						
							vector = vector+" "+wordVectors[i];
						out.write(vector+"\n");
						System.out.println(counter);
					}
				}				
			}
			reader.close();
			reader1.close();
			out.close();
		}catch(FileNotFoundException ex){
			ex.printStackTrace();
		}catch(IOException ex){
			ex.printStackTrace();
		}		
	}
	
	/**
	 * create the file of the form [word {lang1:count lang2:count.....}]
	 */
	private static void createLangFileWangDataset()
	{
		String inputTweetFileName = "/Users/rajarshd/Dropbox/Research/c-lab/DPGMM/my_implementation/data/test/wang/tweetLID-training.txt";
		String outputFileName = "/Users/rajarshd/Dropbox/Research/c-lab/DPGMM/my_implementation/data/test/wang/word_langid_map.txt";
		
		try
		{
			BufferedReader reader =
				    new BufferedReader(
				        new InputStreamReader(new FileInputStream(inputTweetFileName), "UTF-8"));
			PrintStream out = new PrintStream(outputFileName,"UTF-8");
			
			HashMap<String,HashMap<String,Integer>> mapWordsLanguage = new HashMap<String,HashMap<String,Integer>>(); //word -> {lang -> count}
			String line;
			while((line = reader.readLine())!=null)
			{
				String[] lang_tweet = line.split("\t");
				String lang = lang_tweet[0];
				String tweet = lang_tweet[1];
				String[] words = tweet.split(" ");
				
				for(String word:words)
				{
					word = word.toLowerCase();
					if(!mapWordsLanguage.containsKey(word))
					{
						HashMap<String,Integer> langCount = new HashMap<String,Integer>();
						langCount.put(lang, 1);
						mapWordsLanguage.put(word, langCount);
					}
					else
					{
						HashMap<String,Integer> langCount = mapWordsLanguage.get(word);
						if(!langCount.containsKey(lang))						
							langCount.put(lang, 1);
						else
						{
							int count = langCount.get(lang);
							langCount.put(lang, count+1);
						}						
					}
				}					
			}
			reader.close();
			//Now go over the map and print
			Set<Entry<String,HashMap<String,Integer>>> wordEntries = mapWordsLanguage.entrySet();
			for(Entry<String,HashMap<String,Integer>> wordEntry:wordEntries)
			{
				String output = "";
				String word = wordEntry.getKey();
				output = output + word+" {";
				HashMap<String,Integer> langCountMap = wordEntry.getValue();
				Set<Entry<String,Integer>> langCountEntrySet = langCountMap.entrySet();
				for(Entry<String,Integer> langCountEntry:langCountEntrySet)
				{
					String lang = langCountEntry.getKey();
					int count = langCountEntry.getValue();
					output = output+lang+":"+count+",";
				}
				output = output + "}";
				out.println(output);
				System.out.println(output);
			}			
			out.close();
			reader.close();
			System.out.println("done");
		}catch(FileNotFoundException ex){
			ex.printStackTrace();
		}catch(IOException ex){
			ex.printStackTrace();
		}
		
	}
	
	private static void createLangFileBergsmaDataset()
	{
		String[] languages = {"arabic", "hindi", "bulgarian", "farsi", "marathi", "nepali" ,"russian", "ukrainian" ,"urdu"};
		String[] isoCodes = {"ar","hi","bg","fa","mr","ne","ru","uk","ur"};
		String outputFileName = "/Users/rajarshd/Dropbox/Research/c-lab/DPGMM/my_implementation/data/test/bergsma/word_langid_map.txt";
		HashMap<String,HashMap<String,Integer>> mapWordsLanguage = new HashMap<String,HashMap<String,Integer>>(); //word -> {lang -> count}
		try
		{
			PrintStream out = new PrintStream(outputFileName,"UTF-8");
			int isoCounter = 0;
			for(String language:languages)
			{
				String lang = isoCodes[isoCounter];
				String inputTweetFileName = "/Users/rajarshd/Dropbox/Research/c-lab/DPGMM/my_implementation/data/test/bergsma/Shared/test_tweets.txt."+language;
				BufferedReader reader =
					    new BufferedReader(
					        new InputStreamReader(new FileInputStream(inputTweetFileName), "UTF-8"));
				
				String tweet;
				while((tweet = reader.readLine())!=null)
				{
					String[] words = tweet.split(" ");
					for(String word:words)
					{
						word = word.toLowerCase();
						if(!mapWordsLanguage.containsKey(word))
						{
							HashMap<String,Integer> langCount = new HashMap<String,Integer>();
							langCount.put(lang, 1);
							mapWordsLanguage.put(word, langCount);
						}
						else
						{
							HashMap<String,Integer> langCount = mapWordsLanguage.get(word);
							if(!langCount.containsKey(lang))						
								langCount.put(lang, 1);
							else
							{
								int count = langCount.get(lang);
								langCount.put(lang, count+1);
							}						
						}
					}			
					
				}
				reader.close();
				isoCounter++;
			}
			//Now go over the map and print
			Set<Entry<String,HashMap<String,Integer>>> wordEntries = mapWordsLanguage.entrySet();
			for(Entry<String,HashMap<String,Integer>> wordEntry:wordEntries)
			{
				String output = "";
				String word = wordEntry.getKey();
				output = output + word+" {";
				HashMap<String,Integer> langCountMap = wordEntry.getValue();
				Set<Entry<String,Integer>> langCountEntrySet = langCountMap.entrySet();
				for(Entry<String,Integer> langCountEntry:langCountEntrySet)
				{
					String lang = langCountEntry.getKey();
					int count = langCountEntry.getValue();
					output = output+lang+":"+count+",";
				}
				output = output + "}";
				out.println(output);
				System.out.println(output);
			}			
			out.close();
			
		}catch(FileNotFoundException ex){
			ex.printStackTrace();
		}catch(IOException ex){
			ex.printStackTrace();
		}
 	}
	/**
	 * reads all the word_langid_map.txt from all the datasets and merge them together. Also ouput another file which contains word -> {maxLang:maxCount}
	 */
	private static HashMap<String,HashMap<String,Integer>> mergeAllWordLangIDMaps()
	{
		//ok first lets read the Bergsma word_langid_map.txt file
		try{
			HashMap<String,HashMap<String,Integer>> mapWordsLanguage = new HashMap<String,HashMap<String,Integer>>(); //word -> {lang -> count}
			
			String[] inputTweetFileName = {"/Users/rajarshd/Dropbox/Research/c-lab/DPGMM/my_implementation/data/test/bergsma/word_langid_map.txt","/Users/rajarshd/Dropbox/Research/c-lab/DPGMM/my_implementation/data/test/wang/word_langid_map.txt"
					,"/Users/rajarshd/Dropbox/Research/c-lab/DPGMM/my_implementation/data/test/timb/word_langid_map.txt"};
			String outputFileName = "/Users/rajarshd/Dropbox/Research/c-lab/DPGMM/my_implementation/data/test/all/all_word_langid_map.txt";
			PrintStream out = new PrintStream(outputFileName,"UTF-8");
		for(String fileName:inputTweetFileName)
		{
			BufferedReader reader =
				    new BufferedReader(
				        new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
			String line;
			while((line = reader.readLine())!=null)
			{
				String[] split = line.split(" ");
				if(split.length!=2)
					continue;
				String word = split[0];
				//String languagesAndCounts = split[1];
				String[]languagesAndCounts = split[1].split(",");
				//go over each language:count pairs
				for(int i=0;i<languagesAndCounts.length;i++)
				{
					String[] eachLangAndCount =  languagesAndCounts[i].split(":");
					if(eachLangAndCount.length == 2)
					{
						String lang = eachLangAndCount[0];
						int count = Integer.parseInt(eachLangAndCount[1]);
						if(!mapWordsLanguage.containsKey(word))
						{
							HashMap<String,Integer> langCount = new HashMap<String,Integer>();
							langCount.put(lang, count);
							mapWordsLanguage.put(word, langCount);
						}
						else
						{
							HashMap<String,Integer> langCount = mapWordsLanguage.get(word);
							if(!langCount.containsKey(lang))						
								langCount.put(lang, count);
							else
							{
								int prevcount = langCount.get(lang);
								langCount.put(lang, prevcount+count);
							}			
						}
					}
					
				}
				
		}
			reader.close();		
	 }
		//now write the map into a file
		PrintStream outMax = new PrintStream("/Users/rajarshd/Dropbox/Research/c-lab/DPGMM/my_implementation/data/test/all/all_word_langid_map_max.txt","UTF-8");
		Set<Entry<String,HashMap<String,Integer>>> wordEntries = mapWordsLanguage.entrySet();
		for(Entry<String,HashMap<String,Integer>> wordEntry:wordEntries)
		{
			String output = "";
			String word = wordEntry.getKey();
			output = output + word+" ";
			HashMap<String,Integer> langCountMap = wordEntry.getValue();
			Set<Entry<String,Integer>> langCountEntrySet = langCountMap.entrySet();
			int maxCount = Integer.MIN_VALUE;
			String maxLang = "";
			for(Entry<String,Integer> langCountEntry:langCountEntrySet)
			{
				String lang = langCountEntry.getKey();
				int count = langCountEntry.getValue();
				if(count > maxCount)
				{
					maxCount = count;
					maxLang = lang;
				}
				output = output+lang+":"+count+",";
			}
			String outputMax = word+" "+maxLang+":"+maxCount;
			out.println(output);
			outMax.println(outputMax);
			System.out.println(output);
		}			
		out.close();
		outMax.close();
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
	
	private static void generateFakeDoc()
	{
		try{
			PrintStream out = new PrintStream("fake_doc.txt.small","UTF-8");			
			//generate 1000 rand ints in each line. DO this 10000 times
			Random gen = new Random();
			for(int d=0;d<100;d++)
			{
				for(int i=0;i<100;i++)
					out.print(gen.nextInt(100)+" ");
				out.println();
			}
			out.close();
		}catch(FileNotFoundException ex){
			ex.printStackTrace();
		}catch(IOException ex){
			ex.printStackTrace();
		}		
		
	}
	
	private static void generateFakeTestDoc()
	{
		try{
			PrintStream out = new PrintStream("fake_doc.txt.test","UTF-8");			
			//generate 1000 rand ints in each line. DO this 10000 times
			int counter = 0;
			for(int d=0;d<1000;d++)
			{
				for(int i=0;i<200;i++)
				{
					out.print(counter+" ");
					counter++;
				}
				out.println();
			}
			out.close();
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
	/**
	 * creates the corpus.train for NIPS corpus (raw text) where the vectors were trained from itself.
	 */
	public static void createNIPSItselfCorpus()
	{
		try
		{
			BufferedReader reader =
			    new BufferedReader(
			        new InputStreamReader(new FileInputStream("nips_itself_experiment/nips_itself.vocab"), "UTF-8"));
			String line;
			HashMap<String, Integer> corpus = new HashMap<String, Integer>();
			int counter = 0;
			while((line = reader.readLine())!=null)
			{
				corpus.put(line, counter);
				counter++;
			}
			reader.close();
			reader =
				    new BufferedReader(
				        new InputStreamReader(new FileInputStream("nips_itself_experiment/all_text/nips_tokenized_with_document_breakpoints.txt"), "UTF-8"));
			PrintStream out = new PrintStream("nips_itself_experiment/corpus.train","UTF-8");
			int doc_counter = 0;
			while((line = reader.readLine())!=null)
			{
				if(line.equals("~ ~ ~ ~ ~ ~ end of file ~ ~ ~ ~ ~ ~"))
				{
					out.println();
					continue;
				}
				String[] words = line.split(" ");
				for(String word:words)
				{
					if(corpus.get(word)!=null)
						out.print(corpus.get(word)+" ");
				}
			}
				
			
		}catch(FileNotFoundException ex){
			ex.printStackTrace();
		}catch(IOException ex){
			ex.printStackTrace();
		}	
	}	
	/**
	 * Get the intersection with ppdb and vocab file for one of the corpus.
	 */
	public static void checkIntersectionPPDB(String vocabFile, String ppdbFileName,String outputFile, String wikiVocabFile)
	{
		try
		{
		//first read the PPDB map file
			BufferedReader reader =
				    new BufferedReader(
				        new InputStreamReader(new FileInputStream(ppdbFileName), "UTF-8"));
			String line;			
			HashMap<String,String> ppdbMap = new HashMap<String, String>();
			while((line = reader.readLine())!=null)
			{
				String[] split = line.split(" ");
				//System.out.println(split.length);
				//System.out.println(split[1]+" "+split[2]);				
				ppdbMap.put(split[1], split[2]);
			}			
			reader.close();
			HashSet<String> vocab = new HashSet<String>();
			reader =new BufferedReader(new InputStreamReader(new FileInputStream(vocabFile), "UTF-8"));
			while((line = reader.readLine())!=null)
				vocab.add(line);
			reader.close();
			
			//read the wiki.vocab file
			HashSet<String> wikiVocab = new HashSet<String>();
			reader =new BufferedReader(new InputStreamReader(new FileInputStream(wikiVocabFile), "UTF-8"));
			while((line = reader.readLine())!=null)			
				wikiVocab.add(line);
			reader.close();
			
			
			//read the keys of the ppdb map and if the key is present in vocab and value is not in  vocab but present in wiki.vocab, then we keep that map
			
			PrintStream out = new PrintStream(outputFile,"UTF-8");
			int count = 0;
			Set<String> ppdbKeys = ppdbMap.keySet();	
			Iterator<String> iter = ppdbKeys.iterator();
			while(iter.hasNext())
			{
				String key = iter.next();
				//System.out.println(key);
				if(vocab.contains(key) && !vocab.contains(ppdbMap.get(key)) && wikiVocab.contains(ppdbMap.get(key)))
				{
					out.println(key+":"+ppdbMap.get(key));
					count++;
				}
			}
			out.close();
			System.out.println(count);
			
			
		}catch(FileNotFoundException ex){
			ex.printStackTrace();
		}catch(IOException ex){
			ex.printStackTrace();
		}	
		
		
	}
	
	public static void appendNewWordsAndVectors()
	{		
		try
		{
			//read the nips_wiki.vocab file
//			BufferedReader reader =
//				    new BufferedReader(
//				        new InputStreamReader(new FileInputStream("nips/nips_wiki.vocab"), "UTF-8"));
			BufferedReader reader =
				    new BufferedReader(
				        new InputStreamReader(new FileInputStream("20_news/newDict.txt.orig"), "UTF-8")); //read from the original newDict.txt
			String line;
			HashMap<String,Integer> nipsVocabMap = new HashMap<String, Integer>();
			//int counter = 0;
			int counter = 1; //for 20_news starting from 1
			while((line = reader.readLine())!=null)
			{
				nipsVocabMap.put(line, counter);
				counter++;
			}
			reader.close();
			
			//read the nips_ppdb_map.txt file
//			reader =
//				    new BufferedReader(
//				        new InputStreamReader(new FileInputStream("nips/nips_ppdb_map.txt"), "UTF-8"));
			reader =
				    new BufferedReader(
				        new InputStreamReader(new FileInputStream("20_news/wordnet/wordnet_map.txt"), "UTF-8"));
			//open an output to append the nips_wiki.vocab file to add the new words
			//PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("nips/nips_wiki.vocab", true))); //this will append in the existing vocab file
			PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("20_news/wordnet/newDict.txt", true))); //this will append in the existing vocab file
			//PrintWriter mapOut = new PrintWriter(new BufferedWriter(new FileWriter("nips/nips_ppdb_map.txt.numbers")));
			PrintWriter mapOut = new PrintWriter(new BufferedWriter(new FileWriter("20_news/wordnet/wordnet_map.txt.numbers")));
			//PrintWriter nips_vector = new PrintWriter(new BufferedWriter(new FileWriter("nips/nips_vectors_30.txt",true)));
			PrintWriter nips_vector = new PrintWriter(new BufferedWriter(new FileWriter("20_news/wordnet/20_news_vectors_30.txt",true)));
			while((line = reader.readLine())!=null)
			{
				
				String[] split = line.split(":");
				String originalWord = split[0];
				String newPPDBWord = split[1];
				System.out.println("Original word "+originalWord);
				System.out.println("New word "+newPPDBWord);
				int oldId = nipsVocabMap.get(originalWord);
				int newId = counter; //because counter will be the (line_number-1) of the new word.
				out.println(newPPDBWord);				
				System.out.println("Appended to the vocab file");
				mapOut.println(oldId+" "+newId);
				BufferedReader wikiVectors = new BufferedReader(
				        new InputStreamReader(new FileInputStream("wikipedia/new/wiki_vectors.txt.30"), "UTF-8"));
				BufferedReader wikiVocab = new BufferedReader(
				        new InputStreamReader(new FileInputStream("wikipedia/new/wiki.vocab"), "UTF-8"));
				String wikiWord, wikiVector;
				while((wikiWord = wikiVocab.readLine())!=null && (wikiVector = wikiVectors.readLine())!=null)
				{
					if(wikiWord.equals(newPPDBWord))//append to the vector file		
					{
						nips_vector.println(wikiVector);
						System.out.println("Appended to the vector file");
					}
					
				}
				wikiVectors.close();
				wikiVocab.close();
				counter++;
			}
			out.close();
			mapOut.close();
			reader.close();
			nips_vector.close();
		}catch(FileNotFoundException ex){
			ex.printStackTrace();
		}catch(IOException ex){
			ex.printStackTrace();
		}	
	}
	
	
	public static void createSyntheticCorpus()
	{
		//read the vocab file
		try
		{
			//read the nips_ppdb_map.txt.numbers file
//			BufferedReader reader =
//				    new BufferedReader(
//				        new InputStreamReader(new FileInputStream("nips/nips_ppdb_map.txt.numbers"), "UTF-8"));			
			BufferedReader reader =
				    new BufferedReader(
				        new InputStreamReader(new FileInputStream("20_news/wordnet/wordnet_map.txt.numbers"), "UTF-8"));
			
			
			String line;
			HashMap<Integer,Integer> oldId2NewIdMap = new HashMap<Integer,Integer>();
			while((line = reader.readLine())!=null)
			{
				String[] split = line.split(" ");
				int oldId = Integer.parseInt(split[0]);
				int newId = Integer.parseInt(split[1]);
				oldId2NewIdMap.put(oldId, newId);
			}
			reader.close();
			//now read the copus.test file
//			reader =
//				    new BufferedReader(
//				        new InputStreamReader(new FileInputStream("nips/corpus.test"), "UTF-8"));
			reader =
				    new BufferedReader(
				        new InputStreamReader(new FileInputStream("20_news/corpus.test"), "UTF-8"));
			
//			PrintWriter mapOut = new PrintWriter(new BufferedWriter(new FileWriter("nips/corpus.synthetic")));
			PrintWriter mapOut = new PrintWriter(new BufferedWriter(new FileWriter("20_news/wordnet/corpus.synthetic")));

			//replace any words which can be replaced
			int totalSubstitutions = 0;
			while((line = reader.readLine())!=null)
			{
				int numSubstitutionsPerDoc = 0;
				String[] split = line.split(" ");
				for(int i=0;i<split.length;i++)
				{
					if(oldId2NewIdMap.get(Integer.parseInt(split[i]))!=null) //the current word has a replacement
					{
						split[i] = ""+oldId2NewIdMap.get(Integer.parseInt(split[i]));
						numSubstitutionsPerDoc++;
						totalSubstitutions++;
					}
				}
				//write the synthetic document
				for(int i=0;i<split.length;i++)
					mapOut.print(split[i]+" ");
				mapOut.println();
				System.out.println("Total number of substitutions for the document "+numSubstitutionsPerDoc);
			}
			System.out.println("Total substitutions for all docs "+totalSubstitutions);
			reader.close();
			mapOut.close();
			
			
		}catch(FileNotFoundException ex){
			ex.printStackTrace();
		}catch(IOException ex){
			ex.printStackTrace();
		}	
	}
	/**
	 * read the corpus.train+test+synthetic file and convert from id to words
	 */
	public static void replaceCorpusByWords()
	{
		try
		{
			//read the nips_wiki.vocab file
//			BufferedReader reader =
//				    new BufferedReader(
//				        new InputStreamReader(new FileInputStream("nips/nips_wiki.vocab"), "UTF-8"));
			BufferedReader reader =
				    new BufferedReader(
				        new InputStreamReader(new FileInputStream("20_news/wordnet/newDict.txt"), "UTF-8"));
			String line;
			HashMap<Integer,String> nipsVocabMap = new HashMap<Integer,String>(); 
			//int counter = 0;
			int counter = 1; //starting from 1 for 20_news
			while((line = reader.readLine())!=null)
			{
				nipsVocabMap.put(counter,line);
				counter++;
			}
			reader.close();
			
			//read the corpus.train+test+synthetic file
			reader =
				    new BufferedReader(
				        new InputStreamReader(new FileInputStream("20_news/wordnet/corpus.train+test+synthetic"), "UTF-8"));
			PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("20_news/wordnet/wcorpus.train+test+synthetic")));
			while((line = reader.readLine())!=null)
			{
				String[] ids = line.split(" ");
				for(String id:ids)
				{
					String word = nipsVocabMap.get(Integer.parseInt(id));
					if(word!=null)					
						out.print(word+" ");
					else
						System.out.println("An id unmatched "+id);
				}
				out.println();
			}
			out.close();
			
			
		}catch(FileNotFoundException ex){
			ex.printStackTrace();
		}catch(IOException ex){
			ex.printStackTrace();
		}	
	}
	
//	public static void testDictionary() throws IOException {
//		
//		 // construct the URL to the Wordnet dictionary directory
//		 String wnhome = "/usr/local/WordNet-3.0";
//		 //System.out.println(wnhome);
//		 String path = wnhome + File.separator + "dict";
//		 URL url = new URL("file",null,path);
//		 // construct the dictionary object and open it
//		 IDictionary dict = new Dictionary ( url);
//		 dict . open ();
//		
//		 IIndexWord idxWord = dict . getIndexWord ("change", POS.VERB );
//		 for(IWordID wordId:idxWord.getWordIDs())
//		 {
//			 IWord word = dict.getWord(wordId);
//			 ISynset synset = word.getSynset();
//			 for(IWord w:synset.getWords())
//				 System.out.print(w.getLemma()+" ");
//			 System.out.println();
//		 }
//	}
	
	public static void getMaxOccuringPOS()
	{
		//read then 20_news.postagged.nl.sorted.uniq 
		try
		{
			BufferedReader reader =
				    new BufferedReader(
				        new InputStreamReader(new FileInputStream("20_news/20_news.postagged.nl.sorted.uniq"), "UTF-8"));
			PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("20_news/wordnet/pos_tagged")));
			String line;
			String prevWord="";
			int maxCount = Integer.MIN_VALUE;
			String maxPos="";
			boolean isFirstLine = true;
			while((line = reader.readLine())!=null)
			{
				String[] split = line.split(" ");
				int count = Integer.parseInt(split[0]);
				split = split[1].split("_");
				String word = split[0];
				String pos = split[1];	
				if(!word.equals(prevWord))//search for prev word is over, so time to write it into a file
				{
					if(!isFirstLine)
						out.println(prevWord+":"+maxPos.toUpperCase());
					maxCount = count;
					maxPos = pos;
					isFirstLine = false;
				}
				else
				{
					if(count > maxCount)
					{
						maxCount = count;
						maxPos = pos;
					}
					
				}
				prevWord = word;
			}
			reader.close();			
		}catch(FileNotFoundException ex){
			ex.printStackTrace();
		}catch(IOException ex){
			ex.printStackTrace();
		}	
	}
	
//	public static void getSynonymsWordNet()
//	{
//		try
//		{
//			BufferedReader reader =
//				    new BufferedReader(
//				        new InputStreamReader(new FileInputStream("20_news/wordnet/pos_tagged"), "UTF-8"));
//			String line;
//			HashMap<String,String> posMap = new HashMap<String, String>();
//			while((line = reader.readLine())!=null)
//			{
//				String[] split = line.split(":");
//				if(split.length>1)
//					posMap.put(split[0], split[split.length-1]);
//			}			
//			reader.close();
//			//read the vocab file
//			HashSet<String> vocab = new HashSet<String>();
//			reader =
//				    new BufferedReader(
//				        new InputStreamReader(new FileInputStream("20_news/newDict.txt.orig"), "UTF-8"));
//			while((line = reader.readLine())!=null)
//				vocab.add(line);			
//			reader.close();	
//			
//			//read the wiki.vocab file
//			HashSet<String> wikiVocab = new HashSet<String>();
//			reader =new BufferedReader(new InputStreamReader(new FileInputStream("wikipedia/new/wiki.vocab"), "UTF-8"));
//			while((line = reader.readLine())!=null)			
//				wikiVocab.add(line);
//			reader.close();
//			
//			// construct the URL to the Wordnet dictionary directory
//			 String wnhome = "/usr/local/WordNet-3.0";
//			 //System.out.println(wnhome);
//			 String path = wnhome + File.separator + "dict";
//			 URL url = new URL("file",null,path);
//			 // construct the dictionary object and open it
//			 IDictionary dict = new Dictionary ( url);
//			 dict . open ();
//			
//			reader =
//				    new BufferedReader(
//				        new InputStreamReader(new FileInputStream("20_news/newDict.txt.orig"), "UTF-8"));
//			
//			PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter("20_news/wordnet/wordnet_map.txt")));
//			
//			while((line = reader.readLine())!=null)
//			{
//				//get the pos of the word
//				String pos = posMap.get(line);
//				POS p;
//				if(pos == null)
//					continue;
//				if(pos.startsWith("N"))
//					p = POS.NOUN;
//				else if(pos.startsWith("V"))
//					p = POS.VERB;
//				else if(pos.startsWith("J"))
//					p = POS.ADJECTIVE;
//				else if(pos.startsWith("RB"))
//					p = POS.ADVERB;
//				else
//					continue;
//				IIndexWord idxWord = dict . getIndexWord (line, p );
//				int flag = 0;
//				if(idxWord == null)
//					continue;
//				 for(IWordID wordId:idxWord.getWordIDs())
//				 {
//					 IWord word = dict.getWord(wordId);
//					 ISynset synset = word.getSynset();
//					 for(IWord w:synset.getWords())
//					 {
//						 //get each lemma
//						 String synonym = w.getLemma();
//						 if(synonym.split("_").length>1) //ignore multiword
//							 continue;
//						 if(!vocab.contains(synonym) && wikiVocab.contains(synonym))
//						 {
//							 out.println(line+":"+synonym);
//							 flag = 1;
//							 break;
//						 }
//					 }
//					 if(flag == 1)
//					 {
//						 flag = 0;
//						 break;
//					 }
//				 }
//				
//			}
//			reader.close();
//			out.close();
//		}catch(FileNotFoundException ex){
//			ex.printStackTrace();
//		}catch(IOException ex){
//			ex.printStackTrace();
//		}	
//	}
	
	public static void main(String[] args)
	{
		//createLangFileWangDataset();
		//createLangFileBergsmaDataset();
		//gatherVectors();
		//getVectorsFromLineNumbers();
		//mergeAllWordLangIDMaps();
		//readClusterPrintAsHTML();
		//generateFakeTestDoc();
		//create20NewsCorpus("20_news/newTest.txt", "20_news/blacklist.txt");
		//Data.D = 30;
		//createVectorsFor20News();
		
		//createNIPSDataset("nips/nips_5000_test.csv", "nips/corpus.test");
		//createNIPSVectors();
		//createNIPSItselfCorpus();
		
		//checkIntersectionPPDB("nips/nips_wiki.vocab","PPDB/ppdb-1.0-l-lexical","nips/nips_ppdb_map.txt","wikipedia/new/wiki.vocab"); //PPDB/ppdb-1.0-s-lexical
		//checkIntersectionPPDB("20_news/newDict.txt","PPDB/ppdb-1.0-l-lexical","20_news/ppdb/20_news_ppdb_map.txt","wikipedia/new/wiki.vocab"); //PPDB/ppdb-1.0-s-lexical
		//appendNewWordsAndVectors();
		//createSyntheticCorpus();
		replaceCorpusByWords();
//		try {
//			testDictionary();
//		} catch (IOException e) {
//			// TODO Auto-generated catch block
//			e.printStackTrace();
//		}
		
	//	getMaxOccuringPOS();
		//getSynonymsWordNet();
		
		
	}
	

}
