package sampler;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import org.apache.commons.math3.special.Gamma;
import org.ejml.alg.dense.decomposition.TriangularSolver;
import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;
import org.ejml.interfaces.decomposition.CholeskyDecomposition;
import org.ejml.ops.CommonOps;

import priors.NormalInverseWishart;
import util.Util;
import util.VoseAlias;
import data.Data;


/**
 * Implementation of the collapsed gibbs sampler for Dirichlet Process Mixture Models where the distribution of each table are multivariate gaussian
 * with unknown mean and covariances. I have extensively referred Frank Wood's matalab implementation (http://www.robots.ox.ac.uk/~fwood/code/index.html) 
 * @author rajarshd
 *
 */
public class GaussianLDAAlias implements Runnable {

	
	/**
	 * The embedding associated with each word of the vocab. 
	 */
	private static DenseMatrix64F[] dataVectors;
	
	/**
	 * The corpus of documents
	 */
	private static ArrayList<ArrayList<Integer>> corpus;
	/**
	 * Number of iterations of Gibbs sweep
	 */
	private static int numIterations;
	
	/**
	 * Number of tables in the current iteration
	 */
	private static int K;
	
	/**
	 * Number of documents
	 */
	private static int N;
	/**
	 * In the current iteration, map of table_id's to number of customers. ****Table id starts from 0.****
	 */
	private static HashMap<Integer,Integer> tableCounts = new HashMap<Integer,Integer>();
	
	/**
	 * tableCountstableCountsPerDoc is a K X N array. tableCounts[i][j] represents how many words of document j are present in topic i.
	 */  
	private static int[][] tableCountsPerDoc ;
	/**
	 * map of table id to id of customers
	 */
	//private static HashMap<Integer,Set<Integer>> tableMembers = new HashMap<Integer,Set<Integer>>();
	
	/**
	 * Stores the table (topic) assignment of each customer in each iteration. tableAssignments[i][j] gives the table assignment of customer j of the ith document. 
	 */
	private static ArrayList<ArrayList<Integer>> tableAssignments ;
	

	/**
	 * The following 4 parameters are arraylist and not maps because, if they are K tables, they are continuously numbered from 0 to K-1 and hence we can directly index them.  
	 */
	/**
	 * mean vector associated with each table in the current iteration. This is the bayesian mean (i.e has the prior part too)
	 */
	private static ArrayList<DenseMatrix64F> tableMeans = new ArrayList<DenseMatrix64F>();		
	
	

	/**
	 * Cholesky Lower Triangular Decomposition of covariance matrix associated with each table.
	 */
	private static ArrayList<DenseMatrix64F> tableCholeskyLTriangularMat = new ArrayList<DenseMatrix64F>();
	
	/**
	 * log-determinant of covariance matrix for each table. Since 0.5*logDet is required in (see logMultivariateTDensity), therefore that value is kept.
	 */
	private static ArrayList<Double> logDeterminants = new ArrayList<Double>();
	
	/**
	 * current iteration counter
	 */
	private static int currentIteration; 
	
	/**
	 * the normal inverse wishart prior
	 */
	private static NormalInverseWishart prior;
	
	private static CholeskyDecomposition<DenseMatrix64F> decomposer = DecompositionFactory.chol(Data.D, true);
	/**
	 * Caching the choelsky of prior sigma0
	 */
	private static DenseMatrix64F CholSigma0; 
	
	/**
	 * file path for reading vocab (to form mapping) and the initial cluster assignment
	 */
	private static String dirName;
	private static BufferedWriter runLogger = null;
	private static BufferedWriter perplexities = null;
	//the dirichlet hyperparam.
	private static double alpha;

	/**
	 * stores the alias table for each word
	 */
	private static VoseAlias[] q ;
	
	public static  boolean done = false;
	
	private static int MH_STEPS = 2;
/************************************Member Declaration Ends***********************************/	
	/**
	 * updates params -- mean and the cholesky decomp of covariance matrix using rank1 update (customer added) or downdate (customer removed)
	 * @param tableId
	 * @param custId
	 * @param isRemoved
	 */
	private static void updateTableParams(int tableId,int custId, boolean isRemoved)
	{
		int count = tableCounts.get(tableId);
		double k_n = prior.k_0 + count;
		double nu_n = prior.nu_0 + count;		
		double scaleTdistrn = (k_n + 1)/(k_n * (nu_n - Data.D + 1));
		
		DenseMatrix64F oldLTriangularDecomp = tableCholeskyLTriangularMat.get(tableId);		
		if(isRemoved)
		{
			/**
			 * Now use the rank1 downdate to calculate the cholesky decomposition of the updated covariance matrix
			 * the update equaltion is \Sigma_(N+1) =\Sigma_(N) - (k_0 + N+1)/(k_0 + N)(X_{n} - \mu_{n-1})(X_{n} - \mu_{n-1})^T
			 * therefore x = sqrt((k_0 + N - 1)/(k_0 + N)) (X_{n} - \mu_{n})
			 * Note here \mu_n will be the mean before updating. After updating sigma_n, we will update \mu_n.
			 */			
			DenseMatrix64F x = new DenseMatrix64F(Data.D, 1);
			CommonOps.sub(dataVectors[custId], tableMeans.get(tableId), x); //calculate (X_{n} - \mu_{n-1})
			double coeff = Math.sqrt((k_n+1)/k_n);
			CommonOps.scale(coeff, x);
			Util.cholRank1Downdate(oldLTriangularDecomp, x);
			tableCholeskyLTriangularMat.set(tableId, oldLTriangularDecomp);//the cholRank1Downdate modifies the oldLTriangularDecomp, therefore putting it back to the map			
			//updateMean(tableId);
			DenseMatrix64F newMean = new DenseMatrix64F(Data.D, 1);
			CommonOps.scale(k_n+1, tableMeans.get(tableId), newMean);
			CommonOps.subEquals(newMean, dataVectors[custId]);
			CommonOps.divide(k_n, newMean);
			tableMeans.set(tableId, newMean);
			
		}
		else //new customer is added
		{			
			DenseMatrix64F newMean = new DenseMatrix64F(Data.D, 1);
			CommonOps.scale(k_n-1, tableMeans.get(tableId), newMean);
			CommonOps.addEquals(newMean, dataVectors[custId]);
			CommonOps.divide(k_n, newMean);
			tableMeans.set(tableId, newMean);
			/**
			 * The rank1 update equation is
			 * \Sigma_{n+1} = \Sigma_{n} + (k_0 + n + 1)/(k_0 + n) * (x_{n+1} - \mu_{n+1})(x_{n+1} - \mu_{n+1})^T
			 */
			DenseMatrix64F x = new DenseMatrix64F(Data.D, 1);
			CommonOps.sub(dataVectors[custId], tableMeans.get(tableId), x); //calculate (X_{n} - \mu_{n-1})
			double coeff = Math.sqrt(k_n/(k_n - 1));
			CommonOps.scale(coeff, x);
			Util.cholRank1Update(oldLTriangularDecomp, x);
			tableCholeskyLTriangularMat.set(tableId, oldLTriangularDecomp);//the cholRank1Downdate modifies the oldLTriangularDecomp, therefore putting it back to the map
		}
		//calculate the 0.5*log(det) + D/2*scaleTdistrn; the scaleTdistrn is because the posterior predictive distribution sends in a scaled value of \Sigma
		double logDet = 0.0;
		for(int l=0;l<Data.D;l++)				
			logDet = logDet + Math.log(oldLTriangularDecomp.get(l, l));	
		logDet += Data.D*Math.log(scaleTdistrn)/(double)2;
		
		if(tableId < logDeterminants.size())				
			logDeterminants.set(tableId, logDet);
		else
			logDeterminants.add(logDet);
		
	}
	/**
	 * Initialize the gibbs sampler state. I start with log N tables and randomly initialize customers to those tables.
	 * @throws IOException 
	 */
	public static void initialize() throws IOException
	{
		currentIteration = 0;
		//first check the prior degrees of freedom. It has to be >= num_dimension
		if(prior.nu_0 < (double)Data.D)
		{
			System.out.println("The initial degrees of freedom of the prior is less than the dimension!. Setting it to the number of dimension: "+Data.D);
			prior.nu_0 = Data.D;
		}		
		//storing zeros in sumTableCustomers and later will keep on adding each customer. Also initialize tableInverseCovariances and determinants
		double scaleTdistrn = (prior.k_0+1)/(double)(prior.k_0*(prior.nu_0 - Data.D + 1));
		for(int i=0;i<K;i++)
		{
			DenseMatrix64F priorMean = new DenseMatrix64F(prior.mu_0);
			DenseMatrix64F initialCholesky = new DenseMatrix64F(CholSigma0);
			//calculate the 0.5*log(det) + D/2*scaleTdistrn; the scaleTdistrn is because the posterior predictive distribution sends in a scaled value of \Sigma
			double logDet = 0.0;			
			for(int l=0;l<Data.D;l++)				
				logDet = logDet + Math.log(CholSigma0.get(l, l));	
			logDet += Data.D*Math.log(scaleTdistrn)/(double)2;
			logDeterminants.add(logDet);			
			tableMeans.add(priorMean);			
			tableCholeskyLTriangularMat.add(initialCholesky);
		}					
		//randomly assign customers to tables.
		Random gen = new Random();
		for(int d=0;d<N;d++) //for each document
		{
			ArrayList<Integer> doc = corpus.get(d);
			int wordCounter = 0;
			tableAssignments.add(new ArrayList<Integer>());
			for(int i:doc) //for each word in the document.
			{
				int tableId = gen.nextInt(K);
				tableAssignments.get(d).add(tableId);
				if(tableCounts.containsKey(tableId))
				{
					int prevCount = tableCounts.get(tableId);
					tableCounts.put(tableId, prevCount+1);
				}
				else				
					tableCounts.put(tableId, 1);
				tableCountsPerDoc[tableId][d]++; //because in table 'tableId', one more customer is sitting
				//Now table 'tableId' has a new word 'i'. Therefore we will have to update the params of the table (topic)
				updateTableParams(tableId, i, false);
				wordCounter++;
			}
		}
//		Commenting this because, we have to update the table mean, covariances etc which is kinda hard. Previously we were calculating this before initializing the
//		table params.
//		//Now some tables might be empty because of random initialization. But since we need continuous table indexes therefore I am going to make them continuous		
//		ArrayList<Integer> emptyTables = new ArrayList<Integer>();
//		for(int i=0;i<K;i++)		
//			if(!tableCounts.containsKey(i))			
//				emptyTables.add(i);
//		if(emptyTables.size()>0) //empty tables found
//		{
//			int start = 0, end = emptyTables.size()-1;
//			while(start <= end)
//			{
//				if(tableCounts.containsKey(K-1))//shift the contents of the last table to the first table which is non empty
//				{
//					int targetTableId = emptyTables.get(start);
//					tableCounts.put(targetTableId, tableCounts.get(K-1));
//					
//					//This is going to be expensive, go over the tableAssignments datastructure and change the asssignment of those who were assigned to K-1 to targetTableId
//					for(ArrayList<Integer> doc:tableAssignments)
//					{
//						int counter = 0;
//						for(int tableAssignment:doc)
//						{
//							if(tableAssignment == K-1)
//								doc.set(counter, targetTableId);
//							counter++;
//						}
//					}
//					tableCounts.remove(K-1);
//					start++; //incrementing start to point at the next table					
//				}
//				else				
//					end--; //this condition means that the last of the remaining table is empty, hence safely ignoring
//				
//				K = K - 1;
//			}
//		}		
		//double check again
		for(int i=0;i<K;i++)		
			if(!tableCounts.containsKey(i))
			{
				System.out.println("Still some tables are empty....exiting!");
				runLogger.write("Still some tables are empty....exiting!");
				System.exit(1);
			}		
		
		runLogger.write("Initialization complete\n");
		System.out.println("Initialization complete");
		//calculate initial avg ll
		double avgLL = Util.calculateAvgLL(corpus, tableAssignments, dataVectors, tableMeans, tableCholeskyLTriangularMat, K,N,prior,tableCountsPerDoc);
		System.out.println("Average ll at the begining "+avgLL);
		runLogger.write("Average ll at the begining "+avgLL+"\n");
		perplexities.write(avgLL+"\n");
		runLogger.flush();
	}
	private static double logMultivariateTDensity(DenseMatrix64F x, int tableId)
	{
		double logprob = 0.0;
		int count = tableCounts.get(tableId);
		double k_n = prior.k_0 + count;
		double nu_n = prior.nu_0 + count;		
		double scaleTdistrn = Math.sqrt((k_n + 1)/(k_n * (nu_n - Data.D + 1)));
		double nu = prior.nu_0 + count - Data.D + 1;
		//Since I am storing lower triangular matrices, therefore it is easy to calculate the value of (x-\mu)^T\Sigma^-1(x-\mu)
		//therefore I am gonna use triangular solver
		//first calculate (x-mu)
		DenseMatrix64F x_minus_mu = new DenseMatrix64F(Data.D,1);
		CommonOps.sub(x, tableMeans.get(tableId), x_minus_mu);
		//now scale the lower triangular matrix
		DenseMatrix64F lTriangularChol = new DenseMatrix64F(Data.D, Data.D);
		CommonOps.scale(scaleTdistrn, tableCholeskyLTriangularMat.get(tableId), lTriangularChol);
		TriangularSolver.solveL(lTriangularChol.data, x_minus_mu.data, Data.D); //now x_minus_mu has the solved value
		//Now take xTx
		DenseMatrix64F x_minus_mu_T = new DenseMatrix64F(1,Data.D);		
		CommonOps.transpose(x_minus_mu, x_minus_mu_T);
		DenseMatrix64F mul = new DenseMatrix64F(1,1);
		CommonOps.mult(x_minus_mu_T,x_minus_mu, mul);
		double val = mul.get(0, 0);
		logprob = Gamma.logGamma((nu + Data.D)/2) - (Gamma.logGamma(nu/2) + Data.D/2 * (Math.log(nu)+Math.log(Math.PI))+logDeterminants.get(tableId) + (nu + Data.D)/2* Math.log(1+val/nu));
		return logprob;
	}
	/**
	 * for num_iters:
	 * 	for each customer
	 * 		remove him from his old_table and update the table params.
	 * 		if old_table is empty:
	 * 			remove table
	 * 		Calculate prior and likelihood for this customer sitting at each table
	 * 		sample for a table index
	 * 		if new_table is equal to old_table
	 * 			don't have to update the parameters
	 * 		else update params of the old table. 
	 * @throws IOException 
	 */
	private static void sample() throws IOException
	{
		BufferedWriter out = null;
		try {
			out = new BufferedWriter(new OutputStreamWriter(
				    new FileOutputStream(dirName+"table_members.txt"), "UTF-8"));
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
		initRun();
		Thread t1 = (new Thread(new GaussianLDAAlias()));
		t1.start();
		for(currentIteration = 0;currentIteration<numIterations;currentIteration++)
		{	
			long startTime = System.currentTimeMillis();
			runLogger.write("Starting iteration "+currentIteration+"\n");
			System.out.println("Starting iteration "+currentIteration);
			for(int d=0;d<corpus.size();d++)
			{
				ArrayList<Integer> document = corpus.get(d);
				int wordCounter = 0;
				for(int custId:document)
				{
					//remove custId from his old_table
					int oldTableId = tableAssignments.get(d).get(wordCounter); 
					tableAssignments.get(d).set(wordCounter,-1);
					int oldCount = tableCounts.get(oldTableId);													 
					tableCounts.put(oldTableId, oldCount-1); //decrement count
					tableCountsPerDoc[oldTableId][d]--; //topic 'oldTableId' has one member less.
					//now recalculate table parameters for this table
					updateTableParams(oldTableId, custId, true);					
					//Now calculate the prior and likelihood for the customer to sit in each table and sample.				
					ArrayList<Double> posterior = new ArrayList<Double>();
					ArrayList<Integer> nonZeroTopicIndex = new ArrayList<Integer>();
					Double max = Double.NEGATIVE_INFINITY;
					double pSum = 0;
					//go over each table				
					for(int k=0;k<K;k++)
					{		
						if(tableCountsPerDoc[k][d] > 0)
						{
							//Now calculate the likelihood	
							//double count = tableCountsPerDoc[k][d]+alpha;//here count is the number of words of the same doc which are sitting in the same topic.
							double logLikelihood = logMultivariateTDensity(dataVectors[custId],k);					
							//System.out.println(custId+" "+k+" "+logLikelihood);
							//add log prior in the posterior vector
							//double logPosterior = Math.log(count) + logLikelihood;
							double logPosterior = Math.log(tableCountsPerDoc[k][d]) + logLikelihood;
							nonZeroTopicIndex.add(k);
							posterior.add(logPosterior);
							if(logPosterior > max)
								max = logPosterior;
						}
						
					}			
					//to prevent overflow, subtract by log(p_max). This is because when we will be normalizing after exponentiating, each entry will be exp(log p_i - log p_max )/\Sigma_i exp(log p_i - log p_max)
					//the log p_max cancels put and prevents overflow in the exponentiating phase.				
					for(int k=0;k<posterior.size();k++)
					{
						double p = posterior.get(k);
						p = p - max;
						double expP = Math.exp(p);		
						pSum += expP;
						posterior.set(k, pSum);
					}	
					//now sample an index from this posterior vector. The sample method will normalize the vector
					//so no need to normalize now.
					double select_pr = pSum / (pSum + alpha*q[custId].wsum);					

					//MHV to draw new topic
					Random rand = new Random();
					int newTableId = -1;
					for (int r = 0; r < MH_STEPS; ++r)
					{
						//1. Flip a coin	
						if(rand.nextDouble() < select_pr) 						
						{							
							double u = rand.nextDouble() * pSum;
							int temp = Util.binSearchArrayList(posterior, u, 0, posterior.size() - 1);
							newTableId = nonZeroTopicIndex.get(temp);
						}
						else
						{
							newTableId = q[custId].sampleVose();
						}
	
						if (oldTableId != newTableId)
						{
							//2. Find acceptance probability
							double temp_old = logMultivariateTDensity(dataVectors[custId],oldTableId);
							double temp_new = logMultivariateTDensity(dataVectors[custId],newTableId);
							double acceptance = (tableCountsPerDoc[newTableId][d] + alpha) / (tableCountsPerDoc[oldTableId][d] + alpha)
							*Math.exp(temp_new - temp_old)
							*(tableCountsPerDoc[oldTableId][d] * temp_old + alpha*q[custId].w[oldTableId])
							/ (tableCountsPerDoc[newTableId][d] * temp_new + alpha*q[custId].w[newTableId]);
		
							//3. Compare against uniform[0,1]
							double u = rand.nextDouble();
							if (u < acceptance)
							oldTableId = newTableId;
						}
					}
					tableAssignments.get(d).set(wordCounter, newTableId);					
					tableCounts.put(newTableId, tableCounts.get(newTableId)+1);					 
					tableCountsPerDoc[newTableId][d]++;
					updateTableParams(newTableId, custId, false);						
					wordCounter++;
				}
				if(d%10 == 0)
				{
					//runLogger.write("Done for document "+d+"\n");
					System.out.println("Done for document "+d);
					System.out.println("Time for document  "+d+" "+(System.currentTimeMillis() - startTime));
				}
			}								
			//Printing stuffs now
			runLogger.write("Iteration completed: "+currentIteration+"\n");
			long stopTime = System.currentTimeMillis();
		    double elapsedTime = (stopTime - startTime)/(double)1000;
		    runLogger.write("Time taken for this iteration "+elapsedTime+"\n");
			
			//calculate perplexity
			double avgLL = Util.calculateAvgLL(corpus, tableAssignments, dataVectors, tableMeans, tableCholeskyLTriangularMat, K, N, prior, tableCountsPerDoc);
			System.out.println("Avg log-likelihood at the end of iteration "+currentIteration+" is "+avgLL); 
			runLogger.write("Avg log-likelihood at the end of iteration "+currentIteration+" is "+avgLL+"\n");
			perplexities.write(avgLL+"\n");
			runLogger.flush();
			perplexities.flush();
		}
		done = true;
		try {
			t1.join();
		} catch (InterruptedException e1) {
			e1.printStackTrace();
		}
		try {
			out.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	public static void getTopWordsPerTopic()
	{
		
	}
	public static ArrayList<DenseMatrix64F> getTableMeans() {
		return tableMeans;
	}
	public static ArrayList<DenseMatrix64F> getTableCholeskyLTriangularMat() {
		return tableCholeskyLTriangularMat;
	}
	
	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub		
		long startTime = System.currentTimeMillis();
		//Get the input file given as input
		String inputFile = args[0];
		Data.inputFileName = inputFile;
		//First set the dimension of data which user has given input
		int D = Integer.parseInt(args[1]);
		numIterations = Integer.parseInt(args[2]);;
		Data.D = D;
		//read the initial number of clusters for k-means
		K = Integer.parseInt(args[3]);		
		//read the vocab and the cluster file
		dirName = args[4];			
		//Read data vectors into matrix from file
		DenseMatrix64F data = Data.readData();
		dataVectors = new DenseMatrix64F[data.numRows]; //splitting into vectors
		CommonOps.rowsToVector(data, dataVectors);
		System.out.println("Total number of vectors are "+data.numRows);
		//Read corpus
		String inputCorpusFile = args[5];
		corpus = Data.readCorpus(inputCorpusFile);
		System.out.println("Corpus file read");
		N = corpus.size();
		System.out.println("Total number of documents are "+N);
		//initialize the prior
		prior = new NormalInverseWishart();
		prior.mu_0 = Util.getSampleMean(dataVectors);
		//prior.mu_0 = new DenseMatrix64F(Data.D,1);//init to zeros
		prior.nu_0 = Data.D; //initializing to the dimension
		prior.sigma_0 = CommonOps.identity(Data.D); //setting as the identity matrix
		//CommonOps.scale(5*max, prior.sigma_0);
		CommonOps.scale(3*Data.D, prior.sigma_0);
		prior.k_0 = 0.1;
		CholSigma0 = new DenseMatrix64F(Data.D,Data.D);
		CommonOps.addEquals(CholSigma0, prior.sigma_0);
		alpha = 1/(double)K;
		if(!decomposer.decompose(CholSigma0))//cholesky decomp
		{
			System.out.println("Matrix couldnt be Cholesky decomposed");
			System.exit(1);
		}
		//Now initialize each datapoint (customer)
		tableAssignments = new ArrayList<ArrayList<Integer>>();		
		tableCountsPerDoc = new int[K][N];
		
		q = new VoseAlias[Data.numVectors];
		for(int w=0;w<Data.numVectors;w++)
		{
			q[w] = new VoseAlias();
			q[w].init(K);
		}
		
		/***Create log file*****/
		try {
			runLogger = new BufferedWriter(new OutputStreamWriter(
				    new FileOutputStream(dirName+"run_log.txt"), "UTF-8"));
			perplexities = new BufferedWriter(new OutputStreamWriter(
				    new FileOutputStream(dirName+"avgLL.txt"), "UTF-8"));
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		/**************** Initialize ***********/
		System.out.println("Starting to initialize");
		runLogger.write("Starting to initialize\n");
		runLogger.flush();
		initialize();		
		System.out.println("Gibbs sampler will run for "+numIterations+" iterations");
		runLogger.write("Gibbs sampler will run for "+numIterations+" iterations");
		runLogger.flush();
		/******sample*********/
		sample();		
		//System.out.println("Num of tables in each iteration: ");
		runLogger.write("Num of tables in each iteration: \n");
		long stopTime = System.currentTimeMillis();
	    double elapsedTime = (stopTime - startTime)/(double)1000;
	    //System.out.println("Time taken "+elapsedTime);
	    runLogger.write("Time taken "+elapsedTime);
	    System.out.println("Time taken "+elapsedTime);	    
	    runLogger.close();
	    perplexities.close();
	    //Print the gaussian
	    System.out.println("Printing the distributions");
	    Util.printGaussians(tableMeans, tableCholeskyLTriangularMat, K, dirName);	  
	    Util.printDocumentTopicDistribution(tableCountsPerDoc, N, K, dirName, alpha);
	    Util.printTableAssignments(tableAssignments, dirName);
	    Util.printNumCustomersPerTopic(tableCountsPerDoc, dirName, K, N);
	    System.out.println("Done");
	}
	
	@Override
	public void run() {
		
		VoseAlias temp = new VoseAlias();
		temp.init(K);
		//temp.init_temp();
		do{
			for (int w = 0; w < Data.numVectors; ++w)
			{
				double max = Double.NEGATIVE_INFINITY;
				for(int k=0;k<K;k++)
				{		
					double logLikelihood = logMultivariateTDensity(dataVectors[w],k);					
					//posterior.add(logLikelihood);
					temp.w[k] = logLikelihood;
					if(logLikelihood > max)
						max = logLikelihood;					
				}			
				//to prevent overflow, subtract by log(p_max). This is because when we will be normalizing after exponentiating, each entry will be exp(log p_i - log p_max )/\Sigma_i exp(log p_i - log p_max)
				//the log p_max cancels put and prevents overflow in the exponentiating phase.
				temp.wsum = 0.0;
				for(int k=0;k<K;k++)
				{
					double p = temp.w[k];
					p = p - max;
					double expP = Math.exp(p);		
					temp.wsum += expP;
					temp.w[k] = expP;
				}									
				temp.generateTable();						
				q[w].copy(temp);				
			}
		} while (!done);
		
	}
	
	
	public static void initRun()
	{
		VoseAlias temp = new VoseAlias();
		temp.init(K);
		//temp.init_temp();		
		for (int w = 0; w < Data.numVectors; ++w)
		{
			double max = Double.NEGATIVE_INFINITY;
			for(int k=0;k<K;k++)
			{		
				double logLikelihood = logMultivariateTDensity(dataVectors[w],k);					
				//posterior.add(logLikelihood);
				temp.w[k] = logLikelihood;
				if(logLikelihood > max)
					max = logLikelihood;					
			}			
			//to prevent overflow, subtract by log(p_max). This is because when we will be normalizing after exponentiating, each entry will be exp(log p_i - log p_max )/\Sigma_i exp(log p_i - log p_max)
			//the log p_max cancels put and prevents overflow in the exponentiating phase.
			temp.wsum = 0.0;
			for(int k=0;k<K;k++)
			{
				double p = temp.w[k];
				p = p - max;
				double expP = Math.exp(p);		
				temp.wsum += expP;
				temp.w[k] = expP;
			}									
			temp.generateTable();						
			q[w].copy(temp);				
		}
		
	}
	
}
