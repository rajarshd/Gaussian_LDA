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
import java.util.Set;
import java.util.Map.Entry;

import org.apache.commons.math3.special.Gamma;
import org.ejml.alg.dense.decomposition.TriangularSolver;
import org.ejml.data.DenseMatrix64F;
import org.ejml.factory.DecompositionFactory;
import org.ejml.factory.LinearSolverFactory;
import org.ejml.interfaces.decomposition.CholeskyDecomposition;
import org.ejml.interfaces.linsol.LinearSolver;
import org.ejml.ops.CommonOps;
import org.ejml.ops.RandomMatrices;

import priors.NormalInverseWishart;
import util.Util;
import data.Data;


/**
 * Implementation of the collapsed gibbs sampler for Dirichlet Process Mixture Models where the distribution of each table are multivariate gaussian
 * with unknown mean and covariances. I have extensively referred Frank Wood's matalab implementation (http://www.robots.ox.ac.uk/~fwood/code/index.html) 
 * @author rajarshd
 *
 */
public class GaussianLDAWithoutCholesky {

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
	//private static ArrayList<DenseMatrix64F> tableCholeskyLTriangularMat = new ArrayList<DenseMatrix64F>();
	
	/**
	 * log-determinant of covariance matrix for each table. Since 0.5*logDet is required in (see logMultivariateTDensity), therefore that value is kept.
	 */
	//private static ArrayList<Double> logDeterminants = new ArrayList<Double>();
	
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
	//private static DenseMatrix64F CholSigma0; 
	
	/**
	 * file path for reading vocab (to form mapping) and the initial cluster assignment
	 */
	private static String dirName;
	private static BufferedWriter runLogger = null;
	private static BufferedWriter perplexities = null;
	//the dirichlet hyperparam.
	private static double alpha;
	
	
	
	/**
	 * inverse of covariance matrix associated with each table in the current iteration. The covariance matrix is scaled before taking the inverse by \frac{k_N + 1}{k_N (v_N - D + 1)}
	 * (This is because the t-distribution take the matrix scaled as input)
	 */
	private static ArrayList<DenseMatrix64F> tableInverseCovariances = new ArrayList<DenseMatrix64F>();
	
	/**
	 * determinant of covariance matrix for each table.The covariance matrix is scaled before taking the inverse by \frac{k_N + 1}{k_N (v_N - D + 1)}.(This is because the t-distribution take the matrix scaled as input)
	 */
	private static ArrayList<Double> determinants = new ArrayList<Double>();
	
	/**
	 * stores the sum of the vectors of customers at a given table
	 */
	private static ArrayList<DenseMatrix64F> sumTableCustomers = new ArrayList<DenseMatrix64F>();
	
	
	/**
	 * stores the squared sum of the vectors of customers at a given table
	 */
	private static ArrayList<DenseMatrix64F> sumSquaredTableCustomers = new ArrayList<DenseMatrix64F>();
	
	/**
	 *Caching a value. Look into calculateTableParams.
	 */
	private static DenseMatrix64F k0mu0mu0T = null;
	
	private static LinearSolver<DenseMatrix64F> newSolver = LinearSolverFactory.general(Data.D, Data.D);
	
/************************************Member Declaration Ends***********************************/	
	/**
	 * updates params -- mean and the cholesky decomp of covariance matrix using rank1 update (customer added) or downdate (customer removed)
	 * @param tableId
	 * @param custId
	 * @param isRemoved
	 */
//	private static void updateTableParams(int tableId,int custId, boolean isRemoved)
//	{
//		int count = tableCounts.get(tableId);
//		double k_n = prior.k_0 + count;
//		double nu_n = prior.nu_0 + count;		
//		double scaleTdistrn = (k_n + 1)/(k_n * (nu_n - Data.D + 1));
//		
//		DenseMatrix64F oldLTriangularDecomp = tableCholeskyLTriangularMat.get(tableId);		
//		if(isRemoved)
//		{
//			/**
//			 * Now use the rank1 downdate to calculate the cholesky decomposition of the updated covariance matrix
//			 * the update equaltion is \Sigma_(N+1) =\Sigma_(N) - (k_0 + N+1)/(k_0 + N)(X_{n} - \mu_{n-1})(X_{n} - \mu_{n-1})^T
//			 * therefore x = sqrt((k_0 + N - 1)/(k_0 + N)) (X_{n} - \mu_{n})
//			 * Note here \mu_n will be the mean before updating. After updating sigma_n, we will update \mu_n.
//			 */			
//			DenseMatrix64F x = new DenseMatrix64F(Data.D, 1);
//			CommonOps.sub(dataVectors[custId], tableMeans.get(tableId), x); //calculate (X_{n} - \mu_{n-1})
//			double coeff = Math.sqrt((k_n+1)/k_n);
//			CommonOps.scale(coeff, x);
//			Util.cholRank1Downdate(oldLTriangularDecomp, x);
//			tableCholeskyLTriangularMat.set(tableId, oldLTriangularDecomp);//the cholRank1Downdate modifies the oldLTriangularDecomp, therefore putting it back to the map			
//			//updateMean(tableId);
//			DenseMatrix64F newMean = new DenseMatrix64F(Data.D, 1);
//			CommonOps.scale(k_n+1, tableMeans.get(tableId), newMean);
//			CommonOps.subEquals(newMean, dataVectors[custId]);
//			CommonOps.divide(k_n, newMean);
//			tableMeans.set(tableId, newMean);
//			
//		}
//		else //new customer is added
//		{			
//			DenseMatrix64F newMean = new DenseMatrix64F(Data.D, 1);
//			CommonOps.scale(k_n-1, tableMeans.get(tableId), newMean);
//			CommonOps.addEquals(newMean, dataVectors[custId]);
//			CommonOps.divide(k_n, newMean);
//			tableMeans.set(tableId, newMean);
//			/**
//			 * The rank1 update equation is
//			 * \Sigma_{n+1} = \Sigma_{n} + (k_0 + n + 1)/(k_0 + n) * (x_{n+1} - \mu_{n+1})(x_{n+1} - \mu_{n+1})^T
//			 */
//			DenseMatrix64F x = new DenseMatrix64F(Data.D, 1);
//			CommonOps.sub(dataVectors[custId], tableMeans.get(tableId), x); //calculate (X_{n} - \mu_{n-1})
//			double coeff = Math.sqrt(k_n/(k_n - 1));
//			CommonOps.scale(coeff, x);
//			Util.cholRank1Update(oldLTriangularDecomp, x);
//			tableCholeskyLTriangularMat.set(tableId, oldLTriangularDecomp);//the cholRank1Downdate modifies the oldLTriangularDecomp, therefore putting it back to the map
//		}
//		//calculate the 0.5*log(det) + D/2*scaleTdistrn; the scaleTdistrn is because the posterior predictive distribution sends in a scaled value of \Sigma
//		double logDet = 0.0;
//		for(int l=0;l<Data.D;l++)				
//			logDet = logDet + Math.log(oldLTriangularDecomp.get(l, l));	
//		logDet += Data.D*Math.log(scaleTdistrn)/(double)2;
//		
//		if(tableId < logDeterminants.size())				
//			logDeterminants.set(tableId, logDet);
//		else
//			logDeterminants.add(logDet);
//		
//	}
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
		//double scaleTdistrn = (prior.k_0+1)/(double)(prior.k_0*(prior.nu_0 - Data.D + 1));
		
		double degOfFreedom = prior.nu_0 - Data.D + 1;
		//Now calculate the covariance matrix of the multivariate T-distribution
		double coeff = (double) (prior.k_0 + 1)/(prior.k_0 * (degOfFreedom)) ;
		DenseMatrix64F sigma_T = new DenseMatrix64F(Data.D,Data.D);
		CommonOps.scale(coeff, prior.sigma_0,sigma_T);
		
		LinearSolver<DenseMatrix64F> solver = LinearSolverFactory.symmPosDef(Data.D);
		if( !solver.setA(sigma_T) )
			throw new RuntimeException("Invert failed");
		DenseMatrix64F sigma_TInv = new DenseMatrix64F(Data.D,Data.D);
		solver.invert(sigma_TInv);
		
		double sigmaTDet = CommonOps.det(sigma_T);
		
		
//		for(int i=0;i<K;i++)
//		{
//			DenseMatrix64F priorMean = new DenseMatrix64F(prior.mu_0);
//			//DenseMatrix64F initialCholesky = new DenseMatrix64F(CholSigma0);
//			//calculate the 0.5*log(det) + D/2*scaleTdistrn; the scaleTdistrn is because the posterior predictive distribution sends in a scaled value of \Sigma
//			double logDet = 0.0;			
//			for(int l=0;l<Data.D;l++)				
//				logDet = logDet + Math.log(CholSigma0.get(l, l));	
//			logDet += Data.D*Math.log(scaleTdistrn)/(double)2;
//			logDeterminants.add(logDet);			
//			tableMeans.add(priorMean);			
//			tableCholeskyLTriangularMat.add(initialCholesky);
//		}					
		
		//storing zeros in sumTableCustomers and later will keep on adding each customer. Also initialize tableInverseCovariances and determinants
		for(int i=0;i<K;i++)
		{
			DenseMatrix64F zero = new DenseMatrix64F(Data.D,1);
			sumTableCustomers.add(zero);
			zero = new DenseMatrix64F(Data.D,Data.D);
			sumSquaredTableCustomers.add(zero);
			zero = new DenseMatrix64F(Data.D,Data.D);
			tableInverseCovariances.add(zero);
			determinants.add(0.0);
			zero = new DenseMatrix64F(Data.D,1);
			tableMeans.add(zero);
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
				//update the sumTableCustomers
				DenseMatrix64F sum = sumTableCustomers.get(tableId);
				CommonOps.add(dataVectors[i], sum, sum);
				//update the sumSquaredTableCustomers
				DenseMatrix64F sumSquared = sumSquaredTableCustomers.get(tableId);
				DenseMatrix64F custVectorTranspose = new DenseMatrix64F(1,Data.D);
				DenseMatrix64F squaredVector = new DenseMatrix64F(Data.D,Data.D);	
				custVectorTranspose = CommonOps.transpose(dataVectors[i], custVectorTranspose);
				//Multiply x_ix_i^T
				CommonOps.mult(dataVectors[i],custVectorTranspose,squaredVector);
				CommonOps.add(sumSquared,squaredVector,sumSquared);			
				//Now table 'tableId' has a new word 'i'. Therefore we will have to update the params of the table (topic)
				//updateTableParams(tableId, i, false);
				wordCounter++;
			}
		}
		
		//Now compute the table parameters of each table
		//go over each table.
		Set<Entry<Integer,Integer>> tableIdAndCounts = tableCounts.entrySet();
		for(Entry<Integer,Integer> table:tableIdAndCounts)
		{
			int id = table.getKey();
			calculateTableParams(id);
		}		
		System.out.println("Initialization complete");
		
		
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
//		double avgLL = Util.calculateAvgLL(corpus, tableAssignments, dataVectors, tableMeans, tableCholeskyLTriangularMat, K,N,prior,tableCountsPerDoc);
//		System.out.println("Average ll at the begining "+avgLL);
		//runLogger.write("Average ll at the begining "+avgLL+"\n");
//		perplexities.write(avgLL+"\n");
		runLogger.flush();
	}
	
	/**
	 * This method calculates the table params (bayesian mean, covariance^-1, determinant etc.)
	 * All it needs is the table_id and the tableCounts, tableMembers and sumTableCustomers should be updated correctly before calling this. 
	 * @param tableId
	 */
	private static void calculateTableParams(int tableId)
	{
		
			int count = tableCounts.get(tableId);
			double nu_n = prior.nu_0 + count;
			double k_n = prior.k_0 + count;
			
			//calculate mu_n
			DenseMatrix64F mu_n = new DenseMatrix64F(Data.D,1);
			CommonOps.scale(prior.k_0, prior.mu_0, mu_n); //k_0 * mu_o			
			//Now add N X_bar
			CommonOps.add(mu_n,sumTableCustomers.get(tableId),mu_n);
			CommonOps.divide(k_n, mu_n, mu_n); // divide by k_n
			if(tableMeans.size() > tableId)
				tableMeans.set(tableId,mu_n);
			else //for new table
			{
				assert tableId <= tableMeans.size();
				tableMeans.add(mu_n);
			}
			//Re-calculate Sigma_N
			//Sigma_N = Sigma_0 + C + k_0 * N /(k_0 + N) (\bar(X) - \mu_0) (\bar(X) - \mu_0)^T
			// C = \Sigma_{i=1}^{N} (X_i - \bar(X)) (X_i - \bar(X))^T, Since there is one customer at the table, C is zero
//			DenseMatrix64F xBar = new DenseMatrix64F(Data.D,1);
//			CommonOps.divide(count, sumTableCustomers.get(tableId), xBar);
//			//Now calculate C
//			DenseMatrix64F C = new DenseMatrix64F(Data.D, Data.D);
//			Set<Integer> members = tableMembers.get(tableId);
//			for(int member:members)
//			{
//				DenseMatrix64F xi_xBar = new  DenseMatrix64F(Data.D,1);
//				CommonOps.sub(dataVectors[member], xBar, xi_xBar);
//				DenseMatrix64F xi_xBar_T = new  DenseMatrix64F(1,Data.D);
//				xi_xBar_T = CommonOps.transpose(xi_xBar,xi_xBar_T);
//				DenseMatrix64F mul = new DenseMatrix64F(Data.D, Data.D);
//				CommonOps.mult(xi_xBar, xi_xBar_T, mul);
//				//System.out.println(mul.get(1, 2));
//				//System.out.println("i");
//				CommonOps.add(mul, C, C);
//			}	
//			//Now calculate k_0 * N /(k_0 + N) (\bar(X) - \mu_0) (\bar(X) - \mu_0)^T
//			DenseMatrix64F xBar_x0 = new  DenseMatrix64F(Data.D,1);
//			DenseMatrix64F xBar_x0_T = new  DenseMatrix64F(1,Data.D);
//			CommonOps.sub(xBar, prior.mu_0, xBar_x0); //calculating (\bar(X) - \mu_0)
//			//Now compute the transpose
//			xBar_x0_T = CommonOps.transpose(xBar_x0, xBar_x0_T);
//			DenseMatrix64F xBar_x0_xBar_x0_T = new DenseMatrix64F(Data.D,Data.D);
//			CommonOps.mult(xBar_x0, xBar_x0_T, xBar_x0_xBar_x0_T);
//			//calculate the scaling term k_0 * N /(k_0 + N)
//			double scale = (prior.k_0 * count)/(double)(k_n);
//			//calculate k_0 * N /(k_0 + N) (\bar(X) - \mu_0) (\bar(X) - \mu_0)^T
//			CommonOps.scale(scale, xBar_x0_xBar_x0_T, xBar_x0_xBar_x0_T);
//			//Now add Sigma_0, then C, then xBar_x0_xBar_x0_T
//			DenseMatrix64F sigmaN = new DenseMatrix64F(Data.D,Data.D);
//			CommonOps.add(prior.sigma_0,C, sigmaN);			
//			CommonOps.add(sigmaN,xBar_x0_xBar_x0_T, sigmaN);
//			//System.out.println(sigmaN);
//			//Now will scale the covariance matrix by (k_N + 1)/(k_N (v_N - D + 1)) (This is because the t-distribution take the matrix scaled as input)
//			double scaleTdistrn = (k_n + 1)/(k_n * (nu_n - Data.D + 1));
//			CommonOps.scale(scaleTdistrn, sigmaN, sigmaN);
			
			//we will be using the new update
			//Sigma_N = Sigma_0 + \sum(y_iy_i^T) - (k_n)\mu_N\mu_N^T + k_0\mu_0\mu_0^T
			//calculate \mu_N\mu_N^T
			DenseMatrix64F mu_n_T = new DenseMatrix64F(1,Data.D);
			mu_n_T = CommonOps.transpose(mu_n, mu_n_T);
			DenseMatrix64F mu_n_mu_nT = new DenseMatrix64F(Data.D,Data.D);
			CommonOps.mult(mu_n, mu_n_T, mu_n_mu_nT);
			CommonOps.scale(k_n,mu_n_mu_nT);
			
			
			//cache k_0\mu_0\mu_0^T, only compute it once
			if(k0mu0mu0T == null)
			{
				//compute mu0^T
				DenseMatrix64F mu0T = new DenseMatrix64F(1,Data.D);
				mu0T = CommonOps.transpose(prior.mu_0, mu0T);
				k0mu0mu0T = new DenseMatrix64F(Data.D,Data.D);
				CommonOps.mult(prior.mu_0, mu0T, k0mu0mu0T);
				CommonOps.scale(prior.k_0, k0mu0mu0T);
			}
			DenseMatrix64F sigmaN = new DenseMatrix64F(Data.D,Data.D);
			CommonOps.add(prior.sigma_0, sumSquaredTableCustomers.get(tableId), sigmaN);
			CommonOps.subEquals(sigmaN, mu_n_mu_nT);
			CommonOps.add(sigmaN, k0mu0mu0T, sigmaN);
			double scaleTdistrn = (k_n + 1)/(k_n * (nu_n - Data.D + 1));
			CommonOps.scale(scaleTdistrn, sigmaN, sigmaN);
			
			//calculate det(Sigma)
			double det = CommonOps.det(sigmaN);
			//System.out.println(det+" "+count);
			//System.out.println(det);
			if(tableId < determinants.size())
				determinants.set(tableId, det);
			else
				determinants.add(det);
			//Now calculate Sigma^(-1) and det(Sigma) and store them
			//calculate Sigma^(-1)
			if( !newSolver.setA(sigmaN) )
				throw new RuntimeException("Invert failed");
			
			DenseMatrix64F sigmaNInv = new DenseMatrix64F(Data.D,Data.D);
			newSolver.invert(sigmaNInv);
			if(tableId < tableInverseCovariances.size())
				tableInverseCovariances.set(tableId, sigmaNInv);//storing the inverse covariances
			else
				tableInverseCovariances.add(sigmaNInv);			
	}
	
	
	
//	private static double logMultivariateTDensity(DenseMatrix64F x, int tableId)
//	{
//		double logprob = 0.0;
//		int count = tableCounts.get(tableId);
//		double k_n = prior.k_0 + count;
//		double nu_n = prior.nu_0 + count;		
//		double scaleTdistrn = Math.sqrt((k_n + 1)/(k_n * (nu_n - Data.D + 1)));
//		double nu = prior.nu_0 + count - Data.D + 1;
//		//Since I am storing lower triangular matrices, therefore it is easy to calculate the value of (x-\mu)^T\Sigma^-1(x-\mu)
//		//therefore I am gonna use triangular solver
//		//first calculate (x-mu)
//		DenseMatrix64F x_minus_mu = new DenseMatrix64F(Data.D,1);
//		CommonOps.sub(x, tableMeans.get(tableId), x_minus_mu);
//		//now scale the lower triangular matrix
//		DenseMatrix64F lTriangularChol = new DenseMatrix64F(Data.D, Data.D);
//		CommonOps.scale(scaleTdistrn, tableCholeskyLTriangularMat.get(tableId), lTriangularChol);
//		TriangularSolver.solveL(lTriangularChol.data, x_minus_mu.data, Data.D); //now x_minus_mu has the solved value
//		//Now take xTx
//		DenseMatrix64F x_minus_mu_T = new DenseMatrix64F(1,Data.D);		
//		CommonOps.transpose(x_minus_mu, x_minus_mu_T);
//		DenseMatrix64F mul = new DenseMatrix64F(1,1);
//		CommonOps.mult(x_minus_mu_T,x_minus_mu, mul);
//		double val = mul.get(0, 0);
//		logprob = Gamma.logGamma((nu + Data.D)/2) - (Gamma.logGamma(nu/2) + Data.D/2 * (Math.log(nu)+Math.log(Math.PI))+logDeterminants.get(tableId) + (nu + Data.D)/2* Math.log(1+val/nu));
//		return logprob;
//	}
	
	/**
	 * @param x data point
	 * @param mu
	 * @param sigmaInv
	 * @param det
	 * @param nu
	 * @param forEmptyTable : true if you are calculating the likelihood for the empty table. In that case, I am doing a slight saving in computation, since there is a constant part
	 * of the multi-variate T dist which is independent of the datapoint
	 * @return
	 */
	//private static double logMultivariateTDensity(DenseMatrix64F x, DenseMatrix64F mu, DenseMatrix64F sigmaInv,double det, double nu)
	private static double logMultivariateTDensity(DenseMatrix64F x, int tableId)
	{
		DenseMatrix64F mu = tableMeans.get(tableId);
		DenseMatrix64F sigmaInv = tableInverseCovariances.get(tableId);
		double det = determinants.get(tableId);
		int count = tableCounts.get(tableId); //this is the prior					
		//Now calculate the likelihood
		//calculate degrees of freedom of the T-distribution
		double nu = prior.nu_0 + count - Data.D + 1;
		//calculate (x = mu)
		DenseMatrix64F xMinusMu = new DenseMatrix64F(Data.D,1);
		CommonOps.sub(x ,mu, xMinusMu);
		//take the transpose
		DenseMatrix64F xMinusMuTrans = new DenseMatrix64F(1,Data.D);
		xMinusMuTrans = CommonOps.transpose(xMinusMu, xMinusMuTrans);
		//Calculate (x = mu)^TSigma^(-1)(x = mu)
		DenseMatrix64F prod = new DenseMatrix64F(1, Data.D);
		//System.out.println(xMinusMuTrans.numRows+" "+xMinusMuTrans.numCols);
		//System.out.println(sigmaInv.numRows+" "+sigmaInv.numCols);
		CommonOps.mult(xMinusMuTrans, sigmaInv, prod);
		DenseMatrix64F prod1 = new DenseMatrix64F(1,1);
		CommonOps.mult(prod,xMinusMu , prod1);		
		//Finally get the value in a double.
		assert prod1.numCols == 1;
		assert prod1.numRows == 1;
		double val = prod1.get(0, 0); //prod1 is a 1x1 matrix
		//System.out.println("val = "+val);
		//System.out.println("det = "+det);
		double logprob = 0.0;		
		logprob = Gamma.logGamma((nu + Data.D)/2) - (Gamma.logGamma(nu/2) + Data.D/2 * (Math.log(nu)+Math.log(Math.PI)) + 0.5 * Math.log(det) + (nu + Data.D)/2* Math.log(1+val/nu));		
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
					DenseMatrix64F custVectorTranspose = new DenseMatrix64F(1,Data.D);
					custVectorTranspose = CommonOps.transpose(dataVectors[custId], custVectorTranspose);
					DenseMatrix64F squaredVector = new DenseMatrix64F(Data.D,Data.D);	
					//Multiply x_ix_i^T
					CommonOps.mult(dataVectors[custId],custVectorTranspose,squaredVector);
					//remove custId from his old_table
					int oldTableId = tableAssignments.get(d).get(wordCounter); 
					tableAssignments.get(d).set(wordCounter,-1);
					int oldCount = tableCounts.get(oldTableId);													 
					tableCounts.put(oldTableId, oldCount-1); //decrement count
					tableCountsPerDoc[oldTableId][d]--; //topic 'oldTableId' has one member less.
					DenseMatrix64F oldSumTable = sumTableCustomers.get(oldTableId);
					DenseMatrix64F oldSumSquaredTable = sumSquaredTableCustomers.get(oldTableId);
					DenseMatrix64F newSumTable = new DenseMatrix64F(Data.D,1);
					CommonOps.sub(oldSumTable, dataVectors[custId], newSumTable); //subtracting the vector of this customer.
					sumTableCustomers.set(oldTableId, newSumTable);
					
					//subtract x_ix_i^T								
					DenseMatrix64F newSumSquaredTable = new DenseMatrix64F(Data.D,Data.D);					
					CommonOps.sub(oldSumSquaredTable, squaredVector, newSumSquaredTable);
					sumSquaredTableCustomers.set(oldTableId, newSumSquaredTable);
					
					//now recalculate table parameters for this table
					calculateTableParams(oldTableId);
					//now recalculate table parameters for this table
					//updateTableParams(oldTableId, custId, true);					
					//Now calculate the prior and likelihood for the customer to sit in each table and sample.				
					ArrayList<Double> posterior = new ArrayList<Double>();
					Double max = Double.NEGATIVE_INFINITY;
					//go over each table				
					for(int k=0;k<K;k++)
					{		
						double count = tableCountsPerDoc[k][d]+alpha;//here count is the number of words of the same doc which are sitting in the same topic.						
						//Now calculate the likelihood														
						double logLikelihood = logMultivariateTDensity(dataVectors[custId],k);					
						//System.out.println(custId+" "+k+" "+logLikelihood);
						//add log prior in the posterior vector
						double logPosterior = Math.log(count) + logLikelihood;
						posterior.add(logPosterior);
						if(logPosterior > max)
							max = logPosterior;
					}			
					//to prevent overflow, subtract by log(p_max). This is because when we will be normalizing after exponentiating, each entry will be exp(log p_i - log p_max )/\Sigma_i exp(log p_i - log p_max)
					//the log p_max cancels put and prevents overflow in the exponentiating phase.				
					for(int k=0;k<K;k++)
					{
						double p = posterior.get(k);
						p = p - max;
						double expP = Math.exp(p);							
						posterior.set(k, expP);
					}	
					//now sample an index from this posterior vector. The sample method will normalize the vector
					//so no need to normalize now.
					int newTableId = Util.sample(posterior);
					tableAssignments.get(d).set(wordCounter, newTableId);					
					tableCounts.put(newTableId, tableCounts.get(newTableId)+1);					 
					tableCountsPerDoc[newTableId][d]++;
					DenseMatrix64F sum = sumTableCustomers.get(newTableId);
					CommonOps.add(dataVectors[custId],sum,sum);
					
					DenseMatrix64F sqSum = sumSquaredTableCustomers.get(newTableId);
					CommonOps.add(sqSum,squaredVector,sqSum);
					
					calculateTableParams(newTableId); //update the table params.
					//updateTableParams(newTableId, custId, false);				
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
			//double avgLL = Util.calculateAvgLL(corpus, tableAssignments, dataVectors, tableMeans, tableCholeskyLTriangularMat, K, N, prior, tableCountsPerDoc);
			//System.out.println("Avg log-likelihood at the end of iteration "+currentIteration+" is "+avgLL); 
			//runLogger.write("Avg log-likelihood at the end of iteration "+currentIteration+" is "+avgLL+"\n");
			//perplexities.write(avgLL+"\n");
			//runLogger.flush();
			//perplexities.flush();
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
//	public static ArrayList<DenseMatrix64F> getTableCholeskyLTriangularMat() {
//		return tableCholeskyLTriangularMat;
//	}
	
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
//		CholSigma0 = new DenseMatrix64F(Data.D,Data.D);
//		CommonOps.addEquals(CholSigma0, prior.sigma_0);
//		alpha = 1/(double)K;
//		if(!decomposer.decompose(CholSigma0))//cholesky decomp
//		{
//			System.out.println("Matrix couldnt be Cholesky decomposed");
//			System.exit(1);
//		}
		//Now initialize each datapoint (customer)
		tableAssignments = new ArrayList<ArrayList<Integer>>();		
		tableCountsPerDoc = new int[K][N];
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
//	    System.out.println("Printing the distributions");
//	    Util.printGaussians(tableMeans, tableCholeskyLTriangularMat, K, dirName);	  
//	    Util.printDocumentTopicDistribution(tableCountsPerDoc, N, K, dirName, alpha);
//	    Util.printTableAssignments(tableAssignments, dirName);
//	    Util.printNumCustomersPerTopic(tableCountsPerDoc, dirName, K, N);
//	    System.out.println("Done");
	}
}
