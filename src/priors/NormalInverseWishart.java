package priors;

import org.ejml.data.DenseMatrix64F;
import data.Data;



public class NormalInverseWishart {
	
	/**
	 * Hyperparam mean vector. 
	 */
	public  DenseMatrix64F mu_0 ;
	
	/**
	 * initial degrees of freedom 
	 */
	public  double nu_0;
	
	/**
	 * Hyperparam covariance matrix
	 */
	public  DenseMatrix64F sigma_0 ;
	
	/**
	 * mean fraction
	 */
	public  double k_0;
	
	

	public  DenseMatrix64F getMu_0() {
		return mu_0;
	}

	public  void setMu_0(DenseMatrix64F mu_0) {
		this.mu_0 = mu_0;
	}

	public  double getNu_0() {
		return nu_0;
	}

	public  void setNu_0(double nu_0) {
		this.nu_0 = nu_0;
	}

	public  DenseMatrix64F getSigma_0() {
		return sigma_0;
	}

	public  void setSigma_0(DenseMatrix64F sigma_0) {
		this.sigma_0 = sigma_0;
	}

	public  double getK_0() {
		return k_0;
	}

	public  void setK_0(double k_0) {
		this.k_0 = k_0;
	}
	
	
	
}
