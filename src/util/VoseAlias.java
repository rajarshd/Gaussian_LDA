package util;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Random;

public class VoseAlias {
	
	public double[]w;
	public double[] Prob;	//Contains proportions and alias probabilities
	public double wsum;		//Sum of proportions
	public int[] Alias;			//Alias indices
	public int n;				//Dimension
	public double[] p; //temp
	public Random gen ;
	
	public void init(int num)
	{
		n = num;
		Alias = new int[n];
		Prob = new double[n];
		w = new double[n];
		p = new double[n];
		gen = new Random();
	}
	
	public void copy(VoseAlias other)
	{
		n = other.n;
		wsum = other.wsum;
		//std::copy(&other.w[0], &other.w[n], w);
		w = Arrays.copyOf(other.w, other.w.length);
		//std::copy(&other.Prob[0], &other.Prob[n], Prob);
		Prob = Arrays.copyOf(other.Prob, other.Prob.length);
		//std::copy(&other.Alias[0], &other.Alias[n], Alias);
		Alias = Arrays.copyOf(other.Alias, other.Alias.length);		
	}
	
	public void generateTable()
	{
		//1. Create two worklists, Small and Large.
		Queue<Integer> Small = new LinkedList<Integer>();
		Queue<Integer> Large = new LinkedList<Integer>();;
		
		//2. Multiply each probability by n.
		for (int i = 0; i < n; ++i)
		{
			p[i] = (w[i] * n) / wsum;
		}
		
		//3. For each scaled probability pi:
		//		a. If pi<1, add i to Small.
		//		b. Otherwise(pi≥1), add i to Large.
		for (int i = 0; i < n; ++i)
		{
			if (p[i] < 1)
				Small.add(i);
			else
				Large.add(i);
		}
		
		//4. While Small and Large are not empty : (Large might be emptied first)
		//		a. Remove the first element from Small; call it l.
		//		b. Remove the first element from Large; call it g.
		//		c. Set Prob[l] = pl.
		//		d. Set Alias[l] = g.
		//		e. Set pg : = (pg + pl)−1. (This is a more numerically stable option.)
		//		f. If pg<1, add g to Small.
		//		g. Otherwise(pg≥1), add g to Large.
		while (!(Small.isEmpty() || Large.isEmpty()))
		{
			int l = Small.remove(); 
			int g = Large.remove();
			Prob[l] = p[l];
			Alias[l] = g;
			p[g] = (p[g] + p[l]) - 1;
			if (p[g] < 1)
				Small.add(g);
			else
				Large.add(g);
		}
		//5. While Large is not empty :
		//		a. Remove the first element from Large; call it g.
		//		b. Set Prob[g] = 1.
		while (!Large.isEmpty())
		{
			int g = Large.remove();
			Prob[g] = 1;
		}

		//6. While Small is not empty : This is only possible due to numerical instability.
		//		a. Remove the first element from Small; call it l.
		//		b. Set Prob[l] = 1.
		while (!Small.isEmpty())
		{
			int l = Small.remove();
			Prob[l] = 1;
		}
		
	}
	
	public int sampleVose()
	{
		//1. Generate a fair die roll from an n-sided die; call the side i.
		//int fair_die = utils::pick_a_number(0, n - 1);
		int fair_die = gen.nextInt(n);
		//2. Flip a biased coin that comes up heads with probability Prob[i].
		double m = gen.nextDouble();		
		int res = fair_die;
		if(m>Prob[fair_die])
			res = Alias[fair_die];		
		//3. If the coin comes up "heads," return i. Otherwise, return Alias[i].
		return res ;
	}
	
	

}
