// mindist.cpp
//
// The algorithm for computing the minimum distance implemented here 
// is due to Brouwer. It has been described in 
// Betten, Fripertinger et al.~\cite{BettenCodes98}.

#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace combinatorics {
namespace coding_theory {


typedef struct mindist MINDIST;

//! internal class for the algorithm to compute the minimum distance of a linear code


struct mindist {
	int verbose_level;
	int f_v, f_vv, f_vvv;
	int k, n, d, q;
	int p, f;
	int **G;
	int ***S;
	int M;
	int K0;
	int ZC;
	int *Size;
	int *ff_mult; // [q][q]
	int *ff_add; // [q][q]
	int *ff_inv; // [q]
	int idx_zero;
	int idx_one;
	int idx_mone;
	int weight_computations;
};
//Local data structure for the mindist computation.
//Contains tables for the finite field structure.


static void print_matrix(MINDIST *MD, int **G);
static int min_weight(MINDIST *MD);
static void create_systematic_generator_matrices(MINDIST *MD);
static int weight_of_linear_combinations(MINDIST *MD, int t);
static int weight(int *v, int n, int idx_zero);
static void padic(int ind, int *v, int L, int A);
static int nextsub(int k, int l, int *sub);
static void vmmult(MINDIST *MD, int *v, int **mx, int *cv);

int coding_theory_domain::mindist(
		int n, int k, int q, int *G,
	int verbose_level, int idx_zero, int idx_one,
	int *add_table, int *mult_table)
// Main routine for the code minimum distance computation.
// The tables are only needed if $q = p^f$ with $f > 1$.
// In the GF(p) case, just pass a NULL pointer.
{
	MINDIST MD;
	int i, j, a, d;
	int p, e;
	//vector vp, ve;
	int wt_rows, w;
	algebra::number_theory::number_theory_domain NT;

	NT.factor_prime_power(q, p, e);
	MD.verbose_level = verbose_level;
	MD.f_v = (verbose_level >= 1);
	MD.f_vv = (verbose_level >= 2);
	MD.f_vvv = false;
	MD.k = k;
	MD.n = n;
	MD.q = q;
	MD.p = (int) p;
	MD.f = (int) e;
	MD.idx_zero = idx_zero;
	MD.idx_one = idx_one;
	MD.idx_mone = 0; // will be computed when we print the tables next
	MD.ff_mult = (int *) malloc(sizeof(int) * q * q);
	MD.ff_add = (int *) malloc(sizeof(int) * q * q);
	MD.ff_inv = (int *) malloc(sizeof(int) * q);
	MD.weight_computations = 0;
	if (MD.f > 1) {
		if (MD.f_v) {
			cout << "multiplication table:" << endl;
			}
		for (i = 0; i < q; i++) {
			for (j = 0; j < q; j++) {
				a = (int) mult_table[i * q + j];
				MD.ff_mult[i * q + j] = a;
				if (a == idx_one)  {
					MD.ff_inv[i] = j;
					if (i == j) 
						MD.idx_mone = i;
					}
				if (MD.f_v) {
					cout << a << " ";
					}
				}
			if (MD.f_v) {
				cout << endl;
				}
			
			}
		if (MD.f_v) {
			cout << "addition table:" << endl;
			}
		for (i = 0; i < q; i++) {
			for (j = 0; j < q; j++) {
				a = (int) add_table[i * q + j];
				MD.ff_add[i * q + j] = a;
				if (MD.f_v) {
					cout << a << " ";
					}
				}
			if (MD.f_v) {
				cout << endl;
				}
			}
		}
	else {
		if (MD.f_v) {
			cout << "multiplication table:" << endl;
			}
		for (i = 0; i < q; i++) {
			for (j = 0; j < q; j++) {
				a = (i * j) % q;
				MD.ff_mult[i * q + j] = a;
				if (a == idx_one) {
					MD.ff_inv[i] = j;
					if (i == j) 
						MD.idx_mone = i;
					}
				if (MD.f_v) {
					cout << a << " ";
					}
				}
			if (MD.f_v) {
				cout << endl;
				}
			}
		if (MD.f_v) {
			cout << "addition table:" << endl;
			}
		for (i = 0; i < q; i++) {
			for (j = 0; j < q; j++) {
				a = (i + j) % q;
				MD.ff_add[i * q + j] = a;
				if (MD.f_v) {
					cout << a << " ";
					}
				}
			if (MD.f_v) {
				cout << endl;
				}
			}
		}
	if (MD.f_v) {
		cout << "the field: GF(" << MD.q << ") = GF(" << MD.p << "^" << MD.f << ")" << endl;
		cout << "idx_zero = " << MD.idx_zero << ", idx_one = " << MD.idx_one << ", idx_mone = " << MD.idx_mone << endl;
		}

	if (MD.idx_zero != 0) {
		cout << "at the moment, we assume that idx_zero == 0" << endl;
		exit(1);
		}
	
	MD.G = (int **) malloc((sizeof (int *))*(k+2));
	for (i = 1; i <= k; i++) {
		MD.G[i] = (int *)calloc(n+2,sizeof(int));
		}
	wt_rows = n;
	for (i = 0; i < k; i++) {
		w = 0;
		for (j = 0; j < n; j++) {
			if (G[i * n + j])
				w++;
			}
		wt_rows = MINIMUM(wt_rows, w);
		}

	for (i = 1; i <= k; i++) {
		for (j = 1; j <= n; j++) {
			a = G[(i-1)*n + j - 1];
			MD.G[i][j] = a;
			}
		}
	if (MD.f_v) {
		cout << "(" << n << "," << k << ") code over GF(" << q << "), generated by" << endl;
		print_matrix(&MD, MD.G);
		}

	MD.ZC = 0;
	d = min_weight(&MD);

	if (MD.f_v) {
		cout << "the minimum distance is " << d << endl;
		cout << "This was determined by looking at "
			<< MD.weight_computations
			<< " codewords\n(rather than "
			<< NT.i_power_j(q, k) << " codewords)" << endl;
		}
	
	if (d != wt_rows) {
		if (MD.f_v) {
			cout << "min weight = " << d << 
				" is less than the weight of the vectors in "
				"the rows, which is " << wt_rows << endl;
			print_matrix(&MD, MD.G);
			}
		//cin >> i;
		}
	
	for (i = 1; i <= k; i++) {
		free(MD.G[i]);
		}
	free(MD.G);
	free(MD.ff_mult);
	free(MD.ff_add);
	free(MD.ff_inv);
	return d;
}

static void print_matrix(MINDIST *MD, int **G)
{
	int i, j, a;

	for (i = 1; i <= MD->k; i++) {
		for (j = 1; j <= MD->n; j++) {
			a = G[i][j];
			cout << a << " ";
			}
		cout << endl;
		}
	
}


static int min_weight(MINDIST *MD)
// Calculate the minimum-weight of the code which is created by the generator matrix G.
// Main routine, loop with the two bounds (upper and lower)
// for the minimum distance.
{
	int n = MD->n;
	int k = MD->k;
	int i, j, t;
	int a, b;
	int w_c, w_t, w_r; /* minimum-weight of code /regarded /not regarded codevectors */
	int size = 0; /* number of information subsets */
 //int w_1 = -1;
	
	/* allocate base pointer for the (k,n)-generator matrices */
	MD->S = (int ***)malloc((sizeof (int **))*(n+2));
	create_systematic_generator_matrices(MD);

	/* evaluate minimum weight of code created by generator matrix G */
	for (i = 1; i <= MD->M; i++) {
		size = size + MD->Size[i];
	}
	w_c = n;
	w_r = 0;
	t = 0;
	while (w_c > w_r && t < k) {
		t = t + 1;
		
		/* evaluate minimumweight of codevectors created by linearcombination
		   of t rows of the systematic generator matrices S[1],...,S[M] */
		w_t = weight_of_linear_combinations(MD, t); 
#if 0
		if (t == 1) {
			w_1 = w_t;
			}
#endif
		if (MD->f_v) {
			cout << "\\bar{d}_" << t << "=" << w_t << endl;
			}
		w_c = MINIMUM(w_c,w_t);
		if (MD->f_v) {
			cout << "mindist(C_{\\le " << t << "})=" << w_c << endl;
			}
		
		if (MD->f_v) {
			cout << "\\bar{d}_" << t << "=";
			}
		w_r = 0;
		for (i = 1; i <= MD->M; i++) {
			a = k - MD->Size[i];
			b = t + 1 - a;
			if (b > 0) {
				w_r += b;
			}
			if (MD->f_v) {
				cout << " +" << t + 1 << "-(" << k << "-" << MD->Size[i] << ")" << endl;
				}
			}
		if (MD->f_v) {
			cout << "=" << w_r << endl;
			}

	} /* while */
#if 0
	if (w_c != w_1) {
		cout << "min weight = " << w_c << 
			" is less than the weight of the vectors in the rows, which is " << w_1 << endl;
		print_matrix(MD, MD->G);
		for (i = 1; i <= MD->M; i++) {
			cout << i << "-th matrix:" << endl;
			print_matrix(MD, MD->S[MD->M]);
			}
		}
#endif
	
	for (i = 1; i <= MD->M; i++){
		for (j = 1; j <= k; j++) {
			free(MD->S[i][j]);
		}
		free(MD->S[i]);
	}
	free(MD->S);
	free(MD->Size);
	return(w_c);
}


static void create_systematic_generator_matrices(MINDIST *MD)
//create systematic generator matrices
//$(S[1]=(I,*),...,S[z]=(*,I,*),...,S[M]=(*,I))$
//with identity-matrix $I$ beginning at $P+1 = k*(z-1)+1$
//(k = \# lines, m = \# systematic generator matrices),
//by elementary row-transformations and permutations in
//generator matrix G
//allocates S[u] for $1 \le u \le M$,
//allocates S[u][i] for $1 \le u \le M, 1 \le i \le k$,
//allocates Size
{
	int n = MD->n;
	int k = MD->k;
	int q = MD->q;
	
	int i, I = 0, j, J, l;
	int P, h, h1, h2, h3, pivot, pivot_inv;
	int K;
	int M;

	/* allocate memory for systematic generator matrix S[1] */
	MD->S[1] = (int **)calloc(k+2,sizeof(int *));
	for (i = 1; i <= k; i++) {
		MD->S[1][i] = (int *)calloc(n+2,sizeof(int));
	}

	/* S[1] :=  G */
	for (i = 1; i <= k; i++) {
		for (j = 1; j <= n; j++) {
			MD->S[1][i][j] = MD->G[i][j];
		}
	}
	/* allocate memory for size of information subsets of S[1] */
	MD->Size = (int *)calloc(n+1,sizeof(int ));

	M = 1;
	P = 0;
	K = k;

	while (1) {
		if (MD->f_vvv) {
			cout << "loop with M = " << M << ", P = " << P << " K = " << K << endl;
			}

		/*        create identity matrix, columns P+1,...,P+k          */

		for (i = 1; i <= K; i++) {
			if (MD->f_vvv) {
				cout << "i = " << i << " ";
				cout << " (M = " << M << ", P = " << P << " K = " << K << endl;
				}
			/* search for pivot: 
		    	        if the entry is 0 at (i,P+i),
				    first check the entries below
				    (-> row-permutation necessary)
				    then check the columns behind  
				    (-> column-permutation necessary)    */
			/* printf("i=%d, P=%d, S[M][i][P+i]=%d\n",i,P,S[M][i][P+i]);  */
			for (J = P + i; J <= n; J++) {
				for (I = i; I <= K; I++) {
					if (MD->S[M][I][J] != MD->idx_zero) {
						break;
					}
				}
				if (I <= K) {
					/* pivot found ? */
					break;
				}
			} // next J
			if (MD->f_vvv) {
				cout << "I=" << I << ", J=" << J << endl;
				}
			
			/*   if pivot found but end of columns reached: */
			if ((I <= K) && (J == n)) {
				if (MD->f_vvv) {
					cout << "end reached, I = " << I << endl;
				}
				if (P+K >= n) {
					K = n - P;
				}
			}
			if (MD->f_vvv) {
				cout << "pivot in I=" << I << " J=" << J << endl;
				}

			/*   if there doesn't exist a pivot: */
			/*   P(s,t)=0 for s>=i and t>=P+i */
			if (I > K && J > n) {
				K = i - 1;
				if (MD->f_vvv) {
					cout << "no pivot" << endl;
					}
				break;
			}

			/*   if necessary: row-permutation i<->I         */
			if (I != i) {
				if (MD->f_vvv) {
					cout << endl << "swapping rows: " << i << "<->" << I << endl;
					}
				for (j = 1; j <= n; j++) {
					h = MD->S[M][i][j];
					MD->S[M][i][j] = MD->S[M][I][j];
					MD->S[M][I][j] = h;
				}
			}

			/*    if necessary: column-permutation P+i<->J   */
			if (J != P + i) {
				if (MD->f_vvv) {
					cout << endl << "swapping columns: " << P + i << "<->" << J << endl;
					}
				for (j = 1; j <= k; j++) {
					h = MD->S[M][j][i+P];
					MD->S[M][j][i+P] = MD->S[M][j][J];
					MD->S[M][j][J] = h;
				}
			}
		
			pivot = MD->S[M][i][P + i];
			if (pivot == MD->idx_zero) {
				cout << "pivot is 0, exiting!" << endl;
				}
			pivot_inv = MD->ff_inv[pivot];
			if (MD->f_vvv) {
				cout << "pivot = " << pivot << ", pivot_inv = " << pivot_inv << endl;
				}
			/* replace pivot by 1 [multiply pivot-row by inv(pivot)] */
			if (MD->S[M][i][P+i] != MD->idx_one) {
				for (j = 1; j <= n; j++) {
					MD->S[M][i][j] = MD->ff_mult[MD->S[M][i][j] * q +  pivot_inv];
					}
				}
			if (MD->f_vvv) {
				cout << "pivot row normalized" << endl;
				}

			/* replace all elements of pivot-column, except pivot
				element, by 0 by elementary row-trans. */	 
			for (l = 1; l <= k; l++) {
				if (l != i) {
					if ((h = MD->S[M][l][i+P]) != MD->idx_zero)
					for (j = 1; j <= n; j++) {
						h1 = MD->ff_mult[MD->S[M][i][j] * q + h];
						h2 = MD->ff_mult[h1 * q + MD->idx_mone];
						h3 = MD->ff_add[MD->S[M][l][j] * q + h2];
						MD->S[M][l][j] = h3;


#if 0
						MD->S[M][l][j] = 
						mod_p(MD->S[M][l][j] - MD->S[M][i][j] * h);
#endif
					} // next j
				} /* if */
			} /* l */
			if (MD->f_vvv) {
				cout << "pivot col normalized" << endl;
				}

		} /* i */

		if (MD->f_v) {
			cout << endl << "systematic generator matrix s[" << M << "]:" << endl;
			print_matrix(MD, MD->S[M]);
			//printf("K = %d\n",K);
			}
		MD->Size[M] = K;

		if (K == 0) {
			if (MD->f_v) {
				cout << "K = 0, the generator matrix has " << n - P << " zero columns" << endl;
				}
			}
		if (P + K >= n || K == 0) {
			if (K == 0) {
				MD->ZC = n - P; /* number of zero columns in G */
			}
			break;
			}
		else {
			P = P + K;
			}

		M++;
		/*        find first column P+1      */ 
		MD->S[M] = (int **)calloc(k+2,sizeof(int *));

		for (i = 1; i <= k; i++) {
			MD->S[M][i] = (int *)calloc(n+2,sizeof(int));
		}

		/*   S[M] :=  S[M-1] */
		for (i = 1; i <= k; i++) {
			for (j = 1; j <= n; j++) {
				MD->S[M][i][j] = MD->S[M-1][i][j];
			}
		}
	} /* infinite loop */

	if (MD->f_v) {
		//printf("M = %d\n", M);
		//printf("ZC = %d\n", MD->ZC);
		cout << "size of information subsets:" << endl;
		for (i = 1; i <= M; i++) {
			cout << MD->Size[i] << " ";
			}
		cout << endl;
		}
	MD->M = M;

}

static int weight_of_linear_combinations(MINDIST *MD, int l)
//evaluate minimum-weight of all codevectors that are constructed
//by linearcombinations of $l$ rows of matrix $S[1],...,S[M]$
//algorithm: create all possibilities for combining $l$ rows out of $k$
//possible matrix rows by constructing the $l$-element-
//subsets over $\{1,...,k\}$;
//create all possibilities for filling an $l$-element-subset by
//$\{1,...,p\}$ while construct the p-adic numbers of $0,...,lc$
//($lc$ = number of possibilities for filling);
//combining the two results gives all possibilities of linear-
//combinations of $l$ matrix-rows;
//out of this construct the codevectors and
//calculate minimumweight
//$l$ = \# rows taken for linearcombination
// needed: weight(int *p, n, idx\_zero)
{
	int n = MD->n;
	int k = MD->k;
	int q = MD->q;
	int M = MD->M;
	int d1;
	int d_l;		     /* minimum-weight by combining l rows       */
	int lc, dec, h;
	int i, j, z, w;
	int *v,*linc,*lcv;
	int *sub;
	algebra::number_theory::number_theory_domain NT;
	
	/* for l = 1 the weight of a multiple of a generating codevector 
		    is equal to the weight of that codevector (p = prim) */

	/* allocate array for l-subset of a k-set */
	sub = (int *)calloc(k+1,sizeof (int));

	d_l = n;
	if (l == 1) {
		for (z = 1; z <= M; z++) {
			for (i = 1; i <= k; i++) {

				if (MD->Size[z] > 0) {
					d1 = weight(MD->S[z][i], n, MD->idx_zero);
					MD->weight_computations++;
					if (d1 > 0) {
						d_l = MINIMUM(d_l,d1);
						if (MD->f_vv) {
							cout << "matrix " << z << " row " << i << " is ";
							for (j = 1; j <= n; j++) {
								cout << MD->S[z][i][j] << " ";
							}
							cout << " of weight " << d1 << " minimum is " << d_l << endl;
						}
					}
				} // if
			} // next z
		} // next i
		free(sub);
		return(d_l);
	}
	
	/*          construct codevectors by linear combinations of l rows
	            of the generator matrices and calculate their weight */
	else {
		/* allocate memory for help-pointers */
		v = (int*)calloc(l+2,sizeof(int)); 
		linc = (int*)calloc(k+2,sizeof(int)); 
		lcv = (int*)calloc(n+2,sizeof(int));

		lc = NT.i_power_j(q-1, l);
#if 0
		lc = expo(p-1,l);   	/* number of possibilities to replace one 
				     	 * subset with entries 1,...,p-1 */
#endif

		/* If the l-subset is b_1,...,b_l, then form the
		   linear combination of the rows b_i times v[i] */
		for (dec = 1; dec <= lc; dec++) {
			padic(dec, v, l, q-1);

			/* for each systematic-generator-matrix S[z] */
			for (z = 1; z <= M; z++) {
				int K = MD->Size[z];
				if (K < l) {
					continue;
				}

				/* initialize with set: 1,2,...,l */
				for (i = 1; i <= l; i++) {
					sub[i] = i;
				}
				/* for (i=1;i<=l;i++) 
					printf("%d ",sub[i]); printf("\n");  */

				do {
					/* for (i=1;i<=l;i++) printf("%d ",sub[i]);  */
					h = 1;
					for (j = 1; j <= K; j++) {
						if (j == sub[h]) {
							linc[j] = v[h-1]+1;
							if (h != l) {
								h = h+1;
							}
						} /* if */
						else {
							linc[j] = 0;
						}
					} /* j */
					/* construct the corresponding codevector lcv */
					vmmult(MD, linc, MD->S[z], lcv);

					/* calculate its weight and store if minimal */
					w = weight(lcv, n, MD->idx_zero);

					MD->weight_computations++;

					d_l = MINIMUM(d_l,w);
					if (MD->f_vv) {
						for (i = 1; i <= k; i++) {
							cout << linc[i] << " ";
						}
						cout << " : ";
						for (i = 1; i <= n; i++) {
							cout << lcv[i] << " ";
						}
						cout << " of weight " << w << " minimum is " << d_l << endl;
					}
				} while (nextsub(K,l,sub));  /* next l-subset of K-set */
			} /* z */
		} /* dec */
		free(v);
		free(linc);
		free(lcv); 
		free(sub);
		return(d_l);
	} /* else */
	
}

static int weight(int *v, int n, int idx_zero)
/* calculate weight of vector v ( = number or non-zero elements of v ) */
//Note that in any case, 0 stands for the zero element of the field 
//(either $GF(p)$ or $GF(q)$).
{
	int i, w=0;
	
	for (i = 1; i <= n; i++) {
		if (v[i] != idx_zero) {
			w++;
		}
	}
	return(w);
}

static void padic(int ind, int *v, int L, int A)
/* convert ind of radix 10 to a L digit number of radix A, and store it at v */
{
	int i;
	int z = ind;
	for (i = 0; i < L; i++) {
		v[i] = z % A;
		z = (z - v[i]) / A;
	}
}


static int nextsub(int k, int l, int *sub) 
/* return 0 if lexicographically largest subset  reached, otherwise 1 */
{
	int a, i, j;

	if (sub[l] != k) {
		sub[l]++;
		return 1;
	}
	else {
		i = 1;
		while (i < l && sub[l - i] == k-i) {
			i++;
		}
		if (i < l) {
			a = sub[l - i];
			for (j = l - i; j <= l; j++) {
				sub[j] = a + 1 + j - (l - i);
			}
			return 1;
		}
		else {
			return 0;
		}
	}
}



static void vmmult(MINDIST *MD, int *v, int **mx, int *cv)
/* multiply vector v with matrix mx and store it at cv */
{
	int k = MD->k;
	int q = MD->q;
	int i, j;
	int h1, h2;
	
	for (j = 1; j <= MD->n; j++) {
		cv[j] = 0;
		for (i = 1; i <= k; i++) {
			h1 = MD->ff_mult[v[i] * q + mx[i][j]];
			h2 = MD->ff_add[cv[j] * q + h1];
			cv[j] = h2;
			// cv[j] = mod_p(cv[j] + v[i] * mx[i][j]);
			}
	}
}

}}}}


