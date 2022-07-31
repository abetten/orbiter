// mckay.cpp
//
// solver due to Brendan McKay
// Commented by Alfred Wassermann
// C++ translation by Volker Widor 2003
// adapted by Anton Betten
// 1/16/2009

#include "foundations.h"

using namespace std;

namespace orbiter {
namespace layer1_foundations {
namespace solvers {



mckay::tMCKAY::tMCKAY() {
	nb_calls_to_solve = 0;
	first_moved = 0;
	second_moved = 0;
	problem_label = NULL;

	_eqnanz = 0;
	_varanz = 0;
	//vector<bool> unitcoeffs;
	//vector<bool> active;
	rekurs = 0;
	_break = false;

	D = NULL;
	//tLGS *_lgs;

#ifdef MCKAY_DEBUG
	//vector<int> range, split, branch;
	ticks0  = 0;
#endif
};

void mckay::tMCKAY::Init(diophant *lgs,
		const char *label, int aEqnAnz, int aVarAnz)
{
	orbiter_kernel_system::os_interface Os;

	_varanz=aVarAnz;
	_eqnanz=aEqnAnz;
	nb_calls_to_solve = 0;
	D = lgs;
	//_lgs = aLgs;
	rekurs=0;
	unitcoeffs.resize(_eqnanz);
	active.resize(_eqnanz);
	problem_label = label;
#ifdef MCKAY_DEBUG
	int m;

	m = MAXIMUM(_eqnanz, _varanz) + 1;
	range.resize(m);
	split.resize(m);
	branch.resize(m);
	ticks0 = Os.os_ticks();
#endif
}

void mckay::tMCKAY::possolve(vector<int> &lo, vector<int> &hi, 
	vector<equation> &eqn, vector<int> &lorhs, vector<int> &hirhs, 
	vector<int> &neqn, int numeqn, int numvar, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_v4 = (verbose_level >= 4);
        int i,j;
	bool hopeless;

	if (f_v) {
		cout << "possolve" << endl;
		}
	nb_calls_to_solve = 0;
	first_moved = INT_MAX;
	second_moved = INT_MAX;

	
        if (numeqn > _eqnanz || numvar > _varanz) {
            cerr<<"*** >E intsolve: limits exceeded\n";
            throw this;
        }
/* First step:
   If for one equation there is no coefficient different from
   zero, this equation is \"not" active.
   Further mark in \|unitcoeffs| the equations with coefficients
   solely $\in\{0,1\}$.
*/
	hopeless = false;
        for (i = 0; i < numeqn; ++i) {
	    if (neqn[i] == 0) {
		active[i] = false;
		if (lorhs[i] > 0 || hirhs[i] < 0) hopeless = true;
	    	}
	    else {
		active[i] = true;
		}
            for (j = neqn[i]; --j >= 0;) if (eqn[i][j].coeff != 1) break;
            unitcoeffs[i] = (j < 0);
        }

/* More output. */
	if (f_v4) {
        	puteqns(lo,hi,numvar,lorhs,hirhs,eqn,neqn,numeqn);
		}

/* Finally the recursion starts. */
	if (!hopeless)
		_break = false;
	
	if (f_v) {
		cout << "mckay::tMCKAY::possolve: calling solve" << endl;
		}
	solve(0,lo,hi,active,numvar,lorhs,hirhs,eqn,neqn,numeqn, verbose_level);
	if (f_v) {
		cout << "mckay::tMCKAY::possolve done" << endl;
		}
	
	return;
}

/* \subsection{subtract}
  subtract equation e1 from e2 if possible  ---
 return success.
 It is used in function \|pruneqn|.
*/
bool mckay::tMCKAY::subtract(
	int eqn1, equation &e1, int l1, int lors1, int hirs1, 
	int eqn2, equation &e2, int *pl2, int *plors2, int *phirs2, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
        int i,j,k;
        term e1i;
        int l2,factor,minfactor;

        /* First test if subtraction is possible. */
        minfactor = 999999;
        l2 = *pl2;
        if (l1 > l2 || hirs1 > lors1) return false;

        /* Second test if subtraction is possible. */
        j = 0;
        for (i = 0; i < l1; ++i) {
            e1i = e1[i];
            for (; j < l2 && e2[j].var != e1i.var; ++j) {}
            if (j == l2 || e2[j].coeff < e1i.coeff) return false;
            factor = e2[j].coeff / e1i.coeff;
            if (factor < minfactor) minfactor = factor;
        }

        /* Do subtraction */
        k = 0;
        for (i = j = 0; i < l1; ++i, ++j)
        {
            e1i = e1[i];
            for (; j < l2 && e2[j].var != e1i.var; ++j) e2[k++] = e2[j];
            if (j < l2 && e2[j].coeff > minfactor*e1i.coeff) {
                e2[k].var = e2[j].var;
                e2[k].coeff = e2[j].coeff - minfactor*e1i.coeff;
                ++k;
            }
        }
        for (; j < l2; ++j) e2[k++] = e2[j];

        *pl2 = k;
        *plors2 -= minfactor*lors1;
        *phirs2 -= minfactor*hirs1;

        /*  end of subtraction. */
	if (f_v) {
		cout << "subtract: subtracted equation " << eqn1
				<< " from equation " << eqn2 << endl;
		}
	return true;
}

/* \subsection{pruneqn --- remove equations}
 prune equations by subtraction.
*/
void mckay::tMCKAY::pruneqn(vector<int> &lo, vector<int> &hi,
		int numvar,
		vector<int> &lorhs, vector<int> &hirhs,
		vector<equation> &eqn, vector<int> &neqn,
		int numeqn, int verbose_level)
{
	int f_v = (verbose_level >= 1);
        bool ok;
        vector<bool> done;
        done.resize(_eqnanz);
        int i,j;

/* \|done| is always \|false|. Why? */

	for (i = 0; i < numeqn; ++i) {
		done[i] = false;
		}

	do {
		ok = true;
		for (i = 0; i < numeqn; ++i) {
			if (!done[i] && neqn[i] > 0) {
				for (j = 0; j < numeqn; ++j) {
					if (i != j &&
							subtract(i, eqn[i],neqn[i],
									lorhs[i],hirhs[i],j, eqn[j],
									&neqn[j],&lorhs[j],&hirhs[j],
									verbose_level)) {
						if (f_v) {
							cout << "mckay::tMCKAY::pruneqn after subtract:" << endl;
							puteqns(lo, hi,
									numvar, lorhs, hirhs, eqn, neqn, numeqn);
						}
						ok = false;
						done[j] = false;
						} // if
					} // for
				} // if
			} // for
		} while (!ok);
	return;
}

/*
  \subsection{varprune --- remove variables}
   Try to remove free variables by testing if variables are already
   fixed. This is the case if the lower bound on a variable is equal to its
   upper bound.
*/

void mckay::tMCKAY::varprune(vector<int> &lo, vector<int> &hi, 
	vector<int> &lorhs, vector<int> &hirhs, 
	vector<equation> &eqn, vector<int> &neqn, 
	int numeqn, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i,j,sum,len;

	for (j = 0; j < numeqn; ++j) {
		len = neqn[j];
		sum = 0;
		// simple test whether the lower bound 
		// of a variable meets its upper bound:
		for (i = 0; i < len;) {
			if (lo[eqn[j][i].var] == hi[eqn[j][i].var]) {
				sum += eqn[j][i].coeff*lo[eqn[j][i].var];
				if (f_v && sum) {
					cout << "varprune: equation " << j << " variable "
						<< eqn[j][i].var << " is not free any more, "
						"sum += " << eqn[j][i].coeff * lo[eqn[j][i].var]
						<< endl;
					}
				eqn[j][i] = eqn[j][--len];
				}
			else {
				++i;
				}
			}
		lorhs[j] -= sum;
		hirhs[j] -= sum;
		if (f_v && sum) {
			cout << "varprune: equation " << j << " reducing RHS by "
				<< sum << " to lorhs[j]=" << lorhs[j] << " hirhs[j]="
				<< hirhs[j] << endl;
			}
		neqn[j] = len;
		}
}


void mckay::tMCKAY::puteqns(vector<int> &lo, vector<int> &hi,
	int numvar, vector<int> &lorhs, vector<int> &hirhs,
	vector<equation> &eqn, vector<int> &neqn, int numeqn)
{
        int i,j;

        //  First the lower and upper bounds on the variable are written:
        cout << "CON" << endl;
        for (i = 0; i < numvar; ++i) {
            if (i > 0) cout << "," << endl;
            if (lo[i] > 0) cout << "x" << i << " >= " << lo[i] << ", ";
            cout << "x" << i << " <= " << hi[i];
        }

        // print the equations:
        for (i = 0; i < numeqn; ++i) {
            cout << "," << endl;
            cout << "Equation " << i << " : ";
            for (j = 0; j < neqn[i]; ++j) {
                if (j % 16 == 15) cout << endl << "  ";
                cout << (j == 0 ? "" : "+") << eqn[i][j].coeff
                		<< "*x" << eqn[i][j].var;
            }
            cout << (lorhs[i] < hirhs[i] ? " >= " : " = ") << lorhs[i];

            if (lorhs[i] == hirhs[i]) continue;

			// inequalities:
            cout << "," << endl;
            for (j = 0; j < neqn[i]; ++j) {
                if (j % 16 == 15) cout << endl << "  ";
                cout << (j == 0 ? "" : "+") << eqn[i][j].coeff
                		<< "*x" << eqn[i][j].var;
            }
            cout << " <= " << hirhs[i];
        }
        cout << ";" << endl;
}

/*
 \subsection{divideeqns}
 take out common factors, return bad eqn number.
 It is only used in the main program.
*/
int mckay::tMCKAY::divideeqns(vector<int> &lorhs,
		vector<int> &hirhs, vector<equation> &eqn,
		vector<int> &neqn, int numeqn)
{
    int i,j,g,len;

	for (j = 0; j < numeqn; ++j)
        {
            len = neqn[j];
	    if (len == 0) continue;

	    g = eqn[j][0].coeff;
	    i = 1;
	    for (i = 1; i < len && g > 1; ++i) g = gcd(g,eqn[j][i].coeff);
/* $g = \gcd$ of all coefficients of the left hand side of equation $i$.
   If $g=1$ step to the next equation.
*/
	    if (g == 1) continue;
	    for (i = 0; i < len; ++i) eqn[j][i].coeff /= g;

	    lorhs[j] = lorhs[j] < 0 ? 0 : (lorhs[j] + g - 1) / g;
	    hirhs[j] = hirhs[j] < 0 ? -1 : hirhs[j] / g;

/*  Write some information.*/
	    cerr<<"eqn "<<j<<": g="<<g<<" lorhs="<<lorhs[j]
			<<" hirhs="<<hirhs[j]<<"\n";

	    if (lorhs[j] > hirhs[j]) return j;
	}

	return -1;
}

/*
\subsection{gcd}
used in \|divideeqns|.
*/
int mckay::tMCKAY::gcd(int n1,int n2)
{
        int a,b,c;

        if (n1 > n2) {
            a = n1; b = n2;
        }
        else {
            a = n2; b = n1;
        }

        while ((c = a % b) > 0) {
            a = b; b = c;
        }

        return(b);
}

/*
\section{solve --- brute force recursion}
This procedure is called recursively.
*/
void mckay::tMCKAY::solve(int level, 
	vector<int> &alo, vector<int> &ahi, 
	vector<bool> &aactive, int numvar, 
	vector<int> &lorhs, vector<int> &hirhs, 
	vector<equation> &eqn, vector<int> &neqn, 
	int numeqn, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int f_vvv = (verbose_level >= 3);
    int i,j;
	vector<int> lo, hi;
	lo.resize(_varanz);
	hi.resize(_varanz);
	int isplit, mindiff;
	vector<bool> active;
	active.resize(_eqnanz);
	//int current_node;
	int f_restriction_made;

	//current_node = nb_calls_to_solve;
	nb_calls_to_solve++;
#if 0
	if (f_vv) {
		log_12l(nb_calls_to_solve, level);
		//log_12l(current_node, level);
		cout << "mckay::tMCKAY::solve " << problem_label
			//<< " node " << current_node
			<< " first_moved " << first_moved  
			<< " second_moved " << second_moved  
			<< " level " << level << endl;
		}
#ifdef MCKAY_DEBUG
	if (f_vvv) {
		cout << " : ";
		for (i = 0; i < level; ++i) {
			cout << "x_" << split[i] << "=" << branch[i];
			if (i < level - 1) {
				cout << ", ";
				}
			}
		}
#endif
	if (f_vv) {
		cout << endl;
		}
#endif
#ifdef MCKAY_DEBUG
	if (f_vvv) {
		cout << "level=" << level << endl;
		cout << "i : split[i] : range[i] : branch[i]" << endl;
		for (i = 0; i < level; ++i) {
			cout << i << " : " << split[i] << " : " << range[i]
				<< " : " << branch[i] << " : " << endl;
			}
		cout << "low and high:" << endl;
		for (j = 0; j < numvar; j++) {
			cout << j << " : " << alo[j] << " : " << ahi[j] << endl;
			}
		}
#endif

#if 0
	if (f_vvv) {
		log_12l(current_node, level);
		cout << " current system:" << endl;
		puteqns(alo,ahi,numvar,lorhs,hirhs,eqn,neqn,numeqn);
		}
#endif
	//f_debug = FALSE;

	//f_debug = (verbose_level >= 10);
	
	if (_break) {
		return;
		}


	if (FALSE) {
        	int nacc = 0;
        	cout << " SOLVE level "<<level<< endl;
	            cout<<"Number of active equations: ";
	            for (i = numeqn; --i >= 0;) if (aactive[i]) ++nacc;
        	    cout<<":"<<nacc<<endl;
		    for (i = 0; i < numeqn; i++) {
				if (aactive[i])
					cout << i << " ";
				}
			cout << endl;
		    cout << "low and high:" << endl;
		    for (j = 0; j < numvar; j++) {
		    	cout << j << " : " << alo[j] << " : " << ahi[j] << endl;
		    	}
#ifdef MCKAY_DEBUG
		    orbiter_kernel_system::os_interface Os;
	        if (((Os.os_ticks() - ticks0) / Os.os_ticks_per_second())
	        		> INTERVAL_IN_SECONDS) {
			ticks0 = Os.os_ticks();
			//f_debug = TRUE;
		       	puteqns(alo,ahi,numvar,lorhs,hirhs,eqn,neqn,numeqn);
			cout << "level=" << level << endl;
			cout << "range[] : branch[] :" << endl;
			for (i = 0; i < level; ++i)
				cout << range[i] << " : " << branch[i] << " : " << endl;
			}
#endif
		}




/*  \|lo|, \|hi| and \|active| are local arrays. */

	for (i = 0; i < numvar; ++i) {
		lo[i] = alo[i];
		hi[i] = ahi[i];
		}
	for (i = 0; i < numeqn; ++i) {
		active[i] = aactive[i];
		}


	if (!restrict_variables(level, lo, hi, active, numvar, 
		lorhs, hirhs, 
		eqn, neqn, 
		numeqn, f_restriction_made, verbose_level - 1)) {

#if 0
		if (f_vv) {
			log_12l(nb_calls_to_solve, level);
			//log_12l(current_node, level);
			cout << "solve restrict_variables returns FALSE, "
					"backtracking" << endl;
			}
#endif
		return;
		}



#if 0
	if (f_vvv && f_restriction_made) {
		log_12l(nb_calls_to_solve, level);
		//log_12l(current_node, level);
		cout << " after restriction: low and high:" << endl;
		for (i = 0; i < numvar; i++) {
			cout << i << " : " << lo[i] << " : " << hi[i] << endl;
			}
		}
#endif

	// Here comes the searching part. 
	// \|mindiff| gives the smallest
	// difference between lower and upper bound of a variable. 
	// \|isplit|
	// is the corresponding variable number.

#if 0
	if (f_vv) {
		log_12l(nb_calls_to_solve, level);
		//log_12l(current_node, level);
		cout << "searching part" << endl;
		}
#endif

	mindiff = INT_MAX;
	isplit = -1;
	for (i = 0; i < numvar; ++i) {
        if (hi[i] != lo[i] && hi[i] - lo[i] < mindiff) {
            isplit = i;
            mindiff = hi[i] - lo[i];
        	}
		}

	// If \|isplit| $< 0$ we have found a solution. 
	// Otherwise we try to delete variables by 
	// \|varprune| and we try to delete equations
	// by \|pruneqn|.

	if (isplit < 0) {
		if (f_v) {
			log_12l(nb_calls_to_solve, level);
			cout << " solution " << D->_resultanz << endl;
			}
		D->_results.push_back(lo);
		D->_resultanz++;
		if (D->_resultanz < D->_maxresults) {
			_break = false;
			}
		else {
			_break = true;
			}
        }
	else {
		if (level == 0) {
			varprune(lo,hi,lorhs,hirhs,eqn,neqn,numeqn, verbose_level);

//#if VERBOSE > 2
			//puteqns(lo,hi,numvar,lorhs,hirhs,eqn,neqn,numeqn);
//#endif


			pruneqn(lo,hi,numvar,lorhs,hirhs,eqn,neqn,numeqn, verbose_level);

//#if VERBOSE > 2

#if 0
			if (f_vvv) {
				log_12l(nb_calls_to_solve, level);
				//log_12l(current_node, level);
				cout << " after varprune and pruneqn:" << endl;
				puteqns(lo,hi,numvar,lorhs,hirhs,eqn,neqn,numeqn);
				}
#endif
//#endif
			}

/*
   Finally we start the recursion.
   the variable with the number \|isplit| runs from
   \|lo[isplit]| to \|hi[isplit]| and for each step we call \|solve|
   with this fixed value.

   \|branch|, \|split| and \|range| are collected for debugging purposes.
*/

	
#if 0
		if (TRUE /*(f_v && (current_node % 10000) == 0) || f_vv*/) {
			//log_12l(current_node, level);
			log_12l(nb_calls_to_solve, level);
			cout << "solve choosing variable " << isplit
				<< " lo=" << lo[isplit] << " hi=" << hi[isplit] << endl;
			}
#endif
		for (i = 0; i <= mindiff; ++i) {
			int first_moved_orig = first_moved;
			int second_moved_orig = second_moved;
			int f_first_moved_has_changed = FALSE;
			int f_second_moved_has_changed = FALSE;


			if (i) {

				if (level < first_moved) {
					first_moved = level;
					second_moved = INT_MAX;
					}
				else if (level < second_moved) {
					second_moved = level;
					}
				if (first_moved != first_moved_orig) {
					f_first_moved_has_changed = TRUE;
				}
				if (second_moved != second_moved_orig) {
					f_second_moved_has_changed = TRUE;
				}
				}
			if (f_first_moved_has_changed || f_second_moved_has_changed ||
					(nb_calls_to_solve % 10000000) == 0) {
					log_12l(nb_calls_to_solve, level);
					cout << " x_" << isplit << " = " << lo[isplit] << endl;
				}
			hi[isplit] = lo[isplit];
#ifdef MCKAY_DEBUG
			split[level] = isplit;
			branch[level] = lo[isplit];
			range[level] = mindiff+1;
#endif

			// here comes the recursion:
			
			if (f_v) {
				cout << "solve level " << level << " isplit=" << isplit << " before solve i=" << i << " / " << mindiff << endl;
			    for (j = 0; j < numeqn; j++) {
					if (aactive[i])
						cout << j << " ";
					}
				cout << endl;
			    cout << "low and high:" << endl;
			    for (j = 0; j <= level; j++) {
			    	cout << j << " : " << alo[j] << " : " << ahi[j] << endl;
			    	}
#if 0
#ifdef MCKAY_DEBUG
		        if (TRUE/*((os_ticks() - ticks0) / os_ticks_per_second())
		        		> INTERVAL_IN_SECONDS*/) {
					ticks0 = os_ticks();
					//f_debug = TRUE;
					puteqns(alo,ahi,numvar,lorhs,hirhs,eqn,neqn,numeqn);
					cout << "level=" << level << endl;
					cout << "range[] : branch[] :" << endl;
					for (j = 0; j <= level; j++) {
						cout << range[j] << " : " << branch[j] << " : " << endl;
					}
				}
#else
		       // cout << "MCKAY_DEBUG is FALSE" << endl;
#endif
#endif
			}
			solve(level + 1, lo, hi, active, numvar,
					lorhs, hirhs, eqn, neqn, numeqn,
					verbose_level);
			
			if (f_v) {
				cout << "solve level " << level << " isplit=" << isplit << " after solve" << endl;
			}
			++lo[isplit];
			} // next i
        } // else

#if 0
	if (f_vv) {
			log_12l(nb_calls_to_solve, level);
			//log_12l(current_node, level);
			cout << " done" << endl;
		}
#endif
}

int mckay::tMCKAY::restrict_variables(int level, 
	vector<int> &lo, vector<int> &hi, 
	vector<bool> &active, int numvar, 
	vector<int> &lorhs, vector<int> &hirhs, 
	vector<equation> &eqn, vector<int> &neqn, 
	int numeqn, int &f_restriction_made, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	//int f_debug;
    int i, j;
	long int current_node;
    int losum,hisum,eic,eiv,lx,hx;
	int nfree,ok, xlo,xhi;
	int save;
	int f_restriction_made_in_this_eqn;

	if (f_v) {
		cout << "mckay::tMCKAY::restrict_variables" << endl;
	}
	if (f_vv) {
	    cout << "low and high:" << endl;
	    for (j = 0; j <= level; j++) {
	    	cout << j << " : " << lo[j] << " : " << hi[j] << endl;
	    	}
		puteqns(lo, hi, numvar, lorhs, hirhs, eqn, neqn, numeqn);
		cout << "level=" << level << endl;
		cout << "range[] : branch[] :" << endl;
		for (j = 0; j <= level; j++) {
			cout << range[j] << " : " << branch[j] << " : " << endl;
		}
	}
	current_node = nb_calls_to_solve;
/* The following line seems to be a relict from another problem. */
/* *<        if (level == 8 && lo[22] == 1 && lo[19] == 1) nul(); >* */

/* The following loop through the equations tries to restrict the lower
   and upper bounds on the variables.
   We only have to handle active equations.
*/
        ok = 0;
	/* ok = number of equations that have been
	 * checked since the last change of any
	 * lo[] or hi[] was made.
	 * the aim is to check all equations once
	 * without reducing hi[] - lo[];
	 * so that we have reached stability */
	 
	f_restriction_made = FALSE;
	for (j = 0; ok < numeqn; j = (j == numeqn-1 ? 0 : j+1)) {
		// j is the next equation to check;
		// j wraps around all equation indices 
		if (f_vv /*f_debug*/) {
			cout << "mckay::tMCKAY::restrict_variables checking equation " << j << endl;
			}
		++ok;
		if (active[j]) {

			f_restriction_made_in_this_eqn = FALSE;
			// we check equation j: 

			// We distinguish two cases: 
			// First, if there are only $\{0,1\}$-coefficients. 
			

			if (FALSE /*unitcoeffs[j]*/) {
				// The lower and upper bounds on the variables 
				// (belonging to nonzero coefficients) are summed up 
				// in \|losum| and \|hisum|.

				if (f_vv /*f_debug*/) {
					cout << "checking equation " << j
							<< " with unitcoeffs" << endl;
					}
				losum = 0;
				hisum = 0;
				for (i = neqn[j]; --i >= 0;) {
					losum += lo[eqn[j][i].var];
						// lowest possible lhs 
					hisum += hi[eqn[j][i].var];
						// highest possible lhs 
					}
				if (losum > hirhs[j]) {
					if (f_v) {
						cout << "solve " << problem_label << " node "
							<< current_node << " level " << level
							<< "return b/c for equation " << j
							<< " with unitcoeffs "
							"we have losum = " << losum << " > "
							<< hirhs[j] << " = hirhs[j]" << endl;
						}
					return FALSE;
					}
				if (hisum < lorhs[j]) {
					if (f_v) {
						cout << "solve " << problem_label << " node "
							<< current_node << " level " << level
							<< "return b/c for equation " << j
							<< " with unitcoeffs "
							" we have hisum = " << hisum << " < "
							<< lorhs[j] << " = lorhs[j]" << endl;
						}
					return FALSE;
					}

				// If possible the lower or upper bounds on the
				// variables are restricted further.
				// count the number of remaining free
				// variables in nfree.

				nfree = 0;
				for (i = neqn[j]; --i >= 0;) {
					eiv = eqn[j][i].var;
					hx = hi[eiv];
					lx = lo[eiv];
					if (hx != lx) {
						
						xlo = lorhs[j] + hx - hisum;
							// = lorhs[j] - (hisum - hx), i.e., 
							// lorhs[j] minus hisum of all the
							// other variables.
							// This is a lower bound on the amount 
							// that the current variable contributes.
							
						xhi = hirhs[j] + lx - losum;
							// = hirhs[j] - (losum - lx), i.e. 
							// hirhs[j] minus losum of all the
							// other variables.
							// This is an upper bound on the amount 
							// that the current variable contributes.
							
						if (xlo > lx) {
							save = lo[eiv];
							lo[eiv] = xlo;
							if (lo[eiv] != save) {
								f_restriction_made = TRUE;
								f_restriction_made_in_this_eqn = TRUE;
								if (f_v) {
									cout << "increasing lo[" << eiv
										<< "] from " << save << " to "
										<< lo[eiv] << endl;
									}
								}
							ok = 0;
							// a change was made;
							// loop through all
							// equations again.
							}
						if (xhi < hx) {
							save = hi[eiv];
							hi[eiv] = xhi;
							if (lo[eiv] != save) {
								f_restriction_made = TRUE;
								f_restriction_made_in_this_eqn = TRUE;
								if (f_v) {
									cout << "reducing hi[" << eiv
										<< "] from " << save << " to "
										<< hi[eiv] << endl;
									}
								}
							ok = 0;
							}
						if (lo[eiv] != hi[eiv]) {
							++nfree;
							}
						} // if (hx != lx)
					} // next i 
				} // if (unitcoeffs[j]) 
				
			// Now the slightly more complicated case 
			// if there are coefficents greater than $1$.
			// If the lower bound of a variable becomes greater 
			// than its upper bound, the procedure is stopped at once.
			else {
				// Again the lower and upper bounds on the variables 
				// (belonging to nonzero coefficients) are summed
				// up in \|losum| and \|hisum|.
				if (f_v /*f_debug*/) {
					cout << "mckay::tMCKAY::restrict_variables checking equation " << j
						<< " without unitcoeffs" << endl;
					}
				losum = 0;
				hisum = 0;
				for (i = neqn[j]; --i >= 0;) {
					losum += eqn[j][i].coeff * lo[eqn[j][i].var];
					hisum += eqn[j][i].coeff * hi[eqn[j][i].var];
					}
				if (losum > hirhs[j]) {
					if (f_v) {
						cout << "mckay::tMCKAY::restrict_variables "
								<< problem_label << " node "
								<< current_node << " level " << level
								<< "return b/c for equation " << j
								<< " without unitcoeffs "
								"we have losum = " << losum << " > "
								<< hirhs[j] << " = hirhs[j]" << endl;
						}
					return FALSE;
					}
				if (hisum < lorhs[j]) {
					if (f_v) {
						cout << "mckay::tMCKAY::restrict_variables "
							<< problem_label << " node "
							<< current_node << " level " << level
							<< "return b/c for equation " << j
							<< " without unitcoeffs "
							"we have hisum = " << hisum << " < "
							<< lorhs[j] << " = lorhs[j]" << endl;
						}
					return FALSE;
					}

				if (f_vv && f_restriction_made_in_this_eqn) {
					cout << "mckay::tMCKAY::restrict_variables "
							"equation " << j << ", after restriction: "
							"low and high:" << endl;
					for (i = 0; i < numvar; i++) {
						cout << i << " : " << lo[i] << " : "
								<< hi[i] << endl;
						}
					cout << "hisum=" << hisum << endl;
					cout << "losum=" << losum << endl;
					}
				// And if possible the lower or upper bounds
				// on the variables
				// are restricted further.

				f_restriction_made_in_this_eqn = FALSE;
				
				nfree = 0;
				for (i = neqn[j]; --i >= 0;) {
					if (FALSE /*f_debug*/) {
						cout << "restricting lower and upper "
								"bounds equation " << j << endl;
						}
					if (hi[eqn[j][i].var] != lo[eqn[j][i].var]) {
						eic = eqn[j][i].coeff;
						eiv = eqn[j][i].var;
						hx = eic * hi[eiv];
						lx = eic * lo[eiv];
						xlo = lorhs[j] + hx - hisum;
							// = lorhs[j] - (hisum - hx), i.e., 
							// lorhs[j] minus hisum of all the other variables.
							// This is a lower bound on the amount 
							// that the current variable contributes.
							
						xhi = hirhs[j] + lx - losum;
							// = hirhs[j] - (losum - lx), i.e. 
							// hirhs[j] minus losum of all the other variables.
							// This is an upper bound on the amount 
							// that the current variable contributes.

						if (xlo > lx) {
							save = lo[eiv];
							lo[eiv] = (xlo + eic - 1) / eic;
							if (lo[eiv] != save) {
								f_restriction_made = TRUE;
								f_restriction_made_in_this_eqn = TRUE;
								if (f_v) {
									cout << "increasing lo[" << eiv
										<< "] from "
										<< save << " to " << lo[eiv] << " : "
										<< "eqn j=" << j << " variable "
										<< eiv << " coeff=" << eic
										<< " lorhs[j]=" << lorhs[j]
										<< " hisum=" << hisum
										<< " hx=" << hx
										<< " hisum - hx=" << hisum - hx 
										<< endl;
									}
								}
							if (lo[eiv] > hi[eiv]) {
								if (f_v) {
									cout << "return bc for eqn " << j
										<< " term " << i << " xlo > lx "
										"and lo[eiv] > hi[eiv]" << endl;
									}
								return FALSE;
								}
							ok = 0;
							}
						if (xhi < hx) {
							save = hi[eiv];
							hi[eiv] = xhi / eic;
							if (hi[eiv] < save) {
								f_restriction_made = TRUE;
								f_restriction_made_in_this_eqn = TRUE;
								if (f_v) {
									cout << "reducing hi[" << eiv << "] from " 
										<< save << " to " << hi[eiv] << " : "
										<< "eqn j=" << j << " variable "
										<< eiv << " coeff=" << eic
										<< " hirhs[j]=" << hirhs[j]
										<< " losum=" << losum
										<< " lx=" << lx
										<< " losum - lx=" << losum - lx 
										<< endl;
									}
								}
							if (lo[eiv] > hi[eiv]) {
								if (f_v) {
									cout << "return bc for eqn " << j
											<< " term " << i << " xhi < hx "
											"and lo[eiv] > hi[eiv]" << endl;
									cout << "xlo=" << xlo << endl;
									cout << "xhi=" << xhi << endl;
									cout << "lx=" << lx << endl;
									cout << "hx=" << hx << endl;
									cout << "eic=" << eic << endl;
									cout << "eiv=" << eiv << endl;
									cout << "lo[eiv]=" << lo[eiv] << endl;
									cout << "hi[eiv]=" << hi[eiv] << endl;
									}
								return FALSE;
								}
							ok = 0;
							}
						if (lo[eiv] != hi[eiv]) {
							++nfree;
							}
						} // if hi[eqn[j][i]
					} // next i 

				if (f_vv && f_restriction_made_in_this_eqn) {
					cout << "mckay::tMCKAY::restrict_variables "
							"equation " << j << ", after restriction: "
							"low and high:" << endl;
					for (i = 0; i < numvar; i++) {
						cout << i << " : " << lo[i] << " : "
								<< hi[i] << endl;
						}
					cout << "hisum=" << hisum << endl;
					cout << "losum=" << losum << endl;
					}


				} // else 
				
				
			// Now hopefully the variables are in each
			// case further restricted.
			// The equation becomes inactive if
			// \item{(1)} there are no free variables in
			// this equation \"and
			// \item{(2)} if it was not possible to further 
			//            restrict the variables in the last try.
			
			if (ok > 0 && nfree == 0) {
				active[j] = false;
				}
            } // if (active[j])
        } // next j

	return TRUE;
}

void mckay::tMCKAY::log_12l(long int current_node, int level)
{
		cout << "solve " << problem_label 
			<< " node " << current_node 
			<< " first_moved ";
		
		if (first_moved < INT_MAX) {
			cout << first_moved;
			}
		else {
			cout << "infinity";
			}

		cout << " second_moved ";
		if (second_moved < INT_MAX) {
			cout << second_moved;
			}
		else {
			cout << "infinity";
			}
		
		cout << " level " << level;
		cout << " nb_sol=" << D->_resultanz;
}


}}}

