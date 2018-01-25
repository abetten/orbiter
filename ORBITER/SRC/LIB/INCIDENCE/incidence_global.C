// incidence_global.C
// Anton Betten
//
// started: April 16, 2015

#include "galois.h"
#include "incidence.h"


INT diophant_solve_first_mckay(diophant *Dio, INT f_once, INT verbose_level)
{
	INT f_v = TRUE;//(verbose_level >= 1);
	INT j;
	INT maxresults = 10000000;
	vector<int> res;
	INT nb_backtrack_nodes;
	INT nb_sol;

	//verbose_level = 4;
	if (f_v) {
		cout << "diophant::solve_first_mckay calling solve_mckay" << endl;
		}
	if (f_once) {
		maxresults = 1;
		}
	diophant_solve_mckay(Dio, "", maxresults, nb_backtrack_nodes, nb_sol, verbose_level - 2);
	if (f_v) {
		cout << "diophant_solve_first_mckay found " << Dio->_resultanz << " solutions, using " << nb_backtrack_nodes << " backtrack nodes" << endl;
		}
	Dio->_cur_result = 0;
	if (Dio->_resultanz == 0)
		return FALSE;
	res = Dio->_results.front();
	for (j = 0; j < Dio->n; j++) {
		Dio->x[j] = res[j];
		}
	Dio->_results.pop_front();
	Dio->_cur_result++;
	if (f_v) {
		cout << "diophant_solve_first_mckay done" << endl;
		}
	return TRUE;
}

INT diophant_solve_all_mckay(diophant *Dio, INT &nb_backtrack_nodes, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT maxresults = 10000000;
	//INT nb_backtrack_nodes;
	INT nb_sol;
	
	diophant_solve_mckay(Dio, Dio->label, maxresults, nb_backtrack_nodes, nb_sol, verbose_level);
	if (f_v) {
		cout << "diophant_solve_all_mckay found " << Dio->_resultanz << " solutions in " << nb_backtrack_nodes << " backtrack nodes" << endl;
		}
	return Dio->_resultanz;
}

INT diophant_solve_once_mckay(diophant *Dio, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT maxresults = 1;
	INT nb_backtrack_nodes;
	INT nb_sol;

	diophant_solve_mckay(Dio, Dio->label, maxresults, nb_backtrack_nodes, nb_sol, verbose_level - 2);
	if (f_v) {
		cout << "diophant_solve_once_mckay found " << Dio->_resultanz << " solutions in " << nb_backtrack_nodes << " backtrack nodes" << endl;
		}
	return Dio->_resultanz;
}


INT diophant_solve_next_mckay(diophant *Dio, INT verbose_level)
{
	INT j;
	if (Dio->_cur_result < Dio->_resultanz) {
		for (j = 0; j < Dio->n; j++) {
			Dio->x[j] = Dio->_results.front()[j];
			}
		Dio->_results.pop_front();
		Dio->_cur_result++;
		return TRUE;
		}
	else {
		return FALSE;
		}
}


void diophant_solve_mckay(diophant *Dio, const BYTE *label, INT maxresults, INT &nb_backtrack_nodes, INT &nb_sol, INT verbose_level)
{
	diophant_solve_mckay_override_minrhs_in_inequalities(Dio, label, maxresults, nb_backtrack_nodes, 0 /* minrhs */, nb_sol, verbose_level);
}

void diophant_solve_mckay_override_minrhs_in_inequalities(diophant *Dio, const BYTE *label, 
	INT maxresults, INT &nb_backtrack_nodes, 
	INT minrhs, INT &nb_sol, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j, nb;
	vector<int> minres, maxres, fanz;
	mckay::tMCKAY lgs;
	vector<mckay::equation> eqn;
	map<int, int>::iterator it;
	vector<int> minvarvalue;
	vector<int> maxvarvalue;

	if (f_v) {
		cout << "diophant_solve_mckay_override_minrhs_in_inequalities " << label << ", a system of size " << Dio->m << " x " << Dio->n << endl;
		}
	lgs.Init(Dio, label, (int) Dio->m + 1, (int) Dio->n);
	minres.resize(Dio->m + 1);
	maxres.resize(Dio->m + 1);
	fanz.resize(Dio->m + 1);
	eqn.resize(Dio->m + 1);
	
	for (i = 0; i < Dio->m; i++) {
		// the RHS:
		if (Dio->type[i] == t_LE) {
			minres[i] = (int) minrhs;
			maxres[i] = (int) Dio->RHS[i];
			}
		else {
			minres[i] = (int) Dio->RHS[i];
			maxres[i] = (int) Dio->RHS[i];
			}
		
		// count the number of nonzero coefficients:
		nb = 0;
		for (j = 0; j < Dio->n; j++) {
			if (Dio->A[i * Dio->n + j])
				nb++;
			}
		
		// initialize coefficients:
		fanz[i] = (int) nb;
		eqn[i].resize(nb);
		nb = 0;
		for (j = 0; j < Dio->n; j++) {
			if (Dio->A[i * Dio->n + j]) {
				eqn[i][nb].var = (int) j;
				eqn[i][nb].coeff = (int) Dio->A[i * Dio->n + j];
				nb++;
				}
			}
		}
	
	// one more equation for \sum x_j = sum
	i = Dio->m;
	fanz[i] = (int) Dio->n;
	eqn[i].resize(Dio->n);
	for (j = 0; j < Dio->n; j++) {
		eqn[i][j].var = (int) j;
		eqn[i][j].coeff = 1;
		}
	minres[i] = (int) Dio->sum;
	maxres[i] = (int) Dio->sum;
	
	// now the bounds on the x_j
	minvarvalue.resize(Dio->n);
	maxvarvalue.resize(Dio->n);
	if (Dio->f_x_max) {
		for (j = 0; j < Dio->n; j++) {
			minvarvalue[j] = 0;
			maxvarvalue[j] = (int) Dio->x_max[j];
			}
		}
	else {
		for (j = 0; j < Dio->n; j++) {
			minvarvalue[j] = 0;
			maxvarvalue[j] = (int) Dio->sum;
			}
		}
#if 0
  for (j=1; j<=_eqnanz; j++) {
    minres[j-1] = _eqns[j-1].GetMinResult();
    maxres[j-1] = _eqns[j-1].GetMaxResult();
    fanz[j-1] = _eqns[j-1].GetVarAnz();
    eqn[j-1].resize(_eqns[j-1].GetVarAnz());
    it = _eqns[j-1].GetFaktoren().begin();
    for (i=1; i<=_eqns[j-1].GetVarAnz(); i++) {
      eqn[j-1][i-1].var=it->first-1;
      eqn[j-1][i-1].coeff=it->second;
      it++;
    }
  }
#endif
	Dio->_resultanz = 0;
	Dio->_maxresults = (int) maxresults;
	
	lgs.possolve(minvarvalue, maxvarvalue, eqn, minres, maxres, fanz, 
		(int) Dio->m + 1, (int) Dio->n, (int) verbose_level);
	nb_backtrack_nodes = lgs.nb_calls_to_solve;
	nb_sol = Dio->_resultanz;
	if (f_v) {
		cout << "diophant_solve_mckay_override_minrhs_in_inequalities " << label 
			<< " finished, number of solutions = " << Dio->_resultanz 
			<< " nb_backtrack_nodes=" << nb_backtrack_nodes << endl;
		}
}





