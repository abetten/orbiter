// nauty_interface.C
// 
// Anton Betten
// Abdullah Al-Azemi
//
// 2007
//
//
// Interface to Brendan McKay's nauty.
// Note that we have made slight changes 
// to the nauty code itself.
// Search for 'Abdullah' in nautil.c and nauty.c
//

#include <iostream>
#include <stdio.h>
#include <stdlib.h>


using namespace std;



#define HAS_NAUTY 1


#if HAS_NAUTY




#define MAXN 100000
#include "nauty.h"

#define MAX_WORKSPACE 50000


namespace orbiter {
namespace foundations {

static graph g[MAXN * MAXM];
static graph canong[MAXN * MAXM];
static nvector lab[MAXN], ptn[MAXN], orbits[MAXN];
//static setword workspace[MAX_WORKSPACE * MAXM];

using namespace std;

typedef unsigned char uchar;

int bitvector_s_i(uchar *bitvec, int i);
int ij2k(int i, int j, int n);

static void nauty_interface_allocate_data(int n);
static void nauty_interface_free_data();
#else
#endif


typedef unsigned char uchar;


void nauty_interface_graph_bitvec(int v, uchar *bitvector_adjacency, 
	int *labeling, int *partition, 
	int *Aut, int &Aut_counter, 
	int *Base, int &Base_length, 
	int *Transversal_length, int &Ago, int verbose_level)
{
#if HAS_NAUTY
	int f_v = (verbose_level >= 1);
	static DEFAULTOPTIONS(options);
	statsblk(stats);
	set *row;
	int m, n, i, j, k;

	if (f_v) {
		cout << "nauty_interface_graph_bitvec" << endl;
		}
	options.getcanon = TRUE;
	options.defaultptn = FALSE;
	//	options.writeautoms = TRUE;
	// 		options.cartesian = TRUE;
	// 		options.writemarkers = TRUE;

	n = v;
	aut_counter = 0;
	nauty_interface_allocate_data(n);
	base_length = 0;

	m = (n + WORDSIZE - 1) / WORDSIZE;
	if (n >= MAXN) {
		cout << "nauty_interface_graph_bitvec n >= MAXN" << endl;
		exit(1);
		}
	//cout << "nauty_interface_graph_bitvec n = " << n << " m=" << m << endl;
	for (i = 0; i < n; i++) {
		row = GRAPHROW(g, i, m);
		EMPTYSET(row, m);
		}
	
	for (i = 0; i < v; i++) {
		for (j = i + 1; j < v; j++) {
			k = ij2k(i, j, v);
			if (bitvector_s_i(bitvector_adjacency, k)) {
				row = GRAPHROW(g, i, m);
				ADDELEMENT(row, j);
				row = GRAPHROW(g, j, m);
				ADDELEMENT(row, i);
				}
			}
		}

	for (i = 0; i < n; i++) {
		lab[i] = i;
		ptn[i] = partition[i];
		}
	//ptn[v - 1] = 0;
	//cout << "calling nauty..." << endl;
	if (f_v) {
		cout << "nauty_interface_graph_bitvec calling nauty" << endl;
	}
	//	nauty(g, lab, ptn, NILSET, orbits, &options,
	//&stats, workspace, MAX_WORKSPACE * MAXM, m, n, canong);
	densenauty(g, lab, ptn, orbits, &options, &stats, m, n, canong);
	if (f_v) {
		cout << "nauty_interface_graph_bitvec after nauty" << endl;
		cout << "base_length=" << base_length << endl;
		cout << "transversal_length=";
		for (i = 0; i < base_length; i++) {
			cout << transversal_length[i];
			if (i < base_length - 1) {
				cout << ", ";
				}
			}
		cout << endl;
		}
	//cout << "numnodes=" << stats.numnodes << endl;
	for (i = 0; i < n; i++) {
		labeling[i] = lab[i];
		}
#if 1
	Ago = ago;
	Base_length = base_length;
	for (i = base_length - 1; i >= 0; i--) {
		Base[base_length - 1 - i] = base[i];
		Transversal_length[base_length - 1 - i] = transversal_length[i];
		}

	for (i = 0; i < aut_counter; i++) {
		for (j = 0; j < n; j++) {
			Aut[i * n + j] = aut[i * n + j];
			}
		}
	Aut_counter = aut_counter;
#endif
	nauty_interface_free_data();
	if (f_v) {
		cout << "nauty_interface_graph_bitvec done" << endl;
		}
#endif
}

void nauty_interface_graph_int(int v, int *Adj, 
	int *labeling, int *partition, 
	int *Aut, int &Aut_counter, 
	int *Base, int &Base_length, 
	int *Transversal_length, int &Ago, int verbose_level)
{
#if HAS_NAUTY
	int f_v = (verbose_level >= 1);
	static DEFAULTOPTIONS(options);
	statsblk(stats);
	set *row;
	int m, n, i, j;

	if (f_v) {
		cout << "nauty_interface_graph_int" << endl;
		}
	options.getcanon = TRUE;
	options.defaultptn = FALSE;
//	options.writeautoms = TRUE;
// 		options.cartesian = TRUE;
// 		options.writemarkers = TRUE;

	n = v;
	aut_counter = 0;
	nauty_interface_allocate_data(n);
	base_length = 0;

	m = (n + WORDSIZE - 1) / WORDSIZE;
	if (n >= MAXN) {
		cout << "nauty_interface_graph_int n >= MAXN" << endl;
		exit(1);
		}
	//cout << "nauty_interface_graph_int n = " << n << " m=" << m << endl;
	for (i = 0; i < n; i++) {
		row = GRAPHROW(g, i, m);
		EMPTYSET(row, m);
		}
	
	for (i = 0; i < v; i++) {
		for (j = i + 1; j < v; j++) {
			if (Adj[i * v + j]) {
				row = GRAPHROW(g, i, m);
				ADDELEMENT(row, j);
				row = GRAPHROW(g, j, m);
				ADDELEMENT(row, i);
				}
			}
		}

	for (i = 0; i < n; i++) {
		lab[i] = i;
		ptn[i] = partition[i];
		}
	//ptn[v - 1] = 0;
	//cout << "calling nauty..." << endl;
	if (f_v) {
		cout << "nauty_interface_graph_int calling nauty" << endl;
	}
//	nauty(g, lab, ptn, NILSET, orbits, &options, &stats, workspace, MAX_WORKSPACE * MAXM, m, n, canong);
	densenauty(g, lab, ptn, orbits, &options, &stats, m, n, canong);
	if (f_v) {
		cout << "nauty_interface_graph_int after nauty" << endl;
		}
	//cout << "numnodes=" << stats.numnodes << endl;
	for (i = 0; i < n; i++) {
		labeling[i] = lab[i];
		}
#if 1
	Ago = ago;
	Base_length = base_length;
	for (i = base_length - 1; i >= 0; i--) {
		Base[base_length - 1 - i] = base[i];
		Transversal_length[base_length - 1 - i] = transversal_length[i];
		}

	for (i = 0; i < aut_counter; i++) {
		for (j = 0; j < n; j++) {
			Aut[i * n + j] = aut[i * n + j];
			}
		}
	Aut_counter = aut_counter;
#endif
	nauty_interface_free_data();
	if (f_v) {
		cout << "nauty_interface_graph_int done" << endl;
		}
#endif
}


void nauty_interface_int(int v, int b, int *X, int nb_inc, 
	int *labeling, int *partition, 
	int *Aut, int &Aut_counter, 
	int *Base, int &Base_length, 
	int *Transversal_length, int &Ago)
{
#if HAS_NAUTY
	static DEFAULTOPTIONS(options);
	statsblk(stats);
	set *row;
	int m, n, i, j, p1, p2, u, x;

	options.getcanon = TRUE;
	options.defaultptn = FALSE;
//	options.writeautoms = TRUE;
// 		options.cartesian = TRUE;
// 		options.writemarkers = TRUE;

	n = v + b;
	aut_counter = 0;
	nauty_interface_allocate_data(n);
	base_length = 0;

	m = (n + WORDSIZE - 1) / WORDSIZE;
	if (n >= MAXN) {
		cout << "nauty_interface_int n >= MAXN" << endl;
		cout << "nauty_interface_int n = " << n << endl;
		cout << "nauty_interface_int MAXN = " << (int)MAXN << endl;
		exit(1);
		}
	//cout << "nauty_interface_int() n = " << n << " m=" << m << endl;
	for (i = 0; i < n; i++) {
		row = GRAPHROW(g, i, m);
		EMPTYSET(row, m);
		}
	
	for (u = 0; u < nb_inc; u++) {
		x = X[u];
		i = x / b;
		j = x % b;
		p1 = i;
		p2 = v + j;
		row = GRAPHROW(g, p1, m);
		ADDELEMENT(row, p2);
		row = GRAPHROW(g, p2, m);
		ADDELEMENT(row, p1);
		}
#if 0
	for (i = 0; i < v; i++) {
		for (j = 0; j < b; j++) {
			if (M[i * b + j] == 0)
				continue;
			p1 = i;
			p2 = v + j;
			row = GRAPHROW(g, p1, m);
			ADDELEMENT(row, p2);
			row = GRAPHROW(g, p2, m);
			ADDELEMENT(row, p1);
			}
		}
#endif

	for (i = 0; i < n; i++) {
		lab[i] = i;
		ptn[i] = partition[i];
		}
	//ptn[v - 1] = 0;
	//cout << "calling nauty..." << endl;
//	nauty(g, lab, ptn, NILSET, orbits, &options, &stats, workspace, MAX_WORKSPACE * MAXM, m, n, canong);
	densenauty(g, lab, ptn, orbits, &options, &stats, m, n, canong);
	//cout << "numnodes=" << stats.numnodes << endl;
	for (i = 0; i < n; i++) {
		labeling[i] = lab[i];
		}
#if 1
	Ago = ago;
	Base_length = base_length;
	for (i = base_length - 1; i >= 0; i--) {
		Base[base_length - 1 - i] = base[i];
		Transversal_length[base_length - 1 - i] = transversal_length[i];
		}

	for (i = 0; i < aut_counter; i++) {
		for (j = 0; j < n; j++) {
			Aut[i * n + j] = aut[i * n + j];
			}
		}
	Aut_counter = aut_counter;
#endif
	nauty_interface_free_data();
#endif
}

void nauty_interface_low_level(int v, int b, int *X, int nb_inc,
	int *labeling, int *partition, 
	int *Aut, int &Aut_counter, 
	int *Base, int &Base_length, 
	int *Transversal_length, int &Ago)
{
#if HAS_NAUTY
	static DEFAULTOPTIONS(options);
	statsblk(stats);
	set *row;
	int m, n, i, j, p1, p2, u, x;

	options.getcanon = TRUE;
	options.defaultptn = FALSE;
//	options.writeautoms = TRUE;
// 		options.cartesian = TRUE;
// 		options.writemarkers = TRUE;

	n = v + b;
	aut_counter = 0;
	nauty_interface_allocate_data(n);
	base_length = 0;

	m = (n + WORDSIZE - 1) / WORDSIZE;
	if (n >= MAXN) {
		cout << "nauty_interface n >= MAXN" << endl;
		cout << "nauty_interface n = " << n << endl;
		cout << "nauty_interface MAXN = " << (int)MAXN << endl;
		exit(1);
		}
	//cout << "nauty_interface_low_level n = " << n << " m=" << m << endl;
	for (i = 0; i < n; i++) {
		row = GRAPHROW(g, i, m);
		EMPTYSET(row, m);
		}
	
	for (u = 0; u < nb_inc; u++) {
		x = X[u];
		i = x / b;
		j = x % b;
		p1 = i;
		p2 = v + j;
		row = GRAPHROW(g, p1, m);
		ADDELEMENT(row, p2);
		row = GRAPHROW(g, p2, m);
		ADDELEMENT(row, p1);
		}
#if 0
	for (i = 0; i < v; i++) {
		for (j = 0; j < b; j++) {
			if (M[i * b + j] == 0)
				continue;
			p1 = i;
			p2 = v + j;
			row = GRAPHROW(g, p1, m);
			ADDELEMENT(row, p2);
			row = GRAPHROW(g, p2, m);
			ADDELEMENT(row, p1);
			}
		}
#endif

	for (i = 0; i < n; i++) {
		lab[i] = i;
		ptn[i] = partition[i];
		}
	//ptn[v - 1] = 0;
	//cout << "calling nauty..." << endl;
	//	nauty(g, lab, ptn, NILSET, orbits,
	//&options, &stats, workspace, MAX_WORKSPACE * MAXM, m, n, canong);
	densenauty(g, lab, ptn, orbits, &options, &stats, m, n, canong);
	//cout << "numnodes=" << stats.numnodes << endl;
	for (i = 0; i < n; i++) {
		labeling[i] = lab[i];
	}
#if 1
	Ago = ago;
	Base_length = base_length;
	for (i = base_length - 1; i >= 0; i--) {
		Base[base_length - 1 - i] = base[i];
		Transversal_length[base_length - 1 - i] = transversal_length[i];
	}

	for (i = 0; i < aut_counter; i++) {
		for (j = 0; j < n; j++) {
			Aut[i * n + j] = aut[i * n + j];
		}
	}
	Aut_counter = aut_counter;
#endif
	nauty_interface_free_data();
#endif
}

void nauty_interface_matrix(int *M, int v, int b,
	int *labeling, int *partition, 
	int *Aut, int &Aut_counter, 
	int *Base, int &Base_length, 
	int *Transversal_length, int &Ago)
// called from INC_GEN/inc_gen_iso.C
{
#if HAS_NAUTY
	static DEFAULTOPTIONS(options);
	statsblk(stats);
	set *row;
	int m, n, i, j, p1, p2;

	options.getcanon = TRUE;
	options.defaultptn = FALSE;
//	options.writeautoms = TRUE;
// 		options.cartesian = TRUE;
// 		options.writemarkers = TRUE;

	n = v + b;
	aut_counter = 0;

	// global variables in nauty.c:
	nauty_interface_allocate_data(n);


	base_length = 0;

	m = (n + WORDSIZE - 1) / WORDSIZE;
	if (n >= MAXN) {
		cout << "nauty_interface_matrix n >= MAXN" << endl;
		exit(1);
		}
	//cout << "nauty_interface_matrix n = " << n << " m=" << m << endl;
	for (i = 0; i < n; i++) {
		row = GRAPHROW(g, i, m);
		EMPTYSET(row, m);
		}
	
	for (i = 0; i < v; i++) {
		for (j = 0; j < b; j++) {
			if (M[i * b + j] == 0)
				continue;
			p1 = i;
			p2 = v + j;
			row = GRAPHROW(g, p1, m);
			ADDELEMENT(row, p2);
			row = GRAPHROW(g, p2, m);
			ADDELEMENT(row, p1);
			}
		}

	for (i = 0; i < n; i++) {
		lab[i] = i;
		ptn[i] = partition[i];
		}
	//ptn[v - 1] = 0;
	//cout << "nauty_interface_matrix, calling nauty..." << endl;
	//	nauty(g, lab, ptn, NILSET, orbits,
	//&options, &stats, workspace, MAX_WORKSPACE * MAXM, m, n, canong);
	densenauty(g, lab, ptn, orbits, &options, &stats, m, n, canong);
	//cout << "nauty_interface_matrix, numnodes=" << stats.numnodes << endl;
	for (i = 0; i < n; i++) {
		labeling[i] = lab[i];
	}
#if 1
	Ago = ago;
	Base_length = base_length;
	for (i = base_length - 1; i >= 0; i--) {
		Base[base_length - 1 - i] = base[i];
		Transversal_length[base_length - 1 - i] = transversal_length[i];
		}

	for (i = 0; i < aut_counter; i++) {
		for (j = 0; j < n; j++) {
			Aut[i * n + j] = aut[i * n + j];
			}
		}
	Aut_counter = aut_counter;
#endif
	nauty_interface_free_data();
#endif
}

void nauty_interface_matrix_int(int *M, int v, int b, 
	int *labeling, int *partition, 
	int *Aut, int &Aut_counter, 
	int *Base, int &Base_length, 
	int *Transversal_length, int &Ago, int verbose_level)
{
#if HAS_NAUTY
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	static DEFAULTOPTIONS(options);
	statsblk(stats);
	set *row;
	int m, n, i, j, p1, p2;

	if (f_v) {
		cout << "nauty_interface_matrix_int "
				"v=" << v << " b=" << b << endl;
		}
	options.getcanon = TRUE;
	options.defaultptn = FALSE;
	//options.writeautoms = TRUE;
// 		options.cartesian = TRUE;
// 		options.writemarkers = TRUE;

	n = v + b;
	aut_counter = 0;

	// global variables in nauty.c:
	nauty_interface_allocate_data(n);


	base_length = 0;

	m = (n + WORDSIZE - 1) / WORDSIZE;
	if (n >= MAXN) {
		cout << "nauty_interface_matrix_int n >= MAXN" << endl;
		cout << "nauty_interface_matrix_int n = " << n << endl;
		cout << "nauty_interface_matrix_int MAXN = " << (int)MAXN << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "nauty_interface_matrix_int n = " << n << " m=" << m << endl;
		}
	for (i = 0; i < n; i++) {
		row = GRAPHROW(g, i, m);
		EMPTYSET(row, m);
		}
	
	for (i = 0; i < v; i++) {
		for (j = 0; j < b; j++) {
			if (M[i * b + j] == 0)
				continue;
			p1 = i;
			p2 = v + j;
			row = GRAPHROW(g, p1, m);
			ADDELEMENT(row, p2);
			row = GRAPHROW(g, p2, m);
			ADDELEMENT(row, p1);
			}
		}

	for (i = 0; i < n; i++) {
		lab[i] = i;
		ptn[i] = partition[i];
		}
	//ptn[v - 1] = 0;
	if (f_vv) {
		cout << "nauty_interface_matrix_int, calling nauty..." << endl;
	}
	//	nauty(g, lab, ptn, NILSET, orbits,
	//&options, &stats, workspace, MAX_WORKSPACE * MAXM, m, n, canong);
	densenauty(g, lab, ptn, orbits, &options, &stats, m, n, canong);
	if (f_vv) {
		cout << "nauty_interface_matrix_int, after nauty" << endl;
		cout << "nauty_interface_matrix_int, ago=" << ago << endl;
		cout << "nauty_interface_matrix_int, numnodes=" << stats.numnodes << endl;
		}
	for (i = 0; i < n; i++) {
		labeling[i] = lab[i];
		}
#if 1
	Ago = ago;
	Base_length = base_length;
	for (i = base_length - 1; i >= 0; i--) {
		Base[base_length - 1 - i] = base[i];
		Transversal_length[base_length - 1 - i] = transversal_length[i];
		}

	for (i = 0; i < aut_counter; i++) {
		for (j = 0; j < n; j++) {
			Aut[i * n + j] = aut[i * n + j];
			}
		}
	Aut_counter = aut_counter;
#endif
	nauty_interface_free_data();
#endif
}

#if 1
static void nauty_interface_allocate_data(int n)
{
#if HAS_NAUTY
	aut = new int[n * n];
	base = new int[n * n];
	transversal_length = new int[n * n];
#endif
}

static void nauty_interface_free_data()
{
#if HAS_NAUTY
	delete [] base;
	delete [] aut;
	delete [] transversal_length;
#endif
}
#endif


}
}



