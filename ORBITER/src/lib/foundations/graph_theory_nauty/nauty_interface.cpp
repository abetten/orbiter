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


#define HAS_NAUTY 1


#if HAS_NAUTY



#define MAXN 20000
#include "nauty.h"

#define MAX_WORKSPACE 50000

static graph g[MAXN * MAXM];
static graph canong[MAXN * MAXM];
static nvector lab[MAXN], ptn[MAXN], orbits[MAXN];
//static setword workspace[MAX_WORKSPACE * MAXM];

using namespace std;

typedef long int INT;
typedef unsigned char uchar;

INT bitvector_s_i(uchar *bitvec, INT i);
INT ij2k(INT i, INT j, INT n);

static void nauty_interface_allocate_data(INT n);
static void nauty_interface_free_data();
#else
#endif


typedef long int INT;
typedef unsigned char uchar;


void nauty_interface_graph_bitvec(INT v, uchar *bitvector_adjacency, 
	INT *labeling, INT *partition, 
	INT *Aut, INT &Aut_counter, 
	INT *Base, INT &Base_length, 
	INT *Transversal_length, INT &Ago, INT verbose_level)
{
#if HAS_NAUTY
	INT f_v = (verbose_level >= 1);
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
		cout << "nauty_interface_graph_bitvec() n >= MAXN" << endl;
		exit(1);
		}
	//cout << "nauty_interface_graph_bitvec() n = " << n << " m=" << m << endl;
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
//	nauty(g, lab, ptn, NILSET, orbits, &options, &stats, workspace, MAX_WORKSPACE * MAXM, m, n, canong);
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

void nauty_interface_graph_INT(INT v, INT *Adj, 
	INT *labeling, INT *partition, 
	INT *Aut, INT &Aut_counter, 
	INT *Base, INT &Base_length, 
	INT *Transversal_length, INT &Ago, INT verbose_level)
{
#if HAS_NAUTY
	INT f_v = (verbose_level >= 1);
	static DEFAULTOPTIONS(options);
	statsblk(stats);
	set *row;
	int m, n, i, j;

	if (f_v) {
		cout << "nauty_interface_graph_INT" << endl;
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
		cout << "nauty_interface_graph_INT n >= MAXN" << endl;
		exit(1);
		}
	//cout << "nauty_interface_graph_INT() n = " << n << " m=" << m << endl;
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
		cout << "nauty_interface_graph_INT calling nauty" << endl;
	}
//	nauty(g, lab, ptn, NILSET, orbits, &options, &stats, workspace, MAX_WORKSPACE * MAXM, m, n, canong);
	densenauty(g, lab, ptn, orbits, &options, &stats, m, n, canong);
	if (f_v) {
		cout << "nauty_interface_graph_INT after nauty" << endl;
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
		cout << "nauty_interface_graph_INT done" << endl;
		}
#endif
}


void nauty_interface_INT(INT v, INT b, INT *X, INT nb_inc, 
	INT *labeling, INT *partition, 
	INT *Aut, INT &Aut_counter, 
	INT *Base, INT &Base_length, 
	INT *Transversal_length, INT &Ago)
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
		cout << "nauty_interface_INT() n >= MAXN" << endl;
		cout << "nauty_interface_INT() n = " << n << endl;
		cout << "nauty_interface_INT() MAXN = " << (INT)MAXN << endl;
		exit(1);
		}
	//cout << "nauty_interface_INT() n = " << n << " m=" << m << endl;
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

void nauty_interface(int v, int b, INT *X, INT nb_inc, 
	int *labeling, int *partition, 
	INT *Aut, int &Aut_counter, 
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
		cout << "nauty_interface() n >= MAXN" << endl;
		cout << "nauty_interface() n = " << n << endl;
		cout << "nauty_interface() MAXN = " << (INT)MAXN << endl;
		exit(1);
		}
	//cout << "nauty_interface() n = " << n << " m=" << m << endl;
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

void nauty_interface_matrix(int *M, int v, int b,
	int *labeling, int *partition, 
	INT *Aut, int &Aut_counter, 
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
//	nauty(g, lab, ptn, NILSET, orbits, &options, &stats, workspace, MAX_WORKSPACE * MAXM, m, n, canong);
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

void nauty_interface_matrix_INT(INT *M, INT v, INT b, 
	INT *labeling, INT *partition, 
	INT *Aut, INT &Aut_counter, 
	INT *Base, INT &Base_length, 
	INT *Transversal_length, INT &Ago, INT verbose_level)
{
#if HAS_NAUTY
	INT f_v = (verbose_level >= 1);
	INT f_vv = (verbose_level >= 2);
	static DEFAULTOPTIONS(options);
	statsblk(stats);
	set *row;
	INT m, n, i, j, p1, p2;

	if (f_v) {
		cout << "nauty_interface_matrix_INT v=" << v << " b=" << b << endl;
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
		cout << "nauty_interface_matrix_INT n >= MAXN" << endl;
		cout << "nauty_interface_matrix_INT() n = " << n << endl;
		cout << "nauty_interface_matrix_INT() MAXN = " << (INT)MAXN << endl;
		exit(1);
		}
	if (f_vv) {
		cout << "nauty_interface_matrix_INT n = " << n << " m=" << m << endl;
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
		cout << "nauty_interface_matrix_INT, calling nauty..." << endl;
	}
//	nauty(g, lab, ptn, NILSET, orbits, &options, &stats, workspace, MAX_WORKSPACE * MAXM, m, n, canong);
	densenauty(g, lab, ptn, orbits, &options, &stats, m, n, canong);
	if (f_vv) {
		cout << "nauty_interface_matrix_INT, after nauty" << endl;
		cout << "nauty_interface_matrix_INT, ago=" << ago << endl;
		cout << "nauty_interface_matrix_INT, numnodes=" << stats.numnodes << endl;
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
static void nauty_interface_allocate_data(INT n)
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



