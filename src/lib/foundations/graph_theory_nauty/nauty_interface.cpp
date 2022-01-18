// nauty_interface.cpp
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


// don't make this too big, or otherwise the linker will complain on linux machine.
// 100000 is too big for Linux under gcc, but OK for Macintosh with LLVM


#define MAXN 50000
//#define MAXN 22000


// must be defined before reading nauty.h


#include "nauty.h"
#include "foundations.h"



namespace orbiter {
namespace foundations {


#define HAS_NAUTY 1


#if HAS_NAUTY



#define MAX_WORKSPACE 50000


static graph g[MAXN * MAXM];
static graph canong[MAXN * MAXM];
static nvector lab[MAXN], ptn[MAXN], orbits[MAXN];
//static setword workspace[MAX_WORKSPACE * MAXM];

using namespace std;

typedef unsigned char uchar;

static void nauty_interface_allocate_data(int n);
static void nauty_interface_free_data();
static void nauty_interface_fill_nauty_output(int n,
		data_structures::nauty_output *NO,
	int verbose_level);
#else
#endif


typedef unsigned char uchar;


void nauty_interface::nauty_interface_graph_bitvec(int v,
		data_structures::bitvector *Bitvec,
		int *partition,
		data_structures::nauty_output *NO,
		int verbose_level)
{
#if HAS_NAUTY
	int f_v = (verbose_level >= 1);
	static DEFAULTOPTIONS(options);
	statsblk stats;
	set *row;
	int m, n, i, j, k;

	if (f_v) {
		cout << "nauty_interface::nauty_interface_graph_bitvec" << endl;
	}
	options.getcanon = TRUE;
	options.defaultptn = FALSE;
	//	options.writeautoms = TRUE;
	// 		options.cartesian = TRUE;
	// 		options.writemarkers = TRUE;

	n = v;
	nauty_interface_allocate_data(n);

	m = (n + WORDSIZE - 1) / WORDSIZE;
	if (n >= MAXN) {
		cout << "nauty_interface::nauty_interface_graph_bitvec n >= MAXN" << endl;
		exit(1);
	}
	//cout << "nauty_interface_graph_bitvec n = " << n << " m=" << m << endl;
	for (i = 0; i < n; i++) {
		row = GRAPHROW(g, i, m);
		EMPTYSET(row, m);
	}
	
	for (i = 0, k = 0; i < v; i++) {
		for (j = i + 1; j < v; j++, k++) {
			//k = callback_ij2k(i, j, v);
			if (Bitvec->s_i(k)) {
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
		cout << "nauty_interface::nauty_interface_graph_bitvec calling nauty" << endl;
	}
	//	nauty(g, lab, ptn, NILSET, orbits, &options,
	//&stats, workspace, MAX_WORKSPACE * MAXM, m, n, canong);
	densenauty(g, lab, ptn, orbits, &options, &stats, m, n, canong);

	Orbiter->nb_calls_to_densenauty++;

	if (f_v) {
		cout << "nauty_interface::nauty_interface_graph_bitvec after nauty" << endl;
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
#if 0
	for (i = 0; i < n; i++) {
		labeling[i] = lab[i];
	}
#endif

#if 1

	nauty_interface_fill_nauty_output(n, NO, verbose_level);

#if 0
	//Ago = ago;
	longinteger_domain Dom;

	Dom.multiply_up(Ago, transversal_length, base_length, 0 /* verbose_level*/);
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

#endif
	nauty_interface_free_data();
	if (f_v) {
		cout << "nauty_interface::nauty_interface_graph_bitvec done" << endl;
	}
#endif
}


void nauty_interface::nauty_interface_graph_int(int v, int *Adj,
	int *partition,
	data_structures::nauty_output *NO,
	int verbose_level)
{
#if HAS_NAUTY
	int f_v = (verbose_level >= 1);
	static DEFAULTOPTIONS(options);
	statsblk stats;
	set *row;
	int m, n, i, j;

	if (f_v) {
		cout << "nauty_interface::nauty_interface_graph_int" << endl;
	}
	options.getcanon = TRUE;
	options.defaultptn = FALSE;
//	options.writeautoms = TRUE;
// 		options.cartesian = TRUE;
// 		options.writemarkers = TRUE;

	n = v;
	nauty_interface_allocate_data(n);

	m = (n + WORDSIZE - 1) / WORDSIZE;
	if (n >= MAXN) {
		cout << "nauty_interface::nauty_interface_graph_int n >= MAXN" << endl;
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
		cout << "nauty_interface::nauty_interface_graph_int calling nauty" << endl;
	}
//	nauty(g, lab, ptn, NILSET, orbits, &options, &stats, workspace, MAX_WORKSPACE * MAXM, m, n, canong);
	densenauty(g, lab, ptn, orbits, &options, &stats, m, n, canong);

	Orbiter->nb_calls_to_densenauty++;

	if (f_v) {
		cout << "nauty_interface::nauty_interface_graph_int after nauty" << endl;
	}
	//cout << "numnodes=" << stats.numnodes << endl;
#if 0
	for (i = 0; i < n; i++) {
		labeling[i] = lab[i];
	}
#endif

#if 1

	nauty_interface_fill_nauty_output(n, NO, verbose_level);

#if 0
	//Ago = ago;
	longinteger_domain Dom;

	Dom.multiply_up(Ago, transversal_length, base_length, 0 /* verbose_level*/);
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

#endif
	nauty_interface_free_data();
	if (f_v) {
		cout << "nauty_interface::nauty_interface_graph_int done" << endl;
	}
#endif
}


void nauty_interface::nauty_interface_matrix_int(
		combinatorics::encoded_combinatorial_object *Enc,
	data_structures::nauty_output *NO,
	int verbose_level)
{
#if HAS_NAUTY
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	static DEFAULTOPTIONS(options);
	statsblk stats;
	set *row;
	int m, n, i, j, p1, p2;

	if (f_v) {
		cout << "nauty_interface::nauty_interface_matrix_int "
				"nb_rows=" << Enc->nb_rows << " nb_cols=" << Enc->nb_cols << endl;
	}
	options.getcanon = TRUE;
	options.defaultptn = FALSE;
	//options.writeautoms = TRUE;
// 		options.cartesian = TRUE;
// 		options.writemarkers = TRUE;

	n = Enc->nb_rows + Enc->nb_cols;

	// global variables in nauty.c:
	if (f_vv) {
		cout << "nauty_interface::nauty_interface_matrix_int before nauty_interface_allocate_data" << endl;
	}

	nauty_interface_allocate_data(n);

	if (f_vv) {
		cout << "nauty_interface::nauty_interface_matrix_int after nauty_interface_allocate_data" << endl;
	}


	base_length = 0;

	m = (n + WORDSIZE - 1) / WORDSIZE;
	if (n >= MAXN) {
		cout << "nauty_interface::nauty_interface_matrix_int n >= MAXN" << endl;
		cout << "nauty_interface::nauty_interface_matrix_int n = " << n << endl;
		cout << "nauty_interface::nauty_interface_matrix_int MAXN = " << (int)MAXN << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "nauty_interface::nauty_interface_matrix_int n = " << n << " m=" << m << endl;
	}
	for (i = 0; i < n; i++) {
		row = GRAPHROW(g, i, m);
		EMPTYSET(row, m);
	}

	if (f_vv) {
		cout << "nauty_interface::nauty_interface_matrix_int init adjacency" << endl;
	}

	for (i = 0; i < Enc->nb_rows; i++) {
		for (j = 0; j < Enc->nb_cols; j++) {
			if (Enc->Incma[i * Enc->nb_cols + j] == 0) {
				continue;
			}
			p1 = i;
			p2 = Enc->nb_rows + j;
			row = GRAPHROW(g, p1, m);
			ADDELEMENT(row, p2);
			row = GRAPHROW(g, p2, m);
			ADDELEMENT(row, p1);
		}
	}

	if (f_vv) {
		cout << "nauty_interface::nauty_interface_matrix_int init lab[] and ptn[]" << endl;
	}
	for (i = 0; i < n; i++) {
		lab[i] = i;
		ptn[i] = Enc->partition[i];
	}
	//ptn[v - 1] = 0;
	if (f_vv) {
		cout << "nauty_interface::nauty_interface_matrix_int, calling densenauty" << endl;
	}
	//	nauty(g, lab, ptn, NILSET, orbits,
	//&options, &stats, workspace, MAX_WORKSPACE * MAXM, m, n, canong);
	densenauty(g, lab, ptn, orbits, &options, &stats, m, n, canong);

	Orbiter->nb_calls_to_densenauty++;

	if (f_vv) {
		cout << "nauty_interface::nauty_interface_matrix_int, after densenauty" << endl;
		cout << "nauty_interface::nauty_interface_matrix_int, ago=" << ago << endl;
		cout << "nauty_interface::nauty_interface_matrix_int, numnodes=" << stats.numnodes << endl;
	}
#if 1

	nauty_interface_fill_nauty_output(n, NO, verbose_level);


#endif
	nauty_interface_free_data();
#endif
	if (f_v) {
		cout << "nauty_interface::nauty_interface_matrix_int done" << endl;
	}
}

#if 1
static void nauty_interface_allocate_data(int n)
{
#if HAS_NAUTY
	aut = new int[n * n];
	base = new int[n * n];
	transversal_length = new int[n * n];

	aut_counter = 0;
	base_length = 0;
	nb_firstpathnode = 0;
	nb_othernode = 0;
	nb_processnode = 0;
	nb_firstterminal = 0;

#if 1
	fp_nauty = NULL;
#else
	fp_nauty = fopen("nauty_log.txt", "w");
#endif

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


static void nauty_interface_fill_nauty_output(int n,
	data_structures::nauty_output *NO,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	//Ago = ago;
	ring_theory::longinteger_domain Dom;
	int i, j;

	if (f_v) {
		cout << "nauty_interface_fill_nauty_output" << endl;
	}
	Dom.multiply_up(*NO->Ago, transversal_length, base_length, 0 /* verbose_level*/);
	NO->Base_length = base_length;
	for (i = base_length - 1; i >= 0; i--) {
		NO->Base[base_length - 1 - i] = base[i];
		NO->Base_lint[base_length - 1 - i] = base[i];
		NO->Transversal_length[base_length - 1 - i] = transversal_length[i];
	}
	if (f_vv) {
		cout << "transversal_length : ";
		for (i = base_length - 1; i >= 0; i--) {
			cout << transversal_length[i];
			if (i > 0) {
				cout << " * ";
			}
		}
		cout << endl;
	}

	for (i = 0; i < aut_counter; i++) {
		for (j = 0; j < n; j++) {
			NO->Aut[i * n + j] = aut[i * n + j];
		}
	}

	for (i = 0; i < n; i++) {
		NO->canonical_labeling[i] = lab[i];
	}

	NO->Aut_counter = aut_counter;
	NO->nb_firstpathnode = nb_firstpathnode;
	NO->nb_othernode = nb_othernode;
	NO->nb_processnode = nb_processnode;
	NO->nb_firstterminal = nb_firstterminal;
	if (fp_nauty) {
		fprintf(fp_nauty, "-1\n");
		fclose(fp_nauty);
		fp_nauty = NULL;
	}

	if (f_v) {
		cout << "nauty_interface_fill_nauty_output done" << endl;
	}
}

}
}



