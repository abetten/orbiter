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


#define MAXN 300000
//#define MAXN 75000
//#define MAXN 70000
//#define MAXN 22000




// must be defined before reading nauty.h


#include "../combinatorics/graph_theory_nauty/nauty.h"
#include "foundations.h"

static nvector *lab; // [n]


namespace orbiter {
namespace layer1_foundations {
namespace other {
namespace l1_interfaces {


#define HAS_NAUTY 1


#if HAS_NAUTY



//#define MAX_WORKSPACE 50000


//static graph g[MAXN * MAXM];
//static graph canong[MAXN * MAXM];
//static nvector lab[MAXN], ptn[MAXN], orbits[MAXN];

using namespace std;

typedef unsigned char uchar;

static void nauty_interface_allocate_data(
		int n);
static void nauty_interface_free_data();
static void nauty_interface_fill_nauty_output(
		int n,
		l1_interfaces::nauty_output *NO,
	int verbose_level);
#else
#endif


typedef unsigned char uchar;

nauty_interface::nauty_interface()
{
	Record_birth();

}

nauty_interface::~nauty_interface()
{
	Record_death();

}

void nauty_interface::nauty_interface_graph_bitvec(
		int v,
		data_structures::bitvector *Bitvec,
		int *partition,
		l1_interfaces::nauty_output *NO,
		int verbose_level)
{
#if HAS_NAUTY
	int f_v = (verbose_level >= 1);
	static DEFAULTOPTIONS(options);
	statsblk stats;
	setword *row;
	int m, n, i, j, k;

	if (f_v) {
		cout << "nauty_interface::nauty_interface_graph_bitvec" << endl;
	}
	options.getcanon = true;
	options.defaultptn = false;
	//	options.writeautoms = true;
	// 		options.cartesian = true;
	// 		options.writemarkers = true;

	n = v;
	nauty_interface_allocate_data(n);

	m = (n + WORDSIZE - 1) / WORDSIZE;
	if (n >= MAXN) {
		cout << "nauty_interface::nauty_interface_graph_bitvec "
				"n >= MAXN" << endl;
		exit(1);
	}
	//cout << "nauty_interface_graph_bitvec n = " << n << " m=" << m << endl;


	graph *g; // [n * m];
	graph *canong; // [n * m]
	nvector *ptn; // [n]
	nvector *orbits; // [n]


	g = new graph[n * m];
	canong = new graph[n * m];
	lab = new nvector[n];
	ptn = new nvector[n];
	orbits = new nvector[n];


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

	Int_vec_copy(partition, ptn, n);
	for (i = 0; i < n; i++) {
		lab[i] = i;
		//ptn[i] = partition[i];
	}
	//ptn[v - 1] = 0;
	//cout << "calling nauty..." << endl;
	if (f_v) {
		cout << "nauty_interface::nauty_interface_graph_bitvec "
				"calling nauty (densenauty)" << endl;
	}
	//	nauty(g, lab, ptn, NILSET, orbits, &options,
	//&stats, workspace, MAX_WORKSPACE * MAXM, m, n, canong);

	densenauty(g, lab, ptn, orbits, &options, &stats, m, n, canong);
	if (f_v) {
		cout << "nauty_interface::nauty_interface_graph_bitvec "
				"after nauty (densenauty)" << endl;
	}

	orbiter_kernel_system::Orbiter->nb_calls_to_densenauty++;

	if (f_v) {
		cout << "nauty_interface::nauty_interface_graph_bitvec "
				"after nauty" << endl;
		cout << "base_length=" << base_length << endl;
		cout << "transversal_length=";
		Int_vec_print(cout, transversal_length, base_length);
		cout << endl;
	}
	//cout << "numnodes=" << stats.numnodes << endl;

#if 1

	nauty_interface_fill_nauty_output(n, NO, verbose_level);


#endif
	nauty_interface_free_data();
	if (f_v) {
		cout << "nauty_interface::nauty_interface_graph_bitvec done" << endl;
	}
#endif


	delete [] g;
	delete [] canong;
	delete [] lab;
	delete [] ptn;
	delete [] orbits;

}


void nauty_interface::nauty_interface_graph_int(
		int v, int *Adj,
	int *partition,
	l1_interfaces::nauty_output *NO,
	int verbose_level)
// called from:
// nauty_interface_with_group::create_automorphism_group_of_graph_with_partition_and_labeling
// nauty_interface_with_group::create_automorphism_group_of_graph
// nauty_interface_with_group::create_automorphism_group_and_canonical_labeling_of_graph
{
#if HAS_NAUTY
	int f_v = (verbose_level >= 1);
	static DEFAULTOPTIONS(options);
	statsblk stats;
	setword *row;
	int m, n, i, j;

	if (f_v) {
		cout << "nauty_interface::nauty_interface_graph_int" << endl;
	}
	options.getcanon = true;
	options.defaultptn = false;
//	options.writeautoms = true;
// 		options.cartesian = true;
// 		options.writemarkers = true;

	n = v;
	nauty_interface_allocate_data(n);

	m = (n + WORDSIZE - 1) / WORDSIZE;
		// m is the number of words needed to encode one row of the adjacency matrix

	if (n >= MAXN) {
		cout << "nauty_interface::nauty_interface_graph_int n >= MAXN" << endl;
		exit(1);
	}
	//cout << "nauty_interface_graph_int n = " << n << " m=" << m << endl;


	graph *g; // [MAXN * MAXM];
	graph *canong; // [MAXN * MAXM]
	nvector *ptn; // [MAXN]
	nvector *orbits; // [MAXN]


	g = new graph[n * m];
	canong = new graph[n * m];
	lab = new nvector[n];
	ptn = new nvector[n];
	orbits = new nvector[n];


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

	Int_vec_copy(partition, ptn, n);
	for (i = 0; i < n; i++) {
		lab[i] = i;
		//ptn[i] = partition[i];
	}
	//ptn[v - 1] = 0;
	//cout << "calling nauty..." << endl;
	if (f_v) {
		cout << "nauty_interface::nauty_interface_graph_int calling nauty" << endl;
	}
//	nauty(g, lab, ptn, NILSET, orbits, &options, &stats, workspace, MAX_WORKSPACE * MAXM, m, n, canong);
	densenauty(g, lab, ptn, orbits, &options, &stats, m, n, canong);

	orbiter_kernel_system::Orbiter->nb_calls_to_densenauty++;

	if (f_v) {
		cout << "nauty_interface::nauty_interface_graph_int after nauty" << endl;
	}
	//cout << "numnodes=" << stats.numnodes << endl;

#if 1

	nauty_interface_fill_nauty_output(n, NO, verbose_level);


#endif
	nauty_interface_free_data();


	delete [] g;
	delete [] canong;
	delete [] lab;
	delete [] ptn;
	delete [] orbits;

	if (f_v) {
		cout << "nauty_interface::nauty_interface_graph_int done" << endl;
	}
#endif

}


void nauty_interface::Levi_graph(
		combinatorics::canonical_form_classification::encoded_combinatorial_object *Enc,
		l1_interfaces::nauty_output *NO,
	int verbose_level)
// this is called only from
// nauty_interface_for_OwCF::run_nauty_for_OwCF and
// nauty_interface_for_OwCF::run_nauty_for_OwCF_basic
{
#if HAS_NAUTY
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	static DEFAULTOPTIONS(options);
	statsblk stats;
	setword *row;
	int m, n, i, j, p1, p2;

	if (f_v) {
		cout << "nauty_interface::Levi_graph "
				"nb_rows=" << Enc->nb_rows
				<< " nb_cols=" << Enc->nb_cols << endl;
	}
	options.getcanon = true;
	options.defaultptn = false;
	//options.writeautoms = true;
// 		options.cartesian = true;
// 		options.writemarkers = true;

	n = Enc->nb_rows + Enc->nb_cols;

	// global variables in nauty.c:
	if (f_vv) {
		cout << "nauty_interface::Levi_graph "
				"before nauty_interface_allocate_data" << endl;
	}

	nauty_interface_allocate_data(n);

	if (f_vv) {
		cout << "nauty_interface::Levi_graph "
				"after nauty_interface_allocate_data" << endl;
	}


	base_length = 0;

	m = (n + WORDSIZE - 1) / WORDSIZE;
	if (n >= MAXN) {
		cout << "nauty_interface::Levi_graph n >= MAXN" << endl;
		cout << "nauty_interface::Levi_graph n = " << n << endl;
		cout << "nauty_interface::Levi_graph MAXN = " << (int)MAXN << endl;
		exit(1);
	}
	if (f_vv) {
		cout << "nauty_interface::Levi_graph n = " << n << " m=" << m << endl;
	}


	graph *g; // [MAXN * MAXM];
	graph *canong; // [MAXN * MAXM]
	nvector *ptn; // [MAXN]
	nvector *orbits; // [MAXN]


	g = new graph[n * m];
	canong = new graph[n * m];
	lab = new nvector[n];
	ptn = new nvector[n];
	orbits = new nvector[n];

	for (i = 0; i < n; i++) {
		row = GRAPHROW(g, i, m);
		EMPTYSET(row, m);
	}

	if (f_vv) {
		cout << "nauty_interface::Levi_graph "
				"init adjacency" << endl;
	}

	// make Levi graph on n vertices:

	for (i = 0; i < Enc->nb_rows; i++) {
		for (j = 0; j < Enc->nb_cols; j++) {
			if (Enc->get_incidence_ij(i, j) == 0) {
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
		cout << "nauty_interface::Levi_graph "
				"init lab[] and ptn[]" << endl;
	}

	Int_vec_copy(Enc->partition, ptn, n);
	for (i = 0; i < n; i++) {
		lab[i] = i;
		//ptn[i] = Enc->partition[i];
	}
	//ptn[v - 1] = 0;
	if (f_vv) {
		cout << "nauty_interface::Levi_graph "
				"before densenauty" << endl;
	}
	//	nauty(g, lab, ptn, NILSET, orbits,
	//&options, &stats, workspace, MAX_WORKSPACE * MAXM, m, n, canong);
	densenauty(g, lab, ptn, orbits, &options, &stats, m, n, canong);
	if (f_vv) {
		cout << "nauty_interface::Levi_graph "
				"after densenauty" << endl;
	}

	orbiter_kernel_system::Orbiter->nb_calls_to_densenauty++;

	if (f_vv) {
		cout << "nauty_interface::Levi_graph "
				"ago=" << ago << endl;
		cout << "nauty_interface::Levi_graph "
				"numnodes=" << stats.numnodes << endl;
	}
#if 1

	nauty_interface_fill_nauty_output(n, NO, verbose_level);


#endif
	nauty_interface_free_data();
#endif

	delete [] g;
	delete [] canong;
	delete [] lab;
	delete [] ptn;
	delete [] orbits;


	if (f_v) {
		cout << "nauty_interface::Levi_graph done" << endl;
	}
}

#if 1
static void nauty_interface_allocate_data(
		int n)
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


static void nauty_interface_fill_nauty_output(
		int n,
		l1_interfaces::nauty_output *NO,
	int verbose_level)
{
	int f_v = false; //(verbose_level >= 1);
	int f_vv = false; // (verbose_level >= 2);
	//Ago = ago;
	algebra::ring_theory::longinteger_domain Dom;
	int i; //, j;

	if (f_v) {
		cout << "nauty_interface_fill_nauty_output" << endl;
	}
	Dom.multiply_up(
			*NO->Ago, transversal_length, base_length,
			0 /* verbose_level*/);

	NO->Base_length = base_length;

	for (i = base_length - 1; i >= 0; i--) {
		NO->Base[base_length - 1 - i] = base[i];
		NO->Base_lint[base_length - 1 - i] = base[i];
		NO->Transversal_length[base_length - 1 - i] = transversal_length[i];
	}
	if (false) {
		cout << "nauty_interface_fill_nauty_output transversal_length : ";
		for (i = base_length - 1; i >= 0; i--) {
			cout << transversal_length[i];
			if (i > 0) {
				cout << " * ";
			}
		}
		cout << endl;
	}

	Int_vec_copy(aut, NO->Aut, aut_counter * n);

	Int_vec_copy(lab, NO->canonical_labeling, n);

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

}}}}





