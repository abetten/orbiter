// invariants_packing.cpp
// 
// Anton Betten
// Feb 21, 2013
//
//
// 
//
//

#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace top_level {




invariants_packing::invariants_packing()
{
	null();
}

invariants_packing::~invariants_packing()
{
	freeself();
}

void invariants_packing::null()
{
	Inv = NULL;
	Ago = NULL;
	Ago_induced = NULL;
	Ago_int = NULL;
}

void invariants_packing::freeself()
{
	if (Inv) {
		delete [] Inv;
		}
	if (Ago) {
		delete [] Ago;
		}
	if (Ago_induced) {
		delete [] Ago_induced;
		}
	if (Ago_int) {
		FREE_int(Ago_int);
		}
	if (f_self_dual) {
		FREE_int(f_self_dual);
		}
	null();
}

void invariants_packing::init(isomorph *Iso,
		packing_classify *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int orbit, i;
	sorting Sorting;
	
	if (f_v) {
		cout << "invariants_packing::init" << endl;
		}
	invariants_packing::P = P;
	//E = P->E;
	T = P->T;
	invariants_packing::Iso = Iso;


	Inv = NEW_OBJECTS(packing_invariants, Iso->Reps->count);


	for (orbit = 0; orbit < Iso->Reps->count; orbit++) {
		
		if (f_v) {
			cout << "invariants_packing::init orbit " << orbit
					<< " / " << Iso->Reps->count << endl;
			}

		int rep, first, c, id;
		int the_packing[1000];
		
		rep = Iso->Reps->rep[orbit];
		first = Iso->orbit_fst[rep];
		c = Iso->starter_number[first];
		id = Iso->orbit_perm[first];		
		Iso->load_solution(id, the_packing);
		
		Inv[orbit].init(P, Iso->prefix_invariants,
				Iso->prefix_tex, orbit, the_packing,
				/*V, */verbose_level - 1);

		}
	if (f_v) {
		cout << "invariants_packing::init "
				"loading invariants done" << endl;
		}


	

	Iso->compute_Ago_Ago_induced(Ago, Ago_induced, verbose_level - 1);

	Ago_int = NEW_int(Iso->Reps->count);
	for (orbit = 0; orbit < Iso->Reps->count; orbit++) {
		Ago_int[orbit] = Ago[orbit].as_int();
		}


	Type_of_packing = NEW_int(
			Iso->Reps->count * P->nb_spreads_up_to_isomorphism);
	Type_idx_of_packing = NEW_int(Iso->Reps->count);
	for (i = 0;
			i < Iso->Reps->count * P->nb_spreads_up_to_isomorphism;
			i++) {
		Type_of_packing[i] = 0;
		}
	
	for (orbit = 0; orbit < Iso->Reps->count; orbit++) {
		int spread_iso_type[1000];
		int rep, first, c, id;
		int the_packing[1000];
		
		rep = Iso->Reps->rep[orbit];
		first = Iso->orbit_fst[rep];
		c = Iso->starter_number[first];
		id = Iso->orbit_perm[first];		
		Iso->load_solution(id, the_packing);
		
		
		for (i = 0; i < Iso->size; i++) {
			spread_iso_type[i] =
					P->Spread_tables->spread_iso_type[the_packing[i]];
			}
		for (i = 0; i < Iso->size; i++) {
			Type_of_packing[orbit *
							P->nb_spreads_up_to_isomorphism +
							spread_iso_type[i]]++;
			}
		}

	List_of_types = NEW_pint(Iso->Reps->count);
	Frequency = NEW_int(Iso->Reps->count);
	nb_types = 0;
	int idx;

	for (orbit = 0; orbit < Iso->Reps->count; orbit++) {
		if (f_vv) {
			cout << "invariants_packing::init "
					"orbit=" << orbit << endl;
			}
		if (Sorting.vec_search((void **)List_of_types,
				packing_types_compare_function, this,
				nb_types,
				Type_of_packing + orbit * P->nb_spreads_up_to_isomorphism,
				idx,
				0 /* verbose_level */)) {
			Frequency[idx]++;
			}
		else {
			if (f_vv) {
				cout << "invariants_packing::init New type ";
				int_vec_print(cout,
						Type_of_packing +
						orbit * P->nb_spreads_up_to_isomorphism,
						P->nb_spreads_up_to_isomorphism);
				cout << " at position " << idx << endl;
				}
			for (i = nb_types; i > idx; i--) {
				List_of_types[i] = List_of_types[i - 1];
				Frequency[i] = Frequency[i - 1];
				}
			List_of_types[idx] =
					NEW_int(P->nb_spreads_up_to_isomorphism);
			Frequency[idx] = 1;
			for (i = 0; i < P->nb_spreads_up_to_isomorphism; i++) {
				List_of_types[idx][i] =
						Type_of_packing[orbit *
								P->nb_spreads_up_to_isomorphism + i];
				}
			nb_types++;
			if (f_vv) {
				cout << "invariants_packing::init "
						"nb_types=" << nb_types << endl;
				}
			}
		}
	if (f_v) {
		cout << "invariants_packing::init "
				"We found " << nb_types
				<< " types of packings" << endl;
		for (i = 0; i < nb_types; i++) {
			int_vec_print(cout, List_of_types[i],
					P->nb_spreads_up_to_isomorphism);
			cout << " : " << Frequency[i] << endl;
			}
		}

	Type_idx_of_packing = NEW_int(Iso->Reps->count);
	for (orbit = 0; orbit < Iso->Reps->count; orbit++) {
		if (Sorting.vec_search((void **)List_of_types,
			packing_types_compare_function, this,
			nb_types,
			Type_of_packing + orbit * P->nb_spreads_up_to_isomorphism,
			idx,
			0 /* verbose_level */)) {
			Type_idx_of_packing[orbit] = idx;
			}
		else {
			cout << "invariants_packing::init "
					"error: did not find type of packing" << endl;
			exit(1);
			}
		}

	if (f_v) {
		cout << "invariants_packing::init "
				"before compute_dual_packings" << endl;
		}
	compute_dual_packings(Iso, verbose_level);
	if (f_v) {
		cout << "invariants_packing::init "
				"after compute_dual_packings" << endl;
		}

	if (f_v) {
		cout << "invariants_packing::init done" << endl;
		}
}

void invariants_packing::compute_dual_packings(
		isomorph *Iso, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int orbit, i;
	int packing[1000];
	int dual_packing[1000];
	sorting Sorting;
	
	if (f_v) {
		cout << "invariants_packing::compute_dual_packings" << endl;
		}
	Dual_idx = NEW_int(Iso->Reps->count);
	f_self_dual = NEW_int(Iso->Reps->count);
	
	for (orbit = 0; orbit < Iso->Reps->count; orbit++) {

		int rep, first, c, id;
		int f_implicit_fusion = TRUE;
		
		rep = Iso->Reps->rep[orbit];
		first = Iso->orbit_fst[rep];
		c = Iso->starter_number[first];
		id = Iso->orbit_perm[first];		
		Iso->load_solution(id, packing);

	
		for (i = 0; i < Iso->size; i++) {
			dual_packing[i] = P->Spread_tables->dual_spread_idx[packing[i]];
			}
		Sorting.int_vec_heapsort(packing, Iso->size);
		Sorting.int_vec_heapsort(dual_packing, Iso->size);
		for (i = 0; i < Iso->size; i++) {
			if (packing[i] != dual_packing[i]) {
				break;
				}
			}
		if (i == Iso->size) {
			f_self_dual[orbit] = TRUE;
			}
		else {
			f_self_dual[orbit] = FALSE;
			}
	

		Dual_idx[orbit] = Iso->identify_database_is_open(
			dual_packing,
			f_implicit_fusion, verbose_level - 3);
		}

	file_io Fio;

	Fio.int_vecs_write_csv(Dual_idx, f_self_dual,
		Iso->Reps->count, "Dual_idx.csv", "dual_idx", "f_self_dual");
	Fio.int_vec_write_csv(P->Spread_tables->dual_spread_idx,
		P->Spread_tables->nb_spreads, "Dual_spread_idx.csv", "dual_spread_idx");
	
	if (f_v) {
		cout << "invariants_packing::compute_dual_packings done" << endl;
		}
}

void invariants_packing::make_table(
	isomorph *Iso, ostream &ost,
	int f_only_self_dual, int f_only_not_self_dual, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i;
	char fname[1000];
	file_io Fio;
	
	if (f_v) {
		cout << "invariants_packing::make_table" << endl;
		}

	if (f_only_self_dual) {
		ost << "\\chapter{Self Polar Packings}" << endl << endl;
		}
	else if (f_only_not_self_dual) {
		ost << "\\chapter{Not Self Polar Packings}" << endl << endl;
		}
	else {
		ost << "\\chapter{All Packings}" << endl << endl;
		}

	ost << "For each packing, let $a_i$ be the number of spreads "
			"of isomorphism type $i$\\\\" << endl;
	ost << "The type of the packing is the vector $(a_0,\\ldots, "
			"a_{N-1})$ where $N$ is the number of isomorphism types "
			"of spreads of $\\PG(3," << P->q << ")$\\\\" << endl;
	ost << endl;
	ost << "\\begin{center}" << endl;
	ost << "\\begin{tabular}{|c|c|l|p{6cm}}" << endl;
	ost << "\\hline" << endl;
	ost << "Type & Number of Packings & Distr. of Aut Group "
			"Orders\\\\" << endl;
	ost << "\\hline" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_types; i++) {
		if (f_only_self_dual) {
			sprintf(fname, "ids_of_self_dual_type_%d.csv", i);
			}
		else if (f_only_not_self_dual) {
			sprintf(fname, "ids_of_not_self_dual_type_%d.csv", i);
			}
		else {
			sprintf(fname, "ids_of_all_type_%d.csv", i);
			}
		int_vec_print(ost, List_of_types[i],
				P->nb_spreads_up_to_isomorphism);
		ost << " & ";
		// ost << Frequency[i] << " & ";

		int *set;
		int *ago;
		int nb, j, dual, a;

		nb = 0;
		set = NEW_int(Frequency[i]);
		ago = NEW_int(Frequency[i]);
		for (j = 0; j < Iso->Reps->count; j++) {
			if (Type_idx_of_packing[j] == i) {
				if (f_only_self_dual) {
					dual = Dual_idx[j];
					if (dual == j) {
						set[nb++] = j;
						}
					}
				else if (f_only_not_self_dual) {
					dual = Dual_idx[j];
					if (dual != j && dual > j) {
						set[nb++] = j;
						}
					}
				else {
					set[nb++] = j;
					}
				}
			}
		for (j = 0; j < nb; j++) {
			a = set[j];
			ago[j] = Ago_int[a];
			}

		Fio.int_vecs_write_csv(set, ago, nb, fname, "ID", "ago");

		classify C;

		C.init(ago, nb, FALSE, 0);
		
		ost << nb << " & ";
		C.print_naked_tex(ost, TRUE /*f_backwards*/);

		ost << "\\\\" << endl;
		FREE_int(set);
		FREE_int(ago);
		}
	ost << "\\hline" << endl;
	ost << "\\end{tabular}" << endl;
	ost << "\\end{center}" << endl << endl;

	if (f_v) {
		cout << "invariants_packing::make_table done" << endl;
		}
}


int packing_types_compare_function(void *a, void *b, void *data)
{
	invariants_packing *inv = (invariants_packing *) data;
	int *A = (int *) a;
	int *B = (int *) b;
	int i;

	for (i = 0; i < inv->P->nb_spreads_up_to_isomorphism; i++) {
		if (A[i] > B[i]) {
			return 1;
			}
		if (A[i] < B[i]) {
			return -1;
			}
		}
	return 0;
}


}}

