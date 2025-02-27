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
namespace layer5_applications {
namespace packings {


//static int packing_types_compare_function(void *a, void *b, void *data);



invariants_packing::invariants_packing()
{
	Record_birth();
	T = NULL;
	P = NULL;
	Iso = NULL;
	Inv = NULL;
	Ago = NULL;
	Ago_induced = NULL;
	Ago_int = NULL;
	//Type_of_packing = NULL;
	Spread_type_of_packing = NULL;
	Classify = NULL;
#if 0
	List_of_types = NULL;
	Frequency = NULL;
	nb_types = 0;
	packing_type_idx = NULL;
#endif
	Dual_idx = NULL;
	f_self_dual = NULL;

}

invariants_packing::~invariants_packing()
{
	Record_death();
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
	if (Spread_type_of_packing) {
		FREE_int(Spread_type_of_packing);
	}
	if (Classify) {
		FREE_OBJECT(Classify);
	}
#if 0
	if (List_of_types) {
		int i;
		for (i = 0; i < nb_types; i++) {
			FREE_int(List_of_types[i]);
		}
		FREE_pint(List_of_types);
	}
	if (Frequency) {
		FREE_int(Frequency);
	}
	if (packing_type_idx) {
		FREE_int(packing_type_idx);
	}
#endif
	if (Dual_idx) {
		FREE_int(Dual_idx);
	}
	if (f_self_dual) {
		FREE_int(f_self_dual);
	}
}

void invariants_packing::init(
		isomorph::isomorph *Iso,
		packing_classify *P, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int orbit, i;
	other::data_structures::sorting Sorting;
	
	if (f_v) {
		cout << "invariants_packing::init" << endl;
		}
	invariants_packing::P = P;
	T = P->T;
	invariants_packing::Iso = Iso;


	Inv = NEW_OBJECTS(packing_invariants, Iso->Folding->Reps->count);


	for (orbit = 0; orbit < Iso->Folding->Reps->count; orbit++) {
		
		if (f_v) {
			cout << "invariants_packing::init orbit " << orbit
					<< " / " << Iso->Folding->Reps->count << endl;
		}

		int rep, first, /*c,*/ id;
		
		rep = Iso->Folding->Reps->rep[orbit];
		first = Iso->Lifting->flag_orbit_solution_first[rep];
		//c = Iso->starter_number[first];
		id = Iso->Lifting->orbit_perm[first];
		Iso->Lifting->load_solution(id, P->the_packing, verbose_level - 1);
		
		Inv[orbit].init(P, Iso->prefix_invariants,
				Iso->prefix_tex, orbit, P->the_packing,
				verbose_level - 1);

	}
	if (f_v) {
		cout << "invariants_packing::init loading invariants done" << endl;
	}


	

	Iso->Folding->compute_Ago_Ago_induced(Ago, Ago_induced, verbose_level - 1);

	Ago_int = NEW_int(Iso->Folding->Reps->count);
	for (orbit = 0; orbit < Iso->Folding->Reps->count; orbit++) {
		Ago_int[orbit] = Ago[orbit].as_int();
	}


	Spread_type_of_packing = NEW_int(Iso->Folding->Reps->count * P->Spread_table_with_selection->nb_iso_types_of_spreads);
	Int_vec_zero(Spread_type_of_packing, Iso->Folding->Reps->count * P->Spread_table_with_selection->nb_iso_types_of_spreads);
	
	// compute Spread_type_of_packing:

	for (orbit = 0; orbit < Iso->Folding->Reps->count; orbit++) {
		int rep, first, /*c,*/ id, a;
		
		rep = Iso->Folding->Reps->rep[orbit];
		first = Iso->Lifting->flag_orbit_solution_first[rep];
		//c = Iso->starter_number[first];
		id = Iso->Lifting->orbit_perm[first];
		Iso->Lifting->load_solution(id, P->the_packing, verbose_level - 1);
		
		
		for (i = 0; i < Iso->size; i++) {
			a = P->Spread_table_with_selection->Spread_tables->spread_iso_type[P->the_packing[i]];
			Spread_type_of_packing[orbit * P->Spread_table_with_selection->nb_iso_types_of_spreads + a]++;
		}
	}

	Classify = NEW_OBJECT(other::data_structures::tally_vector_data);

	Classify->init(Spread_type_of_packing, Iso->size /* data_length */,
			P->Spread_table_with_selection->nb_iso_types_of_spreads /* data_set_sz */,
			verbose_level);

#if 0
	List_of_types = NEW_pint(Iso->Reps->count);
	Frequency = NEW_int(Iso->Reps->count);
	nb_types = 0;
	int idx;

	for (orbit = 0; orbit < Iso->Reps->count; orbit++) {
		if (f_vv) {
			cout << "invariants_packing::init orbit=" << orbit << endl;
		}
		if (Sorting.vec_search((void **)List_of_types,
				packing_types_compare_function, this,
				nb_types,
				Spread_type_of_packing + orbit * P->nb_iso_types_of_spreads,
				idx,
				0 /* verbose_level */)) {
			Frequency[idx]++;
		}
		else {
			if (f_vv) {
				cout << "invariants_packing::init New type ";
				int_vec_print(cout,
						Spread_type_of_packing + orbit * P->nb_iso_types_of_spreads,
						P->nb_iso_types_of_spreads);
				cout << " at position " << idx << endl;
			}
			for (i = nb_types; i > idx; i--) {
				List_of_types[i] = List_of_types[i - 1];
				Frequency[i] = Frequency[i - 1];
			}
			List_of_types[idx] = NEW_int(P->nb_iso_types_of_spreads);
			Frequency[idx] = 1;
			int_vec_copy(Spread_type_of_packing + orbit * P->nb_iso_types_of_spreads, List_of_types[idx],
					P->nb_iso_types_of_spreads);
			nb_types++;
			if (f_vv) {
				cout << "invariants_packing::init "
						"nb_types=" << nb_types << endl;
			}
		}
	}
#endif

	if (f_v) {
		cout << "invariants_packing::init "
				"We found " << Classify->nb_types
				<< " types of packings" << endl;
		for (i = 0; i < Classify->nb_types; i++) {
			Int_vec_print(cout,
					Classify->Reps_in_lex_order[i],
					P->Spread_table_with_selection->nb_iso_types_of_spreads);
			cout << " : " << Classify->Frequency_in_lex_order[i] << endl;
		}
	}

#if 0
	packing_type_idx = NEW_int(Iso->Reps->count);
	for (orbit = 0; orbit < Iso->Reps->count; orbit++) {
		if (Sorting.vec_search((void **)List_of_types,
			packing_types_compare_function, this,
			nb_types,
			Spread_type_of_packing + orbit * P->nb_iso_types_of_spreads,
			idx,
			0 /* verbose_level */)) {
			packing_type_idx[orbit] = idx;
		}
		else {
			cout << "invariants_packing::init "
					"error: did not find type of packing" << endl;
			exit(1);
		}
	}
#endif

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
		isomorph::isomorph *Iso, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int orbit, i;
	other::data_structures::sorting Sorting;
	
	if (f_v) {
		cout << "invariants_packing::compute_dual_packings" << endl;
	}
	Dual_idx = NEW_int(Iso->Folding->Reps->count);
	f_self_dual = NEW_int(Iso->Folding->Reps->count);
	
	for (orbit = 0; orbit < Iso->Folding->Reps->count; orbit++) {

		int rep, first, /*c,*/ id;
		int f_implicit_fusion = true;
		
		rep = Iso->Folding->Reps->rep[orbit];
		first = Iso->Lifting->flag_orbit_solution_first[rep];
		//c = Iso->starter_number[first];
		id = Iso->Lifting->orbit_perm[first];
		Iso->Lifting->load_solution(id, P->the_packing, verbose_level - 1);

	
		for (i = 0; i < Iso->size; i++) {
			P->dual_packing[i] = P->Spread_table_with_selection->Spread_tables->dual_spread_idx[P->the_packing[i]];
		}
		Sorting.lint_vec_heapsort(P->the_packing, Iso->size);
		Sorting.lint_vec_heapsort(P->dual_packing, Iso->size);
		for (i = 0; i < Iso->size; i++) {
			if (P->the_packing[i] != P->dual_packing[i]) {
				break;
			}
		}
		if (i == Iso->size) {
			f_self_dual[orbit] = true;
		}
		else {
			f_self_dual[orbit] = false;
		}
	

		Dual_idx[orbit] = Iso->Folding->identify_database_is_open(
			P->dual_packing,
			f_implicit_fusion, verbose_level - 3);
	}

	other::orbiter_kernel_system::file_io Fio;
	string fname;
	string label, label1, label2;

	fname.assign("Dual_idx.csv");
	label1.assign("dual_idx");
	label2.assign("f_self_dual");
	Fio.Csv_file_support->int_vecs_write_csv(
			Dual_idx, f_self_dual,
		Iso->Folding->Reps->count, fname, label1, label2);

	fname.assign("Dual_spread_idx.csv");
	label.assign("dual_spread_idx");
	Fio.Csv_file_support->lint_vec_write_csv(
			P->Spread_table_with_selection->Spread_tables->dual_spread_idx,
		P->Spread_table_with_selection->Spread_tables->nb_spreads,
		fname, label);
	
	if (f_v) {
		cout << "invariants_packing::compute_dual_packings done" << endl;
	}
}

void invariants_packing::make_table(
		isomorph::isomorph *Iso, std::ostream &ost,
	int f_only_self_dual, int f_only_not_self_dual, 
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	int i;
	string fname;
	other::orbiter_kernel_system::file_io Fio;
	
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
	for (i = 0; i < Classify->nb_types; i++) {

		if (f_only_self_dual) {
			fname = "ids_of_self_dual_type_" + std::to_string(i) + ".csv";
		}
		else if (f_only_not_self_dual) {
			fname = "ids_of_not_self_dual_type_" + std::to_string(i) + ".csv";
		}
		else {
			fname = "ids_of_all_type_" + std::to_string(i) + ".csv";
		}
		Int_vec_print(ost,
				Classify->Reps + i * P->Spread_table_with_selection->nb_iso_types_of_spreads,
				P->Spread_table_with_selection->nb_iso_types_of_spreads);
		ost << " & ";
		// ost << Frequency[i] << " & ";

		int *set;
		int *ago;
		int nb, j, dual, a;

		nb = 0;
		set = NEW_int(Classify->Frequency[i]);
		ago = NEW_int(Classify->Frequency[i]);
		for (j = 0; j < Iso->Folding->Reps->count; j++) {
			if (Classify->rep_idx[j] == i) {
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

		string label1, label2;

		label1.assign("ID");
		label2.assign("ago");

		Fio.Csv_file_support->int_vecs_write_csv(
				set, ago, nb, fname, label1, label2);

		other::data_structures::tally C;

		C.init(ago, nb, false, 0);
		
		ost << nb << " & ";
		C.print_bare_tex(ost, true /*f_backwards*/);

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

#if 0
static int packing_types_compare_function(void *a, void *b, void *data)
{
	invariants_packing *inv = (invariants_packing *) data;
	int *A = (int *) a;
	int *B = (int *) b;
	int i;

	for (i = 0; i < inv->P->Spread_table_with_selection->nb_iso_types_of_spreads; i++) {
		if (A[i] > B[i]) {
			return 1;
		}
		if (A[i] < B[i]) {
			return -1;
		}
	}
	return 0;
}
#endif


}}}

