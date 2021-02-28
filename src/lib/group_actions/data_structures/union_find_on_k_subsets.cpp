// union_find_on_k_subsets.cpp
//
// Anton Betten
// February 2, 2010

#include "foundations/foundations.h"
#include "group_actions.h"


using namespace std;


namespace orbiter {
namespace group_actions {

union_find_on_k_subsets::union_find_on_k_subsets()
{
	null();
}

union_find_on_k_subsets::~union_find_on_k_subsets()
{
	freeself();
}

void union_find_on_k_subsets::freeself()
{
	if (Ar) {
		FREE_OBJECT(Ar);
		}
	if (Ar_perm) {
		FREE_OBJECT(Ar_perm);
		}
	if (Ark) {
		FREE_OBJECT(Ark);
		}
	if (Arkr) {
		FREE_OBJECT(Arkr);
		}
	if (gens_perm) {
		FREE_OBJECT(gens_perm);
		}
	if (UF) {
		FREE_OBJECT(UF);
		}
	null();
}

void union_find_on_k_subsets::null()
{
	set = NULL;
	interesting_k_subsets = NULL;
	A_original = NULL;
	Ar = NULL;
	Ar_perm = NULL;
	Ark = NULL;
	Arkr = NULL;
	gens_perm = NULL;
	UF = NULL;
	
}

void union_find_on_k_subsets::init(
	action *A_original, sims *S,
	long int *set, int set_sz, int k,
	long int *interesting_k_subsets,
	int nb_interesting_k_subsets,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	longinteger_object go, K_go;
	int i, j, h, len;
	int *data1;
	int *data2;
	int *Elt1;


	if (f_v) {
		cout << "union_find_on_k_subsets::init k=" << k << endl;
		}
	union_find_on_k_subsets::A_original = A_original;
	union_find_on_k_subsets::set = set;
	union_find_on_k_subsets::set_sz = set_sz;
	union_find_on_k_subsets::k = k;
	union_find_on_k_subsets::interesting_k_subsets
		= interesting_k_subsets;
	union_find_on_k_subsets::nb_interesting_k_subsets
		= nb_interesting_k_subsets;

	Ar = NEW_OBJECT(action);
	Ar_perm = NEW_OBJECT(action);
	Ark = NEW_OBJECT(action);
	Arkr = NEW_OBJECT(action);
	
	
	if (f_v) {
		cout << "union_find_on_k_subsets::init "
				"before induced_action_by_restriction" << endl;
		}
	Ar = A_original->create_induced_action_by_restriction(
			S, set_sz, set, TRUE, 0/*verbose_level*/);
	//action *create_induced_action_by_restriction(
	//		sims *S, int size, int *set, int f_induce,
	//		int verbose_level);
#if 0
	Ar->induced_action_by_restriction(*A_original,
			TRUE, S, set_sz, set, 0/*verbose_level*/);
#endif
	if (f_v) {
		cout << "union_find_on_k_subsets::init "
				"after induced_action_by_restriction" << endl;
		}
	Ar->group_order(go);
	Ar->Kernel->group_order(K_go);
	if (f_v) {
		cout << "union_find_on_k_subsets::init "
				"induced action by restriction: "
				"group order = " << go << endl;
		cout << "union_find_on_k_subsets::init "
				"kernel group order = " << K_go << endl;
		}
	
	if (f_vv) {
		cout << "union_find_on_k_subsets::init "
				"induced action:" << endl;
		//Ar->Sims->print_generators();
		//Ar->Sims->print_generators_as_permutations();
		//Ar->Sims->print_basic_orbits();
	
		longinteger_object go;
		Ar->Sims->group_order(go);
		cout << "union_find_on_k_subsets::init "
				"Ar->Sims go=" << go << endl;

		//cout << "induced action, in the original action:" << endl;
		//AA->Sims->print_generators_as_permutations_override_action(A);
		}	
	
	//cout << "kernel:" << endl;
	//K->print_generators();
	//K->print_generators_as_permutations();

	if (f_v) {
		cout << "union_find_on_k_subsets::init "
				"before init_permutation_group" << endl;
		}

	Ar_perm->init_permutation_group(set_sz, 0/*verbose_level*/);
	if (f_v) {
		cout << "Ar_perm:" << endl;
		Ar_perm->print_info();
		}
	
	if (f_v) {
		cout << "union_find_on_k_subsets::init "
				"before induced_action_on_k_subsets" << endl;
		}
	Ark->induced_action_on_k_subsets(*Ar_perm, k, 0/*verbose_level*/);
	if (f_v) {
		cout << "union_find_on_k_subsets::init Ar_on_k_subsets:" << endl;
		Ark->print_info();
		}

	if (f_v) {
		cout << "union_find_on_k_subsets::init "
				"before induced_action_by_restriction, "
				"creating Arkr" << endl;
		}
	Arkr = Ark->create_induced_action_by_restriction(
			NULL /* sims *S */, nb_interesting_k_subsets, interesting_k_subsets,
			FALSE, 0/*verbose_level*/);
#if 0
	Arkr->induced_action_by_restriction(*Ark, FALSE, NULL, 
		nb_interesting_k_subsets, interesting_k_subsets, 
		0/*verbose_level*/);
#endif
	if (f_v) {
		cout << "union_find_on_k_subsets::init after "
				"induced_action_by_restriction, Arkr "
				"has been created" << endl;
		}


	if (f_v) {
		cout << "union_find_on_k_subsets::init "
				"creating gens_perm" << endl;
		}

	if (Ar->Strong_gens == NULL) {
		cout << "Ar->Strong_gens == NULL" << endl;
		exit(1);
		}

	vector_ge *gens = Ar->Strong_gens->gens;

	len = gens->len;
	gens_perm = NEW_OBJECT(vector_ge);

	gens_perm->init(Ar_perm, verbose_level - 2);
	gens_perm->allocate(len, verbose_level - 2);

	data1 = NEW_int(set_sz);
	data2 = NEW_int(set_sz);
	Elt1 = NEW_int(Ar_perm->elt_size_in_int);

	for (h = 0; h < len; h++) {
		if (FALSE /*f_v*/) {
			cout << "union_find_on_k_subsets::init "
					"generator " << h << " / " << len << ":" << endl;
			}
		for (i = 0; i < set_sz; i++) {
			j = Ar->image_of(gens->ith(h), i);
			data1[i] = j;
			}
		if (FALSE /*f_v*/) {
			cout << "union_find_on_k_subsets::init permutation: ";
			Orbiter->Int_vec.print(cout, data1, set_sz);
			cout << endl;
			}
		Ar_perm->make_element(Elt1, data1, 0 /* verbose_level */);
		Ar_perm->element_move(Elt1,
				gens_perm->ith(h), 0 /* verbose_level */);
		}
	if (f_v) {
		cout << "union_find_on_k_subsets::init "
				"created gens_perm" << endl;
		}

	UF = NEW_OBJECT(union_find);
	UF->init(Arkr, verbose_level);
	if (f_v) {
		cout << "union_find_on_k_subsets::init "
				"after UF->init" << endl;
		}
	UF->add_generators(gens_perm, 0 /* verbose_level */);
	if (f_v) {
		cout << "union_find_on_k_subsets::init "
				"after UF->add_generators" << endl;
		}
	if (f_v) {
		int nb, N;
		double f;
		nb = UF->count_ancestors();
		N = Arkr->degree;
		f = ((double)nb / (double)N) * 100;
		cout << "union_find_on_k_subsets::init number of "
				"ancestors = " << nb << " / " << N << " ("
				<< f << "%)" << endl;
		}
	if (f_v) {
		cout << "union_find_on_k_subsets::init finished" << endl;
		}

	FREE_int(data1);
	FREE_int(data2);
	FREE_int(Elt1);

	
	if (f_v) {
		cout << "union_find_on_k_subsets::init done" << endl;
		}
}


int union_find_on_k_subsets::is_minimal(int rk, int verbose_level)
{
	int rk0;
	
	rk0 = UF->ancestor(rk);
	if (rk0 == rk) {
		return TRUE;
		}
	else {
		return FALSE;
		}
}


}}
