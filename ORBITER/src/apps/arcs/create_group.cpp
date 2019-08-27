// create_group.cpp
// 
// Anton Betten
// October 2, 2016
//
//
// 
//
//

#include "orbiter.h"


using namespace std;


using namespace orbiter;

int main(int argc, const char **argv);
void create_group_arcs8(int q, int verbose_level);

int main(int argc, const char **argv)
{
	int verbose_level = 0;
	int f_q = TRUE;
	int q = 0;
	


	int i;

	cout << argv[0] << endl;
	for (i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-q") == 0) {
			f_q = TRUE;
			q = atoi(argv[++i]);
			cout << "-q " << q << endl;
			}
		}
	if (!f_q) {
		cout << "please use option -q <q>" << endl;
		exit(1);
		}

	if (q == 8) {
		create_group_arcs8(q, verbose_level);
		}

}

void create_group_arcs8(int q, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	
	// created with -poly 11

	//int arc_8_nb_reps = 1;
	int arc_8_size = 10;
	int arc_8_reps[] = {
		0, 1, 2, 3, 28, 38, 43, 55, 64, 69, 
	};
	const char *arc_8_stab_order[] = {
		"1512",
	};
	int arc_8_make_element_size = 10;
	int arc_8_stab_gens[] = {
		4, 0, 0, 0, 1, 0, 0, 0, 5, 2, 
		4, 0, 0, 0, 3, 0, 4, 3, 6, 2, 
		1, 0, 0, 6, 5, 2, 7, 5, 3, 1, 
		7, 0, 0, 2, 6, 4, 5, 3, 4, 1, 
		1, 5, 4, 5, 2, 4, 1, 2, 5, 0, 
		2, 5, 1, 0, 3, 0, 7, 7, 7, 1, 
	};
	//int arc_8_stab_gens_fst[] = { 0};
	int arc_8_stab_gens_len[] = { 6};

	finite_field *F;
	action *A;
	sims *S;
	set_and_stabilizer *SaS;
	vector_ge *nice_gens;

	F = NEW_OBJECT(finite_field);

	//F->init(q, verbose_level);
	F->init_override_polynomial(q, "11", verbose_level);

	cout << "creating linear group" << endl;

	A = NEW_OBJECT(action);

	A->init_linear_group(S,
		F, 3, 
		TRUE /*f_projective*/, FALSE /* f_general*/, FALSE /* f_affine */, 
		TRUE /* f_semilinear */, FALSE /* f_special */, 
		nice_gens,
		verbose_level);
	FREE_OBJECT(nice_gens);
	cout << "creating linear group done" << endl;
		
	SaS = NEW_OBJECT(set_and_stabilizer);


	SaS->init(A, A, verbose_level);

	SaS->init_data(arc_8_reps, arc_8_size, verbose_level);

	SaS->init_stab_from_data(arc_8_stab_gens, 
		arc_8_make_element_size, arc_8_stab_gens_len[0], 
		arc_8_stab_order[0], 
		verbose_level);
	
	cout << "created set and stabilizer " << endl;

	sims *Aut;
	longinteger_object go;
	int i, j, o, cnt, nb_zero, nb_zero_max;
	int *Elt;

	Elt = NEW_int(A->elt_size_in_int);

	Aut = SaS->Strong_gens->create_sims(verbose_level);
	cout << "group elements of order 9:" << endl;
	//Aut->print_all_group_elements();

	Aut->group_order(go);
	cnt = 0;
	nb_zero_max = 0;
	for (i = 0; i < go.as_int(); i++) {
		Aut->element_unrank_int(i, Elt);
		o = A->element_order(Elt);
		if (o == 9) {
#if 0
			cout << "Element " << setw(5) << i << " / " << go.as_int() << endl;
			A->element_print(Elt, cout);
			cout << endl;
#endif
			nb_zero = 0;
			for (j = 0; j < 9; j++) {
				if (Elt[j] == 0) {
					nb_zero++;
					}
				}
			cnt++;
			if (nb_zero > nb_zero_max) {
				nb_zero_max = nb_zero;
				}
			}
		}
	cout << "We found " << cnt << " group elements of order 9. "
			"nb_zero_max = " << nb_zero_max << endl;

	for (i = 0; i < go.as_int(); i++) {
		Aut->element_unrank_int(i, Elt);
		o = A->element_order(Elt);
		if (o == 9) {
			nb_zero = 0;
			for (j = 0; j < 9; j++) {
				if (Elt[j] == 0) {
					nb_zero++;
					}
				}
			if (nb_zero == nb_zero_max) {
				cout << "Element " << setw(5) << i << " / "
						<< go.as_int() << endl;
				A->element_print(Elt, cout);
				A->element_print_as_permutation(Elt, cout);
				cout << endl;
				}
			}
		}

	int Q;
	finite_field *FQ;
	subfield_structure *Sub;

	Q = q * q * q;
	cout << "Q = " << Q << endl;
	
	FQ = NEW_OBJECT(finite_field);
	FQ->init(Q, 0);
	
	if (f_v) {
		cout << "linear_set::init creating subfield structure" << endl;
		}

	Sub = NEW_OBJECT(subfield_structure);

	Sub->init(FQ, F, verbose_level);
	if (f_v) {
		cout << "linear_set::init creating subfield structure done" << endl;
		}
	
	int idx;
	
	idx = 1458;
	Aut->element_unrank_int(idx, Elt);
	cout << "Element " << setw(5) << idx << " / " << go.as_int() << endl;
	A->element_print(Elt, cout);
	A->element_print_as_permutation(Elt, cout);
	cout << endl;
	
	

	sims *PGGL3Q;
	action *AQ;
	int *Elt1;
	int *Elt2;
	int Data1[10];
	int Data2[10];
	vector_ge *gens;
	strong_generators *Strong_gens;
	//vector_ge *nice_gens;
	
	cout << "creating linear group over FQ" << endl;

	AQ = NEW_OBJECT(action);

	AQ->init_linear_group(PGGL3Q,
		FQ, 3, 
		TRUE /*f_projective*/, FALSE /* f_general*/, FALSE /* f_affine */, 
		TRUE /* f_semilinear */, FALSE /* f_special */, 
		nice_gens,
		verbose_level);
	FREE_OBJECT(nice_gens);
	cout << "creating linear group over FQ done" << endl;

	Elt1 = NEW_int(AQ->elt_size_in_int);
	Elt2 = NEW_int(AQ->elt_size_in_int);
	for (i = 0; i < 9; i++) {
		Data1[i] = Sub->FQ_embedding[Elt[i]];
		}
	Data1[9] = 0;
	AQ->make_element(Elt1, Data1, 0 /* verbose_level */);
	cout << "embedded element:" << endl;
	AQ->element_print(Elt1, cout);

	FQ->identity_matrix(Data2, 3);
	Data2[9] = 3;
	AQ->make_element(Elt2, Data2, 0 /* verbose_level */);
	cout << "embedded element:" << endl;
	AQ->element_print(Elt2, cout);

	gens = NEW_OBJECT(vector_ge);
	gens->init(AQ, verbose_level - 2);
	gens->allocate(2, verbose_level - 2);
	AQ->element_move(Elt1, gens->ith(0), 0);
	AQ->element_move(Elt2, gens->ith(1), 0);
	

	longinteger_object target_go;

	target_go.create(27);
	AQ->generators_to_strong_generators(
		TRUE /* f_target_go */, target_go, 
		gens, Strong_gens, 
		verbose_level);
	cout << "created strong generators" << endl;

	schreier *Orb;
	
	cout << "computing orbits on points:" << endl;
	Orb = Strong_gens->orbits_on_points_schreier(AQ, verbose_level);
	cout << "orbits on points:" << endl;
	Orb->print_and_list_orbits_sorted_by_length(cout);
	
}




