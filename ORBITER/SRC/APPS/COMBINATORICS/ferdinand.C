// ferdinand.C
// 
// Anton Betten
// May 2, 2017
//
//
// 
//
//

#include "orbiter.h"
#include "discreta.h"



typedef class cayley_graph_search cayley_graph_search;

class cayley_graph_search {

public:

	INT level;
	INT group;
	INT subgroup;

	INT ord;
	INT degree;
	INT data_size;

	INT go;
	INT go_subgroup;
	INT nb_involutions;
	INT *f_has_order2;
	INT *f_subgroup;

	action *A;
	finite_field *F;
	INT target_depth;

	INT *Elt1;
	vector_ge *gens;
	vector_ge *gens_subgroup;
	longinteger_object target_go, target_go_subgroup;
	strong_generators *Strong_gens;
	strong_generators *Strong_gens_subgroup;

	sims *S;
	sims *S_subgroup;

	INT *Table;
	INT *generators;
	INT nb_generators;

	BYTE fname_base[1000];
	BYTE prefix[000];
	BYTE fname[1000];
	BYTE fname_graphs[1000];

	strong_generators *Aut_gens;
	longinteger_object Aut_order;
	action *Aut;
	action *A2;
	generator *gen;


	void init(INT level, INT group, INT subgroup, INT verbose_level);
	void init_group(INT verbose_level);
	void init_group2(INT verbose_level);
	void init_group_level_3(INT verbose_level);
	void init_group_level_4(INT verbose_level);
	void init_group_level_5(INT verbose_level);
	INT incremental_check_func(INT len, INT *S, INT verbose_level);
	void classify_subsets(INT verbose_level);
	void write_file(INT verbose_level);

};




void ferdinand(INT level, INT group, INT subgroup, INT verbose_level);
INT ferdinand_incremental_check_func(INT len, INT *S, void *data, INT verbose_level);


int main(int argc, char **argv)
{
	INT verbose_level = 0;
	INT f_level = FALSE;
	INT level = 0;
	INT f_group = FALSE;
	INT group = 0;
	INT f_subgroup = FALSE;
	INT subgroup = 0;
	INT i;
	
	
	for (i = 1; i < argc - 1; i++) {
		if (strcmp(argv[i], "-v") == 0) {
			verbose_level = atoi(argv[++i]);
			cout << "-v " << verbose_level << endl;
			}
		else if (strcmp(argv[i], "-level") == 0) {
			f_level = TRUE;
			level = atoi(argv[++i]);
			cout << "-level " << level << endl;
			}
		else if (strcmp(argv[i], "-group") == 0) {
			f_group = TRUE;
			group = atoi(argv[++i]);
			cout << "-group " << group << endl;
			}
		else if (strcmp(argv[i], "-subgroup") == 0) {
			f_subgroup = TRUE;
			subgroup = atoi(argv[++i]);
			cout << "-subgroup " << subgroup << endl;
			}
		}
	if (!f_level) {
		cout << "please use option -level <level>" << endl;
		exit(1);
		}
	if (!f_group) {
		cout << "please use option -group <group>" << endl;
		exit(1);
		}
	ferdinand(level, group, subgroup, verbose_level);
	cout << "end" << endl;
}




void ferdinand(INT level, INT group, INT subgroup, INT verbose_level)
{

	cayley_graph_search *Cayley;

	Cayley = new cayley_graph_search;



	cout << "ferdinand level = " << level << " group = " << group << " subgroup = " << subgroup << endl;

	cout << "before init" << endl;
	Cayley->init(level, group, subgroup, verbose_level);
	cout << "after init" << endl;

		
	cout << "before classify_subsets" << endl;
	Cayley->classify_subsets(verbose_level);
	cout << "after classify_subsets" << endl;


	cout << "before write_file" << endl;
	Cayley->write_file(verbose_level);
	cout << "after write_file" << endl;





}


INT ferdinand_incremental_check_func(INT len, INT *S, void *data, INT verbose_level)
{
	cayley_graph_search *Cayley = (cayley_graph_search *) data;

	return Cayley->incremental_check_func(len, S, verbose_level);

}


// ####################################################################################
// cayley_graph_search.C:
// ####################################################################################


void cayley_graph_search::init(INT level, INT group, INT subgroup, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	
	if (f_v) {
		cout << "cayley_graph_search::init" << endl;
		}
	cayley_graph_search::level = level;
	cayley_graph_search::group = group;
	cayley_graph_search::subgroup = subgroup;

	Strong_gens_subgroup = NULL;

	degree = 0;
	data_size = 0;

	if (f_v) {
		cout << "cayley_graph_search::init before init_group" << endl;
		}
	init_group(verbose_level);
	if (f_v) {
		cout << "cayley_graph_search::init after init_group" << endl;
		}


	if (f_v) {
		cout << "cayley_graph_search::init before init_group2" << endl;
		}
	init_group2(verbose_level);
	if (f_v) {
		cout << "cayley_graph_search::init after init_group2" << endl;
		}



	if (f_v) {
		cout << "cayley_graph_search::init done" << endl;
		}


}

void cayley_graph_search::init_group(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cayley_graph_search::init_group" << endl;
		}
	if (level == 3) {
		init_group_level_3(verbose_level);
		}
	else if (level == 4) {
		init_group_level_4(verbose_level);
		}
	else if (level == 5) {
		init_group_level_5(verbose_level);
		}
	else {
		cout << "cayley_graph_search::init illegal level" << endl;
		cout << "level = " << level << endl;
		exit(1);
		}
	if (f_v) {
		cout << "cayley_graph_search::init_group done" << endl;
		}

}

void cayley_graph_search::init_group2(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, j;

	if (f_v) {
		cout << "cayley_graph_search::init_group2" << endl;
		}
	S = Strong_gens->create_sims(0 /* verbose_level */);
	S->print_all_group_elements();

	if (Strong_gens_subgroup == NULL) {
		cout << "Strong_gens_subgroup = NULL" << endl;
		exit(1);
		}

	S_subgroup = Strong_gens_subgroup->create_sims(0 /* verbose_level */);


	f_has_order2 = NEW_INT(go);
	f_subgroup = NEW_INT(go);

	INT_vec_zero(f_subgroup, go);

	if (level == 4) {
		if (group == 2 || group == 3 || group == 5) {
			for (i = 0; i < go_subgroup; i++) {
				S_subgroup->element_unrank_INT(i, Elt1);
				cout << "Element " << setw(5) << i << " / " << go_subgroup << ":" << endl;
				A->element_print(Elt1, cout);
				j = S->element_rank_INT(Elt1);
				f_subgroup[j] = TRUE;
				}
			}
		else if (group == 4) {
			for (i = 0; i < go; i++) {
				S->element_unrank_INT(i, Elt1);
				cout << "Element " << setw(5) << i << " / " << go << ":" << endl;
				A->element_print(Elt1, cout);
				cout << endl;
				if (F->is_identity_matrix(Elt1, 4)) {
					f_subgroup[i] = TRUE;
					}
				else {
					f_subgroup[i] = FALSE;
					}
				}
			}
		}
	else if (level == 5) {
		for (i = 0; i < go_subgroup; i++) {
			S_subgroup->element_unrank_INT(i, Elt1);
			cout << "Element " << setw(5) << i << " / " << go_subgroup << ":" << endl;
			A->element_print(Elt1, cout);
			j = S->element_rank_INT(Elt1);
			f_subgroup[j] = TRUE;
			}
		}


	nb_involutions = 0;	
	for (i = 0; i < go; i++) {
		S->element_unrank_INT(i, Elt1);
		cout << "Element " << setw(5) << i << " / " << go << ":" << endl;
		A->element_print(Elt1, cout);
		cout << endl;
		ord = A->element_order(Elt1);
		if (ord == 2) {
			f_has_order2[i] = TRUE;
			nb_involutions++;
			}
		else {
			f_has_order2[i] = FALSE;
			}
		}

	cout << "We found " << nb_involutions << " involutions" << endl;

	
	

#if 1
	
	nb_generators = Strong_gens->gens->len;
	generators = NEW_INT(nb_generators);
	for (i = 0; i < nb_generators; i++) {
		generators[i] = S->element_rank_INT(Strong_gens->gens->ith(i));
		}
	
	S->create_group_table(Table, go, verbose_level);



	sprintf(fname_base, "Ferdinand%ld_%ld", level, group);
	Aut = create_automorphism_group_from_group_table(fname_base, 
		Table, go, generators, nb_generators, 
		Aut_gens, 
		verbose_level);
		// ACTION/action_global.C

	Aut_gens->group_order(Aut_order);
#endif


#if 1

	A2 = new action;
	A2->induced_action_by_right_multiplication(FALSE /* f_basis */, S, 
		S /* Base_group */, FALSE /* f_ownership */, verbose_level);
#endif


	if (f_v) {
		cout << "cayley_graph_search::init_group2 done" << endl;
		}
}

void cayley_graph_search::init_group_level_3(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i;
	
	if (f_v) {
		cout << "cayley_graph_search::init_group_level_3" << endl;
		}
	target_depth = 5;
	go = 16;
	degree = 6;
	data_size = degree;


	A = new action;
	A->init_permutation_group(degree, verbose_level);


	Elt1 = NEW_INT(A->elt_size_in_INT);


	gens = new vector_ge;
	gens->init(A);

	if (group == 1) {
		INT data[] = {
			1,2,3,0,4,5, // (0,1,2,3)
			2,1,0,3,4,5, // (0,2)
			0,1,2,3,5,4, // (4,5)
			};

		gens->allocate(3);

		for (i = 0; i < 3; i++) {
			A->make_element(Elt1, data + i * data_size, 0 /*verbose_level*/);
			A->element_move(Elt1, gens->ith(i), 0);
			}
		}
	else {
		cout << "illegal group" << endl;
		exit(1);
		}
	go_subgroup = go / 2;
	target_go.create(go);
	generators_to_strong_generators(A, 
		TRUE /* f_target_go */, target_go, 
		gens, Strong_gens, 
		verbose_level);
	Strong_gens->print_generators_ost(cout);
	if (f_v) {
		cout << "cayley_graph_search::init_group_level_3 done" << endl;
		}
}

void cayley_graph_search::init_group_level_4(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cayley_graph_search::init_group_level_4" << endl;
		}
	target_depth = 32;
	go = 32;

	INT i, j;

	A = new action;


	if (group == 2) {
		degree = 10;
		data_size = degree;
		}
	else if (group == 3) {
		degree = 8;
		data_size = degree;
		}
	else if (group == 4) {
		degree = 0;
		data_size = 20;
		}
	else if (group == 5) {
		degree = 12;
		data_size = degree;
		}


	if (degree) {
		A->init_permutation_group(degree, verbose_level);
		}
	else if (group == 4) {
		INT q = 2;

		F = new finite_field;
		F->init(q, 0);
		A->init_affine_group(4, F, 
			FALSE /* f_semilinear */, 
			TRUE /* f_basis */, verbose_level);
		}
	else {
		cout << "group " << group << " not yet implemented" << endl;
		exit(1);
		}

	go_subgroup = go / 2;



	Elt1 = NEW_INT(A->elt_size_in_INT);
	gens = new vector_ge;
	gens_subgroup = new vector_ge;
	gens->init(A);
	gens_subgroup->init(A);

	if (group == 2) {
		INT data[] = { // C_4 x C_2 x C_2 x C_2
			1,2,3,0,4,5,6,7,8,9, // (0,1,2,3)
			0,1,2,3,5,4,6,7,8,9, // (4,5)
			0,1,2,3,4,5,7,6,8,9, // (6,7)
			0,1,2,3,4,5,6,7,9,8, // (8,9)
			};
		INT data_subgroup[] = {
			2,3,0,1,4,5,6,7,8,9, // (0,2)(1,3)
			0,1,2,3,5,4,6,7,8,9, // (4,5)
			0,1,2,3,4,5,7,6,8,9, // (6,7)
			0,1,2,3,4,5,6,7,9,8, // (8,9)
			};

		gens->allocate(4);

		for (i = 0; i < 4; i++) {
			A->make_element(Elt1, data + i * data_size, 0 /*verbose_level*/);
			A->element_move(Elt1, gens->ith(i), 0);
			}
		gens_subgroup->allocate(4);

		for (i = 0; i < 4; i++) {
			A->make_element(Elt1, data_subgroup + i * data_size, 0 /*verbose_level*/);
			A->element_move(Elt1, gens_subgroup->ith(i), 0);
			}
		}
	else if (group == 3) {
		INT data[] = { // D_8 x C_2 x C_2
			1,2,3,0,4,5,6,7, // (0,1,2,3)
			2,1,0,3,4,5,6,7, // (0,2)
			0,1,2,3,5,4,6,7, // (4,5)
			0,1,2,3,4,5,7,6, // (6,7)
			};
		INT data_subgroup1[] = {
			2,1,0,3,4,5,6,7, // (0,2)
			0,3,2,1,4,5,6,7, // (1,3)
			0,1,2,3,5,4,6,7, // (4,5)
			0,1,2,3,4,5,7,6, // (6,7)
			};
		INT data_subgroup2[] = {
			1,0,3,2,4,5,6,7, // (0,1)(2,3)
			3,2,1,0,4,5,6,7, // (0,3)(1,2)
			0,1,2,3,5,4,6,7, // (4,5)
			0,1,2,3,4,5,7,6, // (6,7)
			};

		gens->allocate(4);

		for (i = 0; i < 4; i++) {
			A->make_element(Elt1, data + i * data_size, 0 /*verbose_level*/);
			A->element_move(Elt1, gens->ith(i), 0);
			}
		gens_subgroup->allocate(4);


		if (subgroup == 1) {
			for (i = 0; i < 4; i++) {
				A->make_element(Elt1, data_subgroup1 + i * data_size, 0 /*verbose_level*/);
				A->element_move(Elt1, gens_subgroup->ith(i), 0);
				}
			}
		else if (subgroup == 2) {
			for (i = 0; i < 4; i++) {
				A->make_element(Elt1, data_subgroup2 + i * data_size, 0 /*verbose_level*/);
				A->element_move(Elt1, gens_subgroup->ith(i), 0);
				}
			}
		else {
			cout << "unkown subgroup" << endl;
			exit(1);
			}
		}
	else if (group == 4) {
		INT data[] = {
			1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1, 1,0,0,0,
			1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1, 0,1,0,0,
			1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1, 0,0,1,0,
			1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1, 0,0,0,1,
			1,0,0,0,0,1,0,0,1,0,1,0,0,1,0,1, 0,0,0,0
			};

		INT data_subgroup[] = {
			1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1, 1,0,0,0,
			1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1, 0,1,0,0,
			1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1, 0,0,1,0,
			1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1, 0,0,0,1,
			};

		gens->allocate(5);
		for (i = 0; i < 5; i++) {
			A->make_element(Elt1, data + i * data_size, 0 /*verbose_level*/);
			A->element_move(Elt1, gens->ith(i), 0);
			}

		gens_subgroup->allocate(4);
		for (i = 0; i < 4; i++) {
			A->make_element(Elt1, data_subgroup + i * data_size, 0 /*verbose_level*/);
			A->element_move(Elt1, gens_subgroup->ith(i), 0);
			}


		}
	else if (group == 5) {
		const BYTE *data_str[] = { 
			"(1,2)(5,8,6,7)(9,12,10,11)", 
			"(1,4)(2,3)(5,12)(6,11)(7,10)(8,9)",
			"(5,10)(6,9)(7,12)(8,11)", 
			"(1,2)(3,4)(5,6)(7,8)(9,10)(11,12)", 
			"(5,6)(7,8)(9,10)(11,12)"
			};

		gens->allocate(5);

		for (i = 0; i < 5; i++) {
			INT *perm;
			INT degree;
			
			scan_permutation_from_string(data_str[i], perm, degree, 0 /* verbose_level */);
			cout << "degree=" << degree << endl;
			for (j = 0; j < degree; j++) {
				cout << perm[j] << " ";
				}
			cout << " : ";

			for (j = 1; j < degree; j++) {
				perm[j - 1] = perm[j] - 1;
				}
			for (j = 0; j < degree - 1; j++) {
				cout << perm[j] << " ";
				}
			cout << endl;

			A->make_element(Elt1, perm, 0 /*verbose_level*/);
			A->element_move(Elt1, gens->ith(i), 0);
			}
		const BYTE *data_subgroup_str[] = { 
			"(1,4)(2,3)(5,8)(6,7)(9,12)(10,11)", 
			"(1,4)(2,3)(5,12)(6,11)(7,10)(8,9)",
			"(1,2)(3,4)(12)", 
			"(5,6)(7,8)(9,10)(11,12)"
			};

		gens_subgroup->allocate(4);

		for (i = 0; i < 4; i++) {
			INT *perm;
			INT degree;
			
			scan_permutation_from_string(data_subgroup_str[i], perm, degree, 0 /* verbose_level */);
			for (j = 0; j < degree; j++) {
				cout << perm[j] << " ";
				}
			cout << " : ";

			for (j = 1; j < degree; j++) {
				perm[j - 1] = perm[j] - 1;
				}
			for (j = 0; j < degree - 1; j++) {
				cout << perm[j] << " ";
				}
			cout << endl;


			A->make_element(Elt1, perm, 0 /*verbose_level*/);
			A->element_move(Elt1, gens_subgroup->ith(i), 0);
			}
		}
	else {
		cout << "illegal group" << endl;
		exit(1);
		}
	target_go.create(go);
	target_go_subgroup.create(go_subgroup);

	cout << "creating generators for the group:" << endl;
	generators_to_strong_generators(A, 
		TRUE /* f_target_go */, target_go, 
		gens, Strong_gens, 
		verbose_level);

	cout << "creating generators for the subgroup:" << endl;
	generators_to_strong_generators(A, 
		TRUE /* f_target_go */, target_go_subgroup, 
		gens_subgroup, Strong_gens_subgroup, 
		verbose_level);

	if (f_v) {
		cout << "cayley_graph_search::init_group_level_4 done" << endl;
		}

}

void cayley_graph_search::init_group_level_5(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cayley_graph_search::init_group_level_5" << endl;
		}

	target_depth = 15;
	go = 64;

	INT i;

	A = new action;


	if (group == 1) {
		degree = 0;
		data_size = 30;
		}
	else {
		cout << "unknown type of group" << endl;
		}


	if (group == 1) {
		INT q = 2;

		F = new finite_field;
		F->init(q, 0);
		A->init_affine_group(5, F, 
			FALSE /* f_semilinear */, 
			TRUE /* f_basis */, verbose_level);
		}
	else {
		cout << "group " << group << " not yet implemented" << endl;
		exit(1);
		}

	go_subgroup = go / 2;



	Elt1 = NEW_INT(A->elt_size_in_INT);
	gens = new vector_ge;
	gens_subgroup = new vector_ge;
	gens->init(A);
	gens_subgroup->init(A);


	if (group == 1) {
		INT data[] = {
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1, 1,0,0,0,0,
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1, 0,1,0,0,0,
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1, 0,0,1,0,0,
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1, 0,0,0,1,0,
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1, 0,0,0,0,1,
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,1, 0,0,0,0,0,
			};

		INT data_subgroup[] = {
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1, 1,0,0,0,0,
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1, 0,1,0,0,0,
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1, 0,0,1,0,0,
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1, 0,0,0,1,0,
			1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1, 0,0,0,0,1,
			};

		gens->allocate(6);
		for (i = 0; i < 6; i++) {
			A->make_element(Elt1, data + i * data_size, 0 /*verbose_level*/);
			A->element_move(Elt1, gens->ith(i), 0);
			}

		gens_subgroup->allocate(5);
		for (i = 0; i < 5; i++) {
			A->make_element(Elt1, data_subgroup + i * data_size, 0 /*verbose_level*/);
			A->element_move(Elt1, gens_subgroup->ith(i), 0);
			}


		}
	else {
		cout << "illegal group" << endl;
		exit(1);
		}
	target_go.create(go);
	target_go_subgroup.create(go_subgroup);

	cout << "creating generators for the group:" << endl;
	generators_to_strong_generators(A, 
		FALSE /* f_target_go */, target_go, 
		gens, Strong_gens, 
		verbose_level);

	longinteger_object go1;
	Strong_gens->group_order(go1);
	cout << "go1=" << go1 << endl;
	//exit(1);


	cout << "creating generators for the subgroup:" << endl;
	generators_to_strong_generators(A, 
		FALSE /* f_target_go */, target_go_subgroup, 
		gens_subgroup, Strong_gens_subgroup, 
		verbose_level);

	if (f_v) {
		cout << "cayley_graph_search::init_group_level_5 done" << endl;
		}

}

INT cayley_graph_search::incremental_check_func(INT len, INT *S, INT verbose_level)
{
	INT f_OK = TRUE;
	//verbose_level = 1;
	INT f_v = (verbose_level >= 1);
	INT a;
	
	if (f_v) {
		cout << "checking set ";
		print_set(cout, len, S);
		cout << " (incrementally)";
		}
	if (len) {
		a = S[len - 1];
		if (a == 0) {
			f_OK = FALSE;
			}
		if (f_has_order2[a] == FALSE) {
			f_OK = FALSE;
			}
		}
	if (f_OK) {
		if (f_v) {
			cout << "OK" << endl;
			}
		return TRUE;
		}
	else {
		return FALSE;
		}
}



void cayley_graph_search::classify_subsets(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cayley_graph_search::classify_subsets" << endl;
		}
	




	INT f_W = TRUE;
	INT f_w = TRUE;

	sprintf(prefix, "Ferdinand%ld_%ld", level, group);

	cout << "classifying subsets:" << endl;

	compute_orbits_on_subsets(gen, 
		target_depth,
		prefix, 
		f_W, f_w,
		Aut, Aut, 
		//A, A2, 
		Aut_gens, 
		//Strong_gens, 
		NULL /* ferdinand3_early_test_func */,
		NULL /* void *early_test_func_data */, 
		ferdinand_incremental_check_func /* INT (*candidate_incremental_check_func)(INT len, INT *S, void *data, INT verbose_level)*/, 
		this /* void *candidate_incremental_check_data */, 
		verbose_level);



#if 0
	sprintf(fname, "Ferdinand%ld_%ld", level, group);
	gen->draw_poset(fname, target_depth, 0 /* data */, TRUE /* f_embedded */, FALSE /* f_sideways */, 0 /* verbose_level */);
#endif


	if (f_v) {
		cout << "cayley_graph_search::classify_subsets done" << endl;
		}

}



void cayley_graph_search::write_file(INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "cayley_graph_search::write_file" << endl;
		}
	
	INT nb_orbits;
	INT sz;
	INT i, j;
	INT cnt = 0;

	for (sz = 1; sz <= target_depth; sz++) {

		sprintf(fname_graphs, "ferdinand%ld_%ld_subgroup_%ld_graphs_sz_%ld.txt", level, group, subgroup, sz);
		{
		ofstream fp(fname_graphs);

		for (i = 0; i < go; i++) {
			S->element_unrank_INT(i, Elt1);
			cout << "Element " << setw(5) << i << " / " << go << ":" << endl;
			A->element_print(Elt1, cout);
			cout << endl;

			fp << "[";
			A2->element_print_as_permutation(Elt1, fp);
			fp << "],";
			}
		fp << endl;
		for (i = 0; i < go; i++) {
			S->element_unrank_INT(i, Elt1);
			cout << "Element " << setw(5) << i << " / " << go << ":" << endl;
			A->element_print(Elt1, cout);
			cout << endl;

			if (f_subgroup[i]) {
				fp << "[";
				A2->element_print_as_permutation(Elt1, fp);
				fp << "],";
				}
			}
		fp << endl;

		INT n;
		INT *Adj;
		INT *image_set;

		Adj = NEW_INT(go * sz);
		image_set = NEW_INT(sz);

		nb_orbits = gen->nb_orbits_at_level(sz);
		cout << "We found " << nb_orbits << " orbits on " << sz << "-subsets" << endl;
		//fp << "[";
#if 0
		for (n = 0; n < nb_orbits; n++) {

			set_and_stabilizer *SaS;

			SaS = gen->get_set_and_stabilizer(sz, n, 0 /*verbose_level*/);

			cout << n << " / " << nb_orbits << " : ";
			SaS->print_set_tex(cout);
			cout << "  # " << cnt << endl;
			cout << endl;
			}
#endif
		for (n = 0; n < nb_orbits; n++) {

			set_and_stabilizer *SaS;

			SaS = gen->get_set_and_stabilizer(sz, n, 0 /*verbose_level*/);

			if ((n % 1000) == 0) {
				cout << n << " / " << nb_orbits << " : ";
				SaS->print_set_tex(cout);
				cout << "  # " << cnt << endl;
				cout << endl;
				}


			for (i = 0; i < go; i++) {
				S->element_unrank_INT(i, Elt1);

#if 0
				cout << "Element " << setw(5) << i << " / " << go << ":" << endl;

				cout << "set: ";
				INT_vec_print(cout, SaS->data, target_depth);
				cout << endl;
#endif

				A2->map_a_set_and_reorder(SaS->data, image_set, sz, Elt1, 0 /* verbose_level */);

#if 0
				cout << "image_set: ";
				INT_vec_print(cout, image_set, target_depth);
				cout << endl;
#endif
			
				INT_vec_copy(image_set, Adj + i * sz, sz);
				}
			//cout << "The adjacency sets are:" << endl;
			//print_integer_matrix_with_standard_labels(cout, Adj, go, sz, FALSE /* f_tex */);
			fp << "{";
			for (i = 0; i < go; i++) {
				fp << i << ":[";
				for (j = 0; j < sz; j++) {
					fp << Adj[i * sz + j];
					if (j < sz - 1) {
						fp << ",";
						}
					}
				fp << "]";
#if 1
				if (i < go - 1) {
						fp << ",";
					}
#endif
				}
			fp << "}";

			//action *create_automorphism_group_of_graph(INT *Adj, INT n, INT verbose_level);
#if 0
			if (n < nb_orbits - 1) {
				fp << ", ";
				}
#endif
			fp << "  # " << cnt << endl;
			cnt++;
	
			delete SaS;
			}
		//fp << "];" << endl;

		delete Adj;
		delete image_set;
		} // end of fp

	cout << "written file " << fname_graphs << " of size " << file_size(fname_graphs) << endl;
	} // next sz


#if 0
	INT set[5] = {6,7,8,10,15};
	INT canonical_set[5];
	INT *Elt;
	INT orb;

	Elt = NEW_INT(A->elt_size_in_INT);

	orb = gen->trace_set(set, 5, 5, 
		canonical_set, Elt, 
		0 /*verbose_level */);
	cout << "canonical set : ";
	INT_vec_print(cout, canonical_set, 5);
	cout << endl;
	cout << "orb=" << orb << endl;
	cout << "transporter : ";
	Aut->element_print(Elt, cout);
	//A->element_print(Elt, cout);
	cout << endl;
#endif

	if (f_v) {
		cout << "cayley_graph_search::write_file done" << endl;
		}
}



