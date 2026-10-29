/*
 * data_structures_group_global.cpp
 *
 *  Created on: Oct 24, 2026
 *      Author: betten
 */







#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"

using namespace std;

namespace orbiter {
namespace layer3_group_actions {
namespace data_structures_groups {

data_structures_group_global::data_structures_group_global()
{
	Record_birth();

}

data_structures_group_global::~data_structures_group_global()
{
	Record_death();
}



data_structures_groups::orbit_rep *data_structures_group_global::orbiter_data_file_extract_representative_with_group(
		other::data_structures::orbiter_data_file *Orbiter_data_file,
		actions::action *A,
		//std::string &fname,
	int orbit_idx,
	//long int *&set, int &set_sz, groups::sims *&stab,
	//groups::strong_generators *&Strong_gens,
	int verbose_level)
{
	int f_v = (verbose_level  >= 1);
	int f_vv = (verbose_level  >= 2);

	if (f_v) {
		cout << "data_structures_group_global::orbiter_data_file_extract_representative_with_group " << endl;
	}


	orbit_rep *Orbit_rep;


#if 0
	int level;
	int orbit_at_level;
	int nb_cases;

	long int *rep;

	groups::sims *Stab;
	groups::strong_generators *Strong_gens;

	algebra::ring_theory::longinteger_object *stab_go;
	long int *candidates;
	int nb_candidates;
#endif

	Orbit_rep = NEW_OBJECT(orbit_rep);

	data_structures_groups::group_container *G;
	//int i;




	Orbit_rep->level = Orbiter_data_file->Set_sizes[orbit_idx];
	Orbit_rep->rep = NEW_lint(Orbit_rep->level);
	Lint_vec_copy(Orbiter_data_file->Sets[orbit_idx], Orbit_rep->rep, Orbit_rep->level);
#if 0
	for (i = 0; i < set_sz; i++) {
		set[i] = Orbiter_data_file->Sets[orbit_idx][i];
	}
#endif


	G = NEW_OBJECT(data_structures_groups::group_container);
	G->init(A, verbose_level - 2);
	if (f_vv) {
		cout << "data_structures_group_global::orbiter_data_file_extract_representative_with_group "
				"before G->init_ascii_coding_to_sims" << endl;
	}

	string s;

	s.assign(Orbiter_data_file->Aut_ascii[orbit_idx]);
	G->init_ascii_coding_to_sims(s, verbose_level - 2);
	if (f_vv) {
		cout << "data_structures_group_global::orbiter_data_file_extract_representative_with_group "
				"after G->init_ascii_coding_to_sims" << endl;
	}

	Orbit_rep->Stab = G->S;
	G->S = NULL;
	G->f_has_sims = false;

	algebra::ring_theory::longinteger_object go;

	Orbit_rep->Stab->group_order(go);


	Orbit_rep->Strong_gens = NEW_OBJECT(groups::strong_generators);
	Orbit_rep->Strong_gens->init_from_sims(Orbit_rep->Stab, 0);
	A->f_has_strong_generators = true;

	if (f_vv) {
		cout << "data_structures_group_global::orbiter_data_file_extract_representative_with_group "
				"Group order=" << go << endl;
	}

	FREE_OBJECT(G);
	if (f_vv) {
		cout << "data_structures_group_global::orbiter_data_file_extract_representative_with_group "
				"after FREE_OBJECT  G" << endl;
	}

	if (f_v) {
		cout << "data_structures_group_global::orbiter_data_file_extract_representative_with_group done" << endl;
	}
	return Orbit_rep;
}

data_structures_groups::orbit_rep *data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates(
		other::data_structures::orbiter_data_file *Orbiter_data_file,
		actions::action *A,
		std::string &prefix,
	int level, int orbit_at_level, int level_of_candidates_file,
	//long int *&starter,
	//int &starter_sz,
	//groups::sims *&Stab,
	//groups::strong_generators *&Strong_gens,
	long int *&candidates,
	int &nb_candidates,
	int nb_cases,
	int verbose_level)
// A needs to be the base action
// data_structures_group_global Data_structures_group_global;
// nb_cases is only used for printing purposes
{
	int f_v = (verbose_level >= 1);
	int orbit_at_candidate_level = -1;
	other::orbiter_kernel_system::file_io Fio;


	if (f_v) {
		cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates "
				"prefix=" << prefix << endl;
	}


	orbit_rep *Orbit_rep;

	{
		candidates = NULL;
		//longinteger_object stab_go;

		//string fname1;
		//fname1 = prefix + "_lvl_" + std::to_string(level);

		if (f_v) {
			cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates "
					"before orbiter_data_file_extract_representative_with_group" << endl;
		}


		Orbit_rep = orbiter_data_file_extract_representative_with_group(
					Orbiter_data_file,
						A,
						//fname1,
						orbit_at_level,
						//starter, starter_sz, Stab,
						//Strong_gens,
						verbose_level);

#if 0

		read_set_and_stabilizer(
				Orbiter_data_file,
				A,
				fname1,
			orbit_at_level, starter, starter_sz, Stab,
			Strong_gens,
			nb_cases,
			verbose_level);
#endif

		if (f_v) {
			cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates "
					"after orbiter_data_file_extract_representative_with_group" << endl;
		}



		//Stab->group_order(stab_go);

		if (f_v) {
			cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates "
					"Read starter " << orbit_at_level << " / "
					<< nb_cases << " : ";
			Lint_vec_print(cout, Orbit_rep->rep, Orbit_rep->level);
			cout << endl;
			//cout << "read_orbit_rep_and_candidates_from_files "
			//"Group order=" << stab_go << endl;
		}

		if (level == level_of_candidates_file) {
			orbit_at_candidate_level = orbit_at_level;
		}
		else {
			// level_of_candidates_file < level
			// Now, we need to find out the orbit representative
			// at level_of_candidates_file
			// that matches with the prefix of starter
			// so that we can retrieve it's set of candidates.
			// Once we have the candidates for the prefix, we run it through the
			// test function to find the candidate set of starter as a subset
			// of this set.

			orbit_at_candidate_level = Fio.find_orbit_index_in_data_file(
					prefix,
					level_of_candidates_file, Orbit_rep->rep,
					verbose_level);
		}
		if (f_v) {
			cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates "
					"Found starter, orbit_at_candidate_level="
					<< orbit_at_candidate_level << endl;
		}


		// read the set of candidates from the binary file:

		if (f_v) {
			cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates "
					"before generator_read_candidates_of_orbit" << endl;
		}
		string fname2;

		fname2 = make_fname_candidates(prefix, level_of_candidates_file);

		//fname2 = prefix + "_lvl_" + std::to_string(level_of_candidates_file) + "_candidates.bin";


		if (f_v) {
			cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates "
					"before Fio.poset_classification_read_candidates_of_orbit" << endl;
		}
		Fio.poset_classification_read_candidates_of_orbit(
			fname2, orbit_at_candidate_level,
			candidates, nb_candidates, verbose_level - 1);

		if (f_v) {
			cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates "
					"after Fio.poset_classification_read_candidates_of_orbit" << endl;
		}


		if (candidates == NULL) {
			cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates "
					"could not read the candidates" << endl;
			exit(1);
		}
		if (f_v) {
			cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates "
					"Found " << nb_candidates << " candidates at level "
					<< level_of_candidates_file << endl;
		}
	}
	if (f_v) {
		cout << "data_structures_group_global::orbiter_data_file_get_orbit_rep_and_candidates done" << endl;
	}

	return Orbit_rep;

}

std::string data_structures_group_global::make_fname_candidates(
		std::string &prefix, int level_of_candidates_file)
{
	string fname;

	fname = prefix + "_lvl_" + std::to_string(level_of_candidates_file) + "_candidates.bin";

	return fname;
}



int data_structures_group_global::create_transvection(
		geometry::projective_geometry::projective_space *P,
		actions::action *A,
		int f_semilinear,
		int a, int *v, int *u, int len,
		data_structures_groups::vector_ge *&vec,
		int verbose_level)
// vec is only allocated if the return value is true
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "data_structures_group_global::create_transvection" << endl;
	}
	if (f_v) {
		cout << "data_structures_group_global::create_transvection "
				"v = ";
		Int_vec_print(cout, v, len);
		cout << endl;
		cout << "data_structures_group_global::create_transvection "
				"u = ";
		Int_vec_print(cout, u, len);
		cout << endl;
	}

	if (len != P->Subspaces->n + 1) {
		cout << "data_structures_group_global::create_transvection "
				"len != P->Subspaces->n + 1" << endl;
		exit(1);
	}


	int *z;
	int *Data;
	int i, d, sz;

	d = P->Subspaces->n + 1;

	sz = d * d;
	if (f_semilinear) {
		sz++;
	}

	z = NEW_int(d);
	Data = NEW_int(sz);

	int ret = true;



	if (f_v) {
		cout << "data_structures_group_global::create_transvection "
				"i = " << i << endl;
	}


	P->Subspaces->F->Linear_algebra->make_transvection(
			Data + i * sz, a, v, u,
			len, verbose_level - 2);


	if (P->Subspaces->F->Linear_algebra->matrix_determinant(
			Data, d, 0 /* verbose_level */) == 0) {
		ret = false;
	}


	if (ret) {
		if (f_semilinear) {
			Data[d * d] = 0;
		}




		vec = NEW_OBJECT(data_structures_groups::vector_ge);

		if (f_v) {
			cout << "data_structures_group_global::create_transvection "
					"A = " << endl;
			A->print_info();
		}

		if (f_v) {
			cout << "data_structures_group_global::create_transvection "
					"before vec->init_from_data" << endl;
		}
		vec->init_from_data(
				A, Data,
				1, sz, verbose_level);
		if (f_v) {
			cout << "data_structures_group_global::create_transvection "
					"after vec->init_from_data" << endl;
		}
	}

	FREE_int(Data);
	FREE_int(z);



	if (f_v) {
		cout << "data_structures_group_global::create_transvection done" << endl;
	}
	return ret;
}

void data_structures_group_global::create_all_transvections(
		geometry::projective_geometry::projective_space *P,
		actions::action *A,
		int f_semilinear,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "data_structures_group_global::create_all_transvections" << endl;
	}

	int q;
	int d;
	long int N, big_N;
	long int rk1, rk2, a;

	d = P->Subspaces->n + 1;
	q = P->Subspaces->F->q;

	geometry::other_geometry::geometry_global Geometry_global;

	N = Geometry_global.nb_PG_elements(
			d - 1, q);

	big_N = N * N * (q - 1);

	int *vec_v;
	int *vec_u;

	vec_v = NEW_int(d);
	vec_u = NEW_int(d);



	// First, we count the number of transvections:




	long int cur;
	long int nb_transvections;

	cur = 0;


	for (rk1 = 0; rk1 < N; rk1++) {

		P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
				vec_v, 1 /*stride*/, d, rk1);

		P->Subspaces->F->Projective_space_basic->PG_element_normalize_from_front(
					vec_v, 1, d);


		for (rk2 = 0; rk2 < N; rk2++) {

			P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
					vec_u, 1 /*stride*/, d, rk2);

			P->Subspaces->F->Projective_space_basic->PG_element_normalize_from_front(
						vec_u, 1, d);


			for (a = 1; a < q; a++) {

				data_structures_groups::vector_ge *vec;

				if (create_transvection(
						P,
						A,
						f_semilinear,
						a, vec_v, vec_u, d,
						vec,
						verbose_level - 2)) {

					FREE_OBJECT(vec);

					cur++;
				}



			}
		}
	}

	nb_transvections = cur;




	if (f_v) {
		cout << "data_structures_group_global::create_all_transvections big_N = " << big_N << endl;
		cout << "data_structures_group_global::create_all_transvections nb_transvections = " << nb_transvections << endl;
	}



	// Now we create the transvections themselves
	// and save the data to a csv file:



	data_structures_groups::vector_ge *Vec;

	Vec = NEW_OBJECT(data_structures_groups::vector_ge);

	Vec->init(A, 0 /* verbose_level */);
	Vec->allocate(nb_transvections, 0 /* verbose_level */);




	// save the data to a csv file:


	other::orbiter_kernel_system::file_io Fio;
	std::string fname;

	fname = "PG_" + std::to_string(d - 1) + "_" + std::to_string(q) + "_all_transvections.csv";

	std::string *Col_headings;

	int nb_rows = nb_transvections;
	int nb_cols = 8;

	string *Table;


	Table = new string[nb_rows * nb_cols];

	cur = 0;

	for (rk1 = 0; rk1 < N; rk1++) {

		P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
				vec_v, 1 /*stride*/, d, rk1);

		P->Subspaces->F->Projective_space_basic->PG_element_normalize_from_front(
					vec_v, 1, d);

		for (rk2 = 0; rk2 < N; rk2++) {

			P->Subspaces->F->Projective_space_basic->PG_element_unrank_modified(
					vec_u, 1 /*stride*/, d, rk2);

			P->Subspaces->F->Projective_space_basic->PG_element_normalize_from_front(
						vec_u, 1, d);

			for (a = 1; a < q; a++) {


				data_structures_groups::vector_ge *vec;

				if (create_transvection(
						P,
						A,
						f_semilinear,
						a, vec_v, vec_u, d,
						vec,
						verbose_level - 2)) {

					A->Group_element->element_move(vec->ith(0), Vec->ith(cur), 0 /* verbose_level */);


					Table[cur * nb_cols + 0] = std::to_string(cur);
					Table[cur * nb_cols + 1] = std::to_string(rk1);
					Table[cur * nb_cols + 2] = std::to_string(rk2);
					Table[cur * nb_cols + 3] = std::to_string(a);
					Table[cur * nb_cols + 4] = "\"" + Int_vec_stringify(vec_v, d) + "\"";
					Table[cur * nb_cols + 5] = "\"" + Int_vec_stringify(vec_u, d) + "\"";


					{

						Table[cur * nb_cols + 6] = "\"" + Int_vec_stringify(Vec->ith(cur), d * d) + "\"";
					}
#if 0
					{
					std::string options;
					string s;

					s = PA->A->Group_element->element_stringify(
							Vec->ith(cur), options);


					Table[cur * nb_cols + 7] = "\"" + s + "\"";
					}
#endif

					FREE_OBJECT(vec);


					cur++;
				}




			}
		}
	}


	Col_headings = new string [nb_cols];

	Col_headings[0] = "idx";
	Col_headings[1] = "rk1";
	Col_headings[2] = "rk2";
	Col_headings[3] = "s";
	Col_headings[4] = "v";
	Col_headings[5] = "u";
	Col_headings[6] = "Element";
	Col_headings[7] = "ElementTex";


	Fio.Csv_file_support->write_table_of_strings_with_col_headings(
			fname,
			nb_rows, nb_cols, Table,
			Col_headings,
			verbose_level);

	if (f_v) {
		cout << "data_structures_group_global::create_all_transvections "
				"written file " << fname << " of size "
				<< Fio.file_size(fname) << endl;
	}

	delete [] Col_headings;
	delete [] Table;






	FREE_int(vec_v);
	FREE_int(vec_u);

	FREE_OBJECT(Vec);


	if (f_v) {
		cout << "data_structures_group_global::create_all_transvections done" << endl;
	}
}

int data_structures_group_global::small_generating_set_collineation_group(
		actions::action *A,
		int n, int q, int f_semilinear,
		data_structures_groups::vector_ge *&vec,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "data_structures_group_global::small_generating_set_collineation_group" << endl;
	}
	//data_structures_groups::vector_ge *vec;

	vec = NEW_OBJECT(data_structures_groups::vector_ge);

	if (f_v) {
		cout << "projective_space_activity::perform_activity "
				"A = " << A->label << endl;
	}


	combinatorics::knowledge_base::knowledge_base K;

	int *Data;
	int size_element;
	int nb_generators;

	if (f_v) {
		cout << "data_structures_group_global::small_generating_set_collineation_group "
				"before K.get_small_generating_set_PGL" << endl;
	}
	if (!K.get_small_generating_set_PGL(
			n, q, Data, size_element, nb_generators, verbose_level)) {
		if (f_v) {
			cout << "data_structures_group_global::small_generating_set_collineation_group "
					"I don't have a small generating set for this group" << endl;
		}
		FREE_OBJECT(vec);
		return false;
	}
	if (f_v) {
		cout << "data_structures_group_global::small_generating_set_collineation_group "
				"after K.get_small_generating_set_PGL" << endl;
	}


	if (f_v) {
		cout << "projective_space_activity::perform_activity "
				"before vec->init_from_data" << endl;
	}
	vec->init_from_data(
			A, Data,
			nb_generators, size_element, verbose_level);
	if (f_v) {
		cout << "projective_space_activity::perform_activity "
				"after vec->init_from_data" << endl;
	}


	if (f_v) {
		cout << "data_structures_group_global::small_generating_set_collineation_group done" << endl;
	}
	return true;
}


}}}

