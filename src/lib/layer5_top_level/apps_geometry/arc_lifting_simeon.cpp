/*
 * arc_lifting_simeon.cpp
 *
 *  Created on: Jan 5, 2019
 *      Author: betten
 */


#include "orbiter.h"

using namespace std;

namespace orbiter {
namespace layer5_applications {
namespace apps_geometry {

static void early_test_func_for_arc_callback(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level);




arc_lifting_simeon::arc_lifting_simeon()
{
	Record_birth();
	verbose_level = 0;
	q = 0;
	d = 0; // largest number of points per line
	n = 0; // projective dimension
	k = 0; // size of the arc
	F = NULL;
	f_projective = true;
	f_general = false;
	f_affine = false;
	f_semilinear = false;
	f_special = false;
	//S = NULL;
	A = NULL;
	//longinteger_object go;
	Elt = NULL;
	v = NULL;
	Sch = NULL;
	Poset = NULL;
	Gen = NULL;
	P = NULL;

	A2 = NULL; // action on the lines
	A3 = NULL; // action on lines restricted to filtered_lines

}

arc_lifting_simeon::~arc_lifting_simeon()
{
	Record_death();

}

void arc_lifting_simeon::init(
		int q, int d, int n, int k,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "simeon::init" << endl;
	}
	arc_lifting_simeon::q = q;
	arc_lifting_simeon::d = d;
	arc_lifting_simeon::n = n;
	arc_lifting_simeon::k = k;

	f_projective = true;
	f_general = false;
	f_affine = false;
	f_semilinear = false;
	f_special = false;

	v = NEW_int(n + 1);

	F = NEW_OBJECT(algebra::field_theory::finite_field);
	F->finite_field_init_small_order(q,
			false /* f_without_tables */,
			false /* f_compute_related_fields */,
			0);

	data_structures_groups::vector_ge *nice_gens;

	A = NEW_OBJECT(actions::action);
	A->Known_groups->init_linear_group(//S,
		F, n + 1,
		f_projective, f_general, f_affine,
		f_semilinear, f_special,
		nice_gens,
		verbose_level);
	FREE_OBJECT(nice_gens);
	A->group_order(go);
	cout << "created a group of order " << go << endl;

	Elt = NEW_int(A->elt_size_in_int);

#if 0
	for (i = 0; i < go.as_int(); i++) {
		S->element_unrank_int(i, Elt, 0 /* verbose_level */);
		cout << "element " << i << " / " << go << ":" << endl;
		A->element_print_quick(Elt, cout);
		}
#endif

	for (i = 0; i < A->degree; i++) {
		F->Projective_space_basic->PG_element_unrank_modified(
				v, 1, n + 1, i);
		cout << "point " << i << " / " << A->degree << " is ";
		Int_vec_print(cout, v, d);
		cout << endl;
	}

	cout << "generating set: " << endl;
	A->Strong_gens->print_generators(cout, verbose_level - 1);

	int print_interval = 10000;

	Sch = A->Strong_gens->compute_all_point_orbits_schreier(A, print_interval, verbose_level);

	cout << "We have " << Sch->nb_orbits << " orbits on points" << endl;

	Sch->print_and_list_orbits(cout);

	P = NEW_OBJECT(geometry::projective_geometry::projective_space);

	P->projective_space_init(n /* n */,
		F /* finite_field *F*/,
		true /* f_init_incidence_structure */,
		0 /* verbose_level */);

	P->Subspaces->init_incidence_structure(0 /*verbose_level*/);

	poset_classification::poset_with_group_action *Poset;
	poset_classification::poset_classification_control *Control;

	Control = NEW_OBJECT(poset_classification::poset_classification_control);
	Poset = NEW_OBJECT(poset_classification::poset_with_group_action);
	Poset->init_subset_lattice(A, A,
			A->Strong_gens,
			verbose_level);

	if (f_v) {
		cout << "before "
				"Poset->add_testing_without_group" << endl;
	}
	Poset->add_testing_without_group(
			early_test_func_for_arc_callback,
			this /* void *data */,
			verbose_level);


	Gen = NEW_OBJECT(poset_classification::poset_classification);

	Gen->compute_orbits_on_subsets(
		k /* target_depth */,
		//"" /* const char *prefix */,
		//false /* f_W */, false /* f_w */,
		Control,
		Poset,
		verbose_level);


	Gen->print_orbit_numbers(k);

	if (f_v) {
		cout << "simeon::init done" << endl;
	}

}

void arc_lifting_simeon::early_test_func(
		long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int j;
	int f_OK;

	if (f_v) {
		cout << "arc_lifting_simeon::early_test_func checking set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
		cout << "candidate set of size "
				<< nb_candidates << ":" << endl;
		Lint_vec_print(cout, candidates, nb_candidates);
		cout << endl;
		}

	int *type_collected;

	type_collected = NEW_int(len + 2);

	if (len == 0) {
		Lint_vec_copy(candidates, good_candidates, nb_candidates);
		nb_good_candidates = nb_candidates;
		}
	else {
		nb_good_candidates = 0;

		if (f_vv) {
			cout << "arc_lifting_simeon::early_test_func "
					"before testing" << endl;
			}
		for (j = 0; j < nb_candidates; j++) {


			if (f_vv) {
				cout << "arc_lifting_simeon::early_test_func "
						"testing " << j << " / "
						<< nb_candidates << endl;
				}

			int i;

			f_OK = true;

			S[len] = candidates[j];

			//cout << "test_function_for_arc" << endl;
			P->Subspaces->line_intersection_type_collected(
				S /*int *set */,
				len + 1 /* int set_size */,
				type_collected,
				0 /*verbose_level */);
			for (i = d + 1; i <= len + 1; i++) {
				if (type_collected[i]) {
					//cout << "test_function_for_arc fail" << endl;
					f_OK = false;
					}
				}

			if (f_OK) {
				good_candidates[nb_good_candidates++] =
						candidates[j];
				}
			} // next j
		} // else
	FREE_int(type_collected);
}


void arc_lifting_simeon::do_covering_problem(
		data_structures_groups::set_and_stabilizer *SaS)
{
	int *type;
	long int *original_arc;
	int original_arc_sz;
	int *bisecants;
	int *c2_points;
	long int *external_lines;
	int nb_external_lines;
	int h, i, j, pi, pj;
	int nb_bisecants, nb_c2points, bi, bj, a, idx, u, pt;
	combinatorics::other_combinatorics::combinatorics_domain Combi;
	other::data_structures::sorting Sorting;

	original_arc = SaS->data;
	original_arc_sz = SaS->sz;

	nb_bisecants = Combi.int_n_choose_k(original_arc_sz, 2);
	nb_c2points = nb_bisecants * nb_bisecants;
	type = NEW_int(P->Subspaces->N_lines);
	external_lines = NEW_lint(P->Subspaces->N_lines);
	nb_external_lines = 0;
	P->Subspaces->line_intersection_type(original_arc,
			original_arc_sz, type, 0 /*verbose_level*/);

	for (i = 0; i < P->Subspaces->N_lines; i++) {
		if (type[i] == 0) {
			external_lines[nb_external_lines++] = i;
			}
		}
	cout << "We found " << nb_external_lines
			<< " external lines, they are: ";
	Lint_vec_print(cout, external_lines, nb_external_lines);
	cout << endl;

	cout << "compute bisecants and c2 points:" << endl;


	bisecants = NEW_int(nb_bisecants);

	h = 0;
	for (i = 0; i < original_arc_sz; i++) {
		pi = original_arc[i];
		for (j = i + 1; j < original_arc_sz; j++) {
			pj = original_arc[j];
			bisecants[h++] = P->line_through_two_points(pi, pj);
			}
		}
	if (h != nb_bisecants) {
		cout << "h != nb_bisecants" << endl;
		exit(1);
		}
	cout << "We found " << nb_bisecants << " bisecants : ";
	Int_vec_print(cout, bisecants, nb_bisecants);
	cout << endl;

	c2_points = NEW_int(nb_c2points);

	h = 0;
	for (i = 0; i < nb_bisecants; i++) {
		bi = bisecants[i];
		for (j = 0; j < nb_bisecants; j++) {
			if (j == i) {
				//continue;
				}
			else {
				bj = bisecants[j];
				a = P->intersection_of_two_lines(bi, bj);

				if (Sorting.lint_vec_search_linear(original_arc,
						original_arc_sz, a, idx)) {
					}
				else {
					if (!Sorting.int_vec_search(c2_points, h, a, idx)) {
						for (u = h; u > idx; u--) {
							c2_points[u] = c2_points[u - 1];
							}
						c2_points[idx] = a;
						h++;
						}
					}
				}
			}
		}
	cout << "We found " << h << " c2-points: ";
	Int_vec_print(cout, c2_points, h);
	cout << endl;

	cout << "filtering the external lines:" << endl;
	int nb_filtered_lines;
	long int *filtered_lines;
	int cnt;

	nb_filtered_lines = 0;
	filtered_lines = NEW_lint(nb_external_lines);
	for (i = 0; i < nb_external_lines; i++) {
		a = external_lines[i];
		cnt = 0;
		for (j = 0; j < q + 1; j++) {
			pt = P->Subspaces->Implementation->lines(a, j);
			if (Sorting.int_vec_search(c2_points, h, pt, idx)) {
				cnt++;
				}
			}
		if (cnt > 1) {
			filtered_lines[nb_filtered_lines++] = a;
			}
		}
	cout << "We found " << nb_filtered_lines << " lines of the "
			<< nb_external_lines << " external lines which intersect "
			"the set of c2 points in at least 2 points" << endl;

#if 1

	std::string label_of_set;
	std::string label_of_set_tex;


	label_of_set.assign("_filtered_lines");
	label_of_set_tex.assign("\\_filtered\\_lines");

	A2 = A->Induced_action->induced_action_on_grassmannian(2, verbose_level);
	A3 = A2->Induced_action->restricted_action(
			filtered_lines, nb_filtered_lines,
			label_of_set, label_of_set_tex,
			verbose_level);


	int target_depth = 6;
	poset_classification::poset_with_group_action *Poset2;
	poset_classification::poset_classification_control *Control2;
	poset_classification::poset_classification *Gen2;

	Poset2 = NEW_OBJECT(poset_classification::poset_with_group_action);
	Control2 = NEW_OBJECT(poset_classification::poset_classification_control);
	Poset2->init_subset_lattice(A, A3,
			SaS->Strong_gens,
			verbose_level);

	Gen2 = NEW_OBJECT(poset_classification::poset_classification);

	Gen2->compute_orbits_on_subsets(
		target_depth,
		Control2,
		Poset2,
		//NULL /* int (*candidate_incremental_check_func)(
		//int len, int *S, void *data, int verbose_level) */,
		//NULL /* void *candidate_incremental_check_data */,
		5 /* verbose_level */);


	Gen2->print_orbit_numbers(target_depth);

	int nb_orbits;
	int *covering_number;
	int count;
	int nb_sol = 0;

	nb_orbits = Gen2->nb_orbits_at_level(target_depth);
	cout << "We found " << nb_orbits << " orbits of subsets "
			"of filtered external lines of size " << k << endl;

	covering_number = NEW_int(h);

	string fname;

	fname = "arc_lifting_simeon_q" + std::to_string(q) + "_n"
			+ std::to_string(n) + "_d" + std::to_string(d) + "_k" + std::to_string(k)
			+ "_solutions.txt";
	{
	ofstream fp(fname);

	long int *S; // the arc
	int sz, idx, t;

	S = NEW_lint(nb_external_lines);

	for (i = 0; i < nb_orbits; i++) {

		data_structures_groups::set_and_stabilizer *SaS;

		SaS = Gen2->get_set_and_stabilizer(target_depth,
				i /* orbit_at_level */, 0 /* verbose_level */);

		if ((i % 10000) == 0) {
			cout << "testing orbit " << i << endl;
			}
		Int_vec_zero(covering_number, h);
		for (j = 0; j < h; j++) {
			for (u = 0; u < target_depth; u++) {
				a = SaS->data[u];
				a = filtered_lines[a];
				if (P->Subspaces->is_incident(c2_points[j], a)) {
					covering_number[j]++;
					}
				}
			}
		count = 0;
		for (j = 0; j < h; j++) {
			if (covering_number[j]) {
				count++;
				}
			}
		if (count >= h) {
			cout << "solution" << endl;
			cout << "orbit " << i << " / " << nb_orbits << " : ";
			SaS->print_set_tex(cout);
			cout << endl;
			cout << "covering_number: ";
			Int_vec_print(cout, covering_number, h);
			cout << endl;


			//external_lines[nb_external_lines];
			// subtract the solution from the set
			// of external lines to get the arc:

			sz = nb_external_lines;
			Lint_vec_copy(external_lines, S, nb_external_lines);
			Sorting.lint_vec_heapsort(S, nb_external_lines);


			for (u = 0; u < target_depth; u++) {
				a = SaS->data[u];
				a = filtered_lines[a];
				if (!Sorting.lint_vec_search(S, sz, a, idx, 0)) {
					cout << "the element a=" << a << " cannot be "
							"found in the set of external lines" << endl;
					exit(1);
				}
				for (t = idx + 1; t < sz; t++) {
					S[t - 1] = S[t];
				}
				sz--;
			}


			fp << sz;
			for (t = 0; t < sz; t++) {
				fp << " " << S[t];
			}
			fp << endl;


			nb_sol++;
			}

		//SaS->print_generators_tex(cout);


		} // next i
	fp << -1 << endl;

	FREE_lint(S);

	}
	other::orbiter_kernel_system::file_io Fio;

	cout << "number of solutions = " << nb_sol << endl;
	cout << "written file " << fname << " of size " << Fio.file_size(fname) << endl;

#endif


	FREE_int(type);
}

// #############################################################################
// global functions
// #############################################################################


static void early_test_func_for_arc_callback(long int *S, int len,
	long int *candidates, int nb_candidates,
	long int *good_candidates, int &nb_good_candidates,
	void *data, int verbose_level)
{

	arc_lifting_simeon *Simeon = (arc_lifting_simeon *) data;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "early_test_func_for_arc_callback for set ";
		Lint_vec_print(cout, S, len);
		cout << endl;
		}
	Simeon->early_test_func(S, len,
		candidates, nb_candidates,
		good_candidates, nb_good_candidates,
		verbose_level);
	if (f_v) {
		cout << "early_test_func_for_arc_callback done" << endl;
		}
}

}}}
