// schreier_vector_handler.cpp
//
// Anton Betten
// started: Nov 5, 2018

#include "layer1_foundations/foundations.h"
#include "group_actions.h"

using namespace std;


namespace orbiter {
namespace layer3_group_actions {
namespace data_structures_groups {


schreier_vector_handler::schreier_vector_handler()
{
	A = NULL;
	A2 = NULL;
	cosetrep = NULL;
	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
	f_check_image = false;
	f_allow_failure = false;
	nb_calls_to_coset_rep_inv = 0;
	nb_calls_to_coset_rep_inv_recursion = 0;
}


schreier_vector_handler::~schreier_vector_handler()
{
	if (cosetrep) {
		FREE_int(cosetrep);
	}
	if (Elt1) {
		FREE_int(Elt1);
	}
	if (Elt2) {
		FREE_int(Elt2);
	}
	if (Elt3) {
		FREE_int(Elt3);
	}
}

void schreier_vector_handler::init(
		actions::action *A, actions::action *A2,
		int f_allow_failure,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier_vector_handler::init" << endl;
	}
	schreier_vector_handler::A = A;
	schreier_vector_handler::A2 = A2;
	schreier_vector_handler::f_allow_failure = f_allow_failure;
	nb_calls_to_coset_rep_inv = 0;
	nb_calls_to_coset_rep_inv_recursion = 0;
	cosetrep = NEW_int(A->elt_size_in_int);
	Elt1 = NEW_int(A->elt_size_in_int);
	Elt2 = NEW_int(A->elt_size_in_int);
	Elt3 = NEW_int(A->elt_size_in_int);
	if (f_v) {
		cout << "schreier_vector_handler::init done" << endl;
	}
}

void schreier_vector_handler::print_info_and_generators(
		schreier_vector *S)
{
	cout << "action A:" << endl;
	A->print_info();
	cout << "action A2:" << endl;
	A2->print_info();
	if (S->f_has_local_generators) {
		cout << "action S->local_gens->A:" << endl;
		S->local_gens->A->print_info();
		cout << "schreier_vector_handler::coset_rep_inv "
				"we have " << S->local_gens->len
		<< " local generators" << endl;
		int i;
		cout << "the local generators are:" << endl;
		for (i = 0; i < S->local_gens->len; i++) {
			cout << "generator " << i << " / "
					<< S->local_gens->len << ":" << endl;
			S->local_gens->A->Group_element->element_print_quick(
					S->local_gens->ith(i), cout);
		}
	}
	else {
		cout << "there are no local generators" << endl;
	}
}

int schreier_vector_handler::coset_rep_inv_lint(
		schreier_vector *S,
		long int pt, long int &pt0,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret;
	int pt_int;
	int pt0_int;

	if (f_v) {
		cout << "schreier_vector_handler::coset_rep_inv_lint "
				"tracing point pt" << endl;
	}
	pt_int = pt;
	ret = coset_rep_inv(S, pt_int, pt0_int, verbose_level);
	if (f_v) {
		cout << "schreier_vector_handler::coset_rep_inv_lint "
				"pt = " << pt_int << " -> pt0 = " << pt0_int << endl;
	}
	pt0 = pt0_int;
	return ret;
}

int schreier_vector_handler::coset_rep_inv(
		schreier_vector *S,
		int pt, int &pt0,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int ret;

	if (f_v) {
		cout << "schreier_vector_handler::coset_rep_inv "
				"tracing point pt" << endl;
	}
	nb_calls_to_coset_rep_inv++;
	A->Group_element->element_one(cosetrep, 0);
	if (f_vv) {
		cout << "schreier_vector_handler::coset_rep_inv "
				"cosetrep:" << endl;
		A->Group_element->element_print_quick(cosetrep, cout);
	}
	if (f_v) {
		print_info_and_generators(S);
	}
	if (f_v) {
		cout << "schreier_vector_handler::coset_rep_inv "
				"before coset_rep_inv_recursion" << endl;
	}
	ret = coset_rep_inv_recursion(
		S, pt, pt0,
		verbose_level - 2);
	if (f_vv) {
		cout << "schreier_vector_handler::coset_rep_inv "
				"after coset_rep_inv_recursion cosetrep:" << endl;
		A->Group_element->element_print_quick(cosetrep, cout);
	}
	if (f_v) {
		if (ret) {
			cout << "schreier_vector_handler::coset_rep_inv "
					"done " << pt << "->" << pt0 << endl;
		}
		else {
			cout << "schreier_vector_handler::coset_rep_inv "
					"failure to find point" << endl;
		}
	}
	return ret;
}

int schreier_vector_handler::coset_rep_inv_recursion(
	schreier_vector *S,
	int pt, int &pt0,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 2);
	int hdl, pt_loc, pr, la, n;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "schreier_vector_handler::coset_rep_inv_recursion "
				"tracing point " << pt << endl;
	}
	nb_calls_to_coset_rep_inv_recursion++;

	//cout << "schreier_vector_coset_rep_inv_compact_general "
	//"pt = " << pt << endl;
	n = S->sv[0];
	if (!Sorting.int_vec_search(S->sv + 1, S->sv[0], pt, pt_loc)) {
		if (f_allow_failure) {
			if (f_v) {
				cout << "schreier_vector_handler::coset_rep_inv_recursion "
						"did not find point. "
						"f_allow_failure is true, "
						"so we return false" << endl;
			}
			return false;
		}
		else {
			cout << "schreier_vector_handler::coset_rep_inv_recursion "
					"did not find pt" << endl;
			cout << "pt = " << pt << endl;
			cout << "vector of length " << n << endl;
			Int_vec_print(cout, S->sv + 1, n);
			cout << endl;
			exit(1);
		}
	}

	// test if the group is trivial:
	if (S->nb_gen == 0) {
		pt0 = pt;
		return true;
	}
	pr = S->sv[1 + n + pt_loc];
	la = S->sv[1 + 2 * n + pt_loc];
	if (pr != -1) {

		if (f_v) {
			cout << "schreier_vector_handler::coset_rep_inv_recursion "
					"prev = " << pr << " label = " << la << endl;
		}
		//hdl = hdl_gen[la];
		if (S->f_has_local_generators) {
			if (f_v) {
				cout << "schreier_vector_handler::coset_rep_inv_recursion "
						"using local_generator" << endl;
				cout << "generator " << la << ":" << endl;
			}
			if (f_vv) {
				A->Group_element->element_print_quick(S->local_gens->ith(la), cout);
			}
			A->Group_element->element_move(S->local_gens->ith(la), Elt1, 0);
		}
		else {
			if (f_v) {
				cout << "schreier_vector_handler::coset_rep_inv_recursion "
						"using global generator" << endl;
			}
			hdl = S->gen_hdl_first + la;
			A->Group_element->element_retrieve(hdl, Elt1, 0);
			//cout << "retrieving generator " << gen_idx << endl;
		}
		//A->element_print_verbose(Elt1, cout);
		A->Group_element->element_invert(Elt1, Elt2, 0);

		if (f_check_image) {
			int prev;

			if (f_v) {
				cout << "schreier_vector_handler::coset_rep_inv_recursion "
						"check_image is true" << endl;
			}
			prev = A2->Group_element->element_image_of(pt, Elt2, 0);

			//cout << "prev = " << prev << endl;
			if (pr != prev) {
				cout << "schreier_vector_handler::coset_rep_inv_recursion: "
						"pr != prev" << endl;
				cout << "pr = " << pr << endl;
				cout << "prev = " << prev << endl;
				cout << "Elt1:" << endl;
				A->Group_element->element_print_quick(Elt1, cout);
				cout << "Elt2:" << endl;
				A->Group_element->element_print_quick(Elt2, cout);
				exit(1);
			}
		}

		A->Group_element->element_mult(cosetrep, Elt2, Elt3, 0);
		A->Group_element->element_move(Elt3, cosetrep, 0);
		if (f_v) {
			cout << "schreier_vector_handler::coset_rep_inv_recursion "
					"cosetrep:" << endl;
			A->Group_element->element_print_quick(cosetrep, cout);
		}

		if (f_v) {
			cout << "schreier_vector_handler::coset_rep_inv_recursion "
					"before coset_rep_inv_recursion" << endl;
		}
		if (!coset_rep_inv_recursion(
			S,
			pr, pt0,
			verbose_level)) {
			return false;
		}
		if (f_v) {
			cout << "schreier_vector_handler::coset_rep_inv_recursion "
					"after coset_rep_inv_recursion cosetrep" << endl;
			A->Group_element->element_print_quick(cosetrep, cout);
		}

	}
	else {
		if (f_v) {
			cout << "prev = -1" << endl;
			}
		pt0 = pt;
	}
	return true;
}



schreier_vector *schreier_vector_handler::sv_read_file(
		int gen_hdl_first, int nb_gen,
		std::ifstream &fp, int verbose_level)
{
	int i, len;
	int I, n;
	int f_v = (verbose_level >= 1);
	int f_trivial_group;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "schreier_vector_handler::sv_read_file" << endl;
	}
	fp.read((char *)&I, sizeof(int));
	//I = Fio.fread_int4(fp);
	if (I == 0) {
		cout << "schreier_vector_handler::sv_read_file, "
				"no schreier vector" << endl;
		return NULL;
	}
	fp.read((char *)&f_trivial_group, sizeof(int));
	//f_trivial_group = Fio.fread_int4(fp);
	fp.read((char *)&n, sizeof(int));
	//n = Fio.fread_int4(fp);

	schreier_vector *Sv;

	int *osv;
	if (f_trivial_group) {
		osv = NEW_int(n + 1);
		len = n;
	}
	else {
		osv = NEW_int(3 * n + 1);
		len = 3 * n;
	}
	osv[0] = n;
	for (i = 0; i < len; i++) {
		//osv[1 + i] = Fio.fread_int4(fp);
		fp.read((char *)&osv[1 + i], sizeof(int));
	}
	//sv = osv;
	Sv = NEW_OBJECT(schreier_vector);
	Sv->init(gen_hdl_first, nb_gen, osv, verbose_level);
	if (f_v) {
		cout << "schreier_vector_handler::sv_read_file "
				"read sv with " << n << " live points" << endl;
	}
	if (f_v) {
		cout << "schreier_vector_handler::sv_read_file finished" << endl;
	}
	return Sv;
}

void schreier_vector_handler::sv_write_file(
		schreier_vector *Sv,
		std::ofstream &fp, int verbose_level)
{
	int i, len, tmp;
	int f_v = (verbose_level >= 1);
	int f_trivial_group;
	orbiter_kernel_system::file_io Fio;

	if (f_v) {
		cout << "schreier_vector_handler::sv_write_file" << endl;
	}
	if (Sv == NULL) {
		//Fio.fwrite_int4(fp, 0);
		tmp = 0;
		fp.write((char *)&tmp, sizeof(int));
	}
	else {
		//Fio.fwrite_int4(fp, 1);
		tmp = 1;
		fp.write((char *)&tmp, sizeof(int));
		if (Sv->nb_gen == 0) {
			f_trivial_group = true;
		}
		else {
			f_trivial_group = false;
		}
		//Fio.fwrite_int4(fp, f_trivial_group);
		fp.write((char *)&f_trivial_group, sizeof(int));
		if (Sv->sv == NULL) {
			cout << "schreier_vector_handler::sv_write_file "
					"Sv->sv == NULL" << endl;
			exit(1);
		}
		int *osv = Sv->sv;
		int n = osv[0];
		//Fio.fwrite_int4(fp, n);
		fp.write((char *)&n, sizeof(int));
		if (f_trivial_group) {
			len = n;
		}
		else {
			len = 3 * n;
		}
		for (i = 0; i < len; i++) {
			//Fio.fwrite_int4(fp, osv[1 + i]);
			fp.write((char *)&osv[1 + i], sizeof(int));
		}
	}

	if (f_v) {
		cout << "schreier_vector_handler::sv_write_file "
				"finished" << endl;
	}
}

data_structures::set_of_sets *schreier_vector_handler::get_orbits_as_set_of_sets(
		schreier_vector *Sv,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *orbit_reps;
	int nb_orbits;
	data_structures::set_of_sets *SoS;
	int i, t;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "schreier_vector_handler::get_orbits_as_set_of_sets" << endl;
	}
	if (Sv->nb_gen == 0) {
		cout << "schreier_vector_handler::get_orbits_as_set_of_sets "
				"Sv->nb_gen == 0" << endl;
		exit(1);
	}
	int n;
	int *pts;
	int *depth;
	int *ancestor;

	n = Sv->sv[0];
	pts = Sv->sv + 1;

	Sv->count_number_of_orbits_and_get_orbit_reps(
		orbit_reps, nb_orbits);
	SoS = NEW_OBJECT(data_structures::set_of_sets);
	int *prev;

	prev = pts + n;
#if 0
	cout << "i : pts : prev" << endl;
	for (i = 0; i < n; i++) {
		cout << i << " : " << pts[i] << " : " << prev[i] << endl;
	}
#endif


#if 0
	depth = NEW_int(n);
	ancestor = NEW_int(n);

	for (i = 0; i < n; i++) {
		depth[i] = -1;
		ancestor[i] = -1;
	}
	for (i = 0; i < n; i++) {
		Sorting.schreier_vector_determine_depth_recursion(n,
				pts, prev, false, depth, ancestor, i);
	}
#else
	Sorting.schreier_vector_compute_depth_and_ancestor(
			n, pts, prev, false /* f_prev_is_point_index */, NULL,
			depth, ancestor, verbose_level - 2);
#endif
#if 0
	cout << "i : pts : depth : ancestor" << endl;
	for (i = 0; i < n; i++) {
		cout << i << " : " << pts[i] << " : " << depth[i] << " : " << ancestor[i] << endl;
	}
#endif

	data_structures::tally C;
	int f, a;

	C.init(ancestor, n, false, 0);

	SoS->init_basic_with_Sz_in_int(A2->degree /* underlying_set_size*/,
			C.nb_types, C.type_len, verbose_level);

	FREE_int(depth);
	FREE_int(ancestor);

	for (t = 0; t < C.nb_types; t++) {
		f = C.type_first[t];
		for (i = 0; i < C.type_len[t]; i++) {
			a = C.sorting_perm_inv[f + i];
			SoS->Sets[t][i] = pts[a];
		}
	}

	if (f_v) {
		cout << "schreier_vector_handler::get_orbits_as_set_of_sets "
				"done" << endl;
	}
	return SoS;
}

}}}

