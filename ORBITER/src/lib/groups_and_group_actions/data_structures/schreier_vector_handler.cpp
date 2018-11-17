// schreier_vector_handler.C
//
// Anton Betten
// started: Nov 5, 2018

#include "foundations/foundations.h"
#include "groups_and_group_actions.h"

schreier_vector_handler::schreier_vector_handler()
{
	null();
}

schreier_vector_handler::~schreier_vector_handler()
{
	freeself();
}

void schreier_vector_handler::null()
{
	A = NULL;
	cosetrep = NULL;
	Elt1 = NULL;
	Elt2 = NULL;
	Elt3 = NULL;
	f_check_image = FALSE;
}

void schreier_vector_handler::freeself()
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
	null();
}

void schreier_vector_handler::init(action *A, int f_allow_failure,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "schreier_vector_handler::init" << endl;
	}
	schreier_vector_handler::A = A;
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

int schreier_vector_handler::coset_rep_inv(
		schreier_vector *S,
		int pt, int &pt0,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int ret;

	if (f_v) {
		cout << "schreier_vector_handler::coset_rep_inv "
				"tracing point pt" << endl;
		}
	nb_calls_to_coset_rep_inv++;
	A->element_one(cosetrep, 0);
	if (f_v) {
		cout << "schreier_vector_handler::coset_rep_inv "
				"cosetrep:" << endl;
		A->element_print_quick(cosetrep, cout);
	}
	if (f_v) {
		cout << "schreier_vector_handler::coset_rep_inv "
				"before coset_rep_inv_recursion" << endl;
		}
	ret = coset_rep_inv_recursion(
		S, pt, pt0,
		verbose_level - 1);
	if (f_v) {
		cout << "schreier_vector_handler::coset_rep_inv "
				"after coset_rep_inv_recursion cosetrep:" << endl;
		A->element_print_quick(cosetrep, cout);
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
	//int f_vv = (verbose_level >= 2);
	int hdl, pt_loc, pr, la, n;

	if (f_v) {
		cout << "schreier_vector_handler::coset_rep_inv_recursion "
				"tracing point " << pt << endl;
		}
	nb_calls_to_coset_rep_inv_recursion++;

	//cout << "schreier_vector_coset_rep_inv_compact_general "
	//"pt = " << pt << endl;
	n = S->sv[0];
	if (!int_vec_search(S->sv + 1, S->sv[0], pt, pt_loc)) {
		if (f_allow_failure) {
			if (f_v) {
				cout << "schreier_vector_handler::coset_rep_inv_recursion "
						"did not find point. "
						"f_allow_failure is TRUE, "
						"so we return FALSE" << endl;
			}
			return FALSE;
			}
		else {
			cout << "schreier_vector_handler::coset_rep_inv_recursion "
					"did not find pt" << endl;
			cout << "pt = " << pt << endl;
			cout << "vector of length " << n << endl;
			int_vec_print(cout, S->sv + 1, n);
			cout << endl;
			exit(1);
			}
		}

	// test if the group is trivial:
	if (S->nb_gen == 0) {
		pt0 = pt;
		return TRUE;
		}
	pr = S->sv[1 + n + pt_loc];
	la = S->sv[1 + 2 * n + pt_loc];
	if (pr != -1) {

		if (f_v) {
			cout << "schreier_vector_handler::coset_rep_inv_recursion "
					"prev = " << pr << " label = " << la << endl;
			}
		//hdl = hdl_gen[la];
		hdl = S->gen_hdl_first + la;
		A->element_retrieve(hdl, Elt1, 0);
		//cout << "retrieving generator " << gen_idx << endl;
		//A->element_print_verbose(Elt1, cout);
		A->element_invert(Elt1, Elt2, 0);

		if (f_check_image) {
			int prev;

			prev = A->element_image_of(pt, Elt2, 0);

			//cout << "prev = " << prev << endl;
			if (pr != prev) {
				cout << "schreier_vector_handler::coset_rep_inv_recursion: "
						"pr != prev" << endl;
				cout << "pr = " << pr << endl;
				cout << "prev = " << prev << endl;
				exit(1);
				}
			}

		A->element_mult(cosetrep, Elt2, Elt3, 0);
		A->element_move(Elt3, cosetrep, 0);
		if (f_v) {
			cout << "schreier_vector_handler::coset_rep_inv_recursion "
					"cosetrep:" << endl;
			A->element_print_quick(cosetrep, cout);
			}

		if (f_v) {
			cout << "schreier_vector_handler::coset_rep_inv_recursion "
					"before coset_rep_inv_recursion" << endl;
			}
		if (!coset_rep_inv_recursion(
			S,
			pr, pt0,
			verbose_level)) {
			return FALSE;
			}
		if (f_v) {
			cout << "schreier_vector_handler::coset_rep_inv_recursion "
					"after coset_rep_inv_recursion cosetrep" << endl;
			A->element_print_quick(cosetrep, cout);
			}

		}
	else {
		if (f_v) {
			cout << "prev = -1" << endl;
			}
		pt0 = pt;
		}
	return TRUE;
}



schreier_vector *schreier_vector_handler::sv_read_file(
		int gen_hdl_first, int nb_gen,
		FILE *fp, int verbose_level)
{
	int i, len;
	int4 I, n;
	int f_v = (verbose_level >= 1);
	int f_trivial_group;

	if (f_v) {
		cout << "schreier_vector_handler::sv_read_file" << endl;
		}
	I = fread_int4(fp);
	if (I == 0) {
		cout << "schreier_vector_handler::sv_read_file, "
				"no schreier vector" << endl;
		return NULL;
		}
	f_trivial_group = fread_int4(fp);
	n = fread_int4(fp);

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
		osv[1 + i] = fread_int4(fp);
		}
	//sv = osv;
	Sv = NEW_OBJECT(schreier_vector);
	Sv->init(gen_hdl_first, nb_gen, osv, verbose_level);
	cout << "schreier_vector_handler::sv_read_file "
			"read sv with " << n << " live points" << endl;

	if (f_v) {
		cout << "schreier_vector_handler::sv_read_file finished" << endl;
		}
	return Sv;
}

void schreier_vector_handler::sv_write_file(schreier_vector *Sv,
		FILE *fp, int verbose_level)
{
	int i, len;
	int f_v = (verbose_level >= 1);
	int f_trivial_group;

	if (f_v) {
		cout << "schreier_vector_handler::sv_write_file" << endl;
		}
	if (Sv == NULL) {
		fwrite_int4(fp, 0);
		}
	else {
		fwrite_int4(fp, 1);
		if (Sv->nb_gen == 0) {
			f_trivial_group = TRUE;
			}
		else {
			f_trivial_group = FALSE;
			}
		fwrite_int4(fp, f_trivial_group);
		if (Sv->sv == NULL) {
			cout << "schreier_vector_handler::sv_write_file "
					"Sv->sv == NULL" << endl;
			exit(1);
		}
		int *osv = Sv->sv;
		int n = osv[0];
		fwrite_int4(fp, n);
		if (f_trivial_group) {
			len = n;
			}
		else {
			len = 3 * n;
			}
		for (i = 0; i < len; i++) {
			fwrite_int4(fp, osv[1 + i]);
			}
		}

	if (f_v) {
		cout << "schreier_vector_handler::sv_write_file "
				"finished" << endl;
		}
}

