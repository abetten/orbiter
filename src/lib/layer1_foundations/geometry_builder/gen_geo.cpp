/*
 * gen_geo.cpp
 *
 *  Created on: Aug 14, 2021
 *      Author: betten
 */




#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace geometry_builder {



#define MAX_V 300



gen_geo::gen_geo()
{
	GB = NULL;

	Decomposition_with_fuse = NULL;

	inc = NULL;

	forget_ivhbar_in_last_isot = FALSE;
	//gen_print_intervall = FALSE;

	//std::string inc_file_name;


	//std::string fname_search_tree;
	ost_search_tree = NULL;
	//std::string fname_search_tree_flags;
	ost_search_tree_flags = NULL;

	Girth_test = NULL;

	Test_semicanonical = NULL;

	Geometric_backtrack_search = NULL;
}

gen_geo::~gen_geo()
{
	if (Decomposition_with_fuse) {
		FREE_OBJECT(Decomposition_with_fuse);
	}

	if (inc) {
		FREE_OBJECT(inc);
	}

	if (Girth_test) {
		FREE_OBJECT(Girth_test);
	}
	if (Test_semicanonical) {
		FREE_OBJECT(Test_semicanonical);
	}

	if (Geometric_backtrack_search) {
		FREE_OBJECT(Geometric_backtrack_search);
	}
}

void gen_geo::init(geometry_builder *GB, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::init" << endl;
	}
	gen_geo::GB = GB;

	inc = NEW_OBJECT(incidence);

	if (f_v) {
		cout << "gen_geo::init before inc->init" << endl;
	}
	inc->init(this, GB->V, GB->B, GB->R, verbose_level);
	if (f_v) {
		cout << "gen_geo::init after inc->init" << endl;
	}

	forget_ivhbar_in_last_isot = FALSE;


	Decomposition_with_fuse = NEW_OBJECT(decomposition_with_fuse);

	if (f_v) {
		cout << "gen_geo::init before Decomposition_with_fuse->init" << endl;
	}
	Decomposition_with_fuse->init(this, verbose_level);
	if (f_v) {
		cout << "gen_geo::init after Decomposition_with_fuse->init" << endl;
	}


	if (f_v) {
		cout << "gen_geo::init before init_semicanonical" << endl;
	}
	init_semicanonical(verbose_level);
	if (f_v) {
		cout << "gen_geo::init before init_semicanonical done" << endl;
	}


	if (GB->Descr->f_girth_test) {

		Girth_test = NEW_OBJECT(girth_test);

		Girth_test->init(this, GB->Descr->girth, verbose_level);

	}

	Geometric_backtrack_search = NEW_OBJECT(geometric_backtrack_search);

	if (f_v) {
		cout << "gen_geo::init before Geometric_backtrack_search->init" << endl;
	}
	Geometric_backtrack_search->init(this, verbose_level);
	if (f_v) {
		cout << "gen_geo::init after Geometric_backtrack_search->init" << endl;
	}

	if (f_v) {
		cout << "gen_geo::init done" << endl;
	}

}


void gen_geo::init_semicanonical(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::init_semicanonical" << endl;
	}

	Test_semicanonical = NEW_OBJECT(test_semicanonical);

	if (f_v) {
		cout << "gen_geo::init_semicanonical before Test_semicanonical->init" << endl;
	}
	Test_semicanonical->init(this, MAX_V, verbose_level);
	if (f_v) {
		cout << "gen_geo::init_semicanonical after Test_semicanonical->init" << endl;
	}

	if (f_v) {
		cout << "gen_geo::init_semicanonical done" << endl;
	}
}


void gen_geo::print_pairs(int line)
{
	int i1, i2, a;

	for (i1 = 0; i1 < line; i1++) {
		cout << "line " << i1 << " : ";
		for (i2 = 0; i2 < i1; i2++) {
			a = inc->pairs[i1][i2];
			cout << a;
		}
		cout << endl;
	}
}

void gen_geo::main2(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "gen_geo::main2, verbose_level = " << verbose_level << endl;
	}
	int V;


	if (f_v) {
		cout << "gen_geo::main2 before generate_all" << endl;
	}
	generate_all(verbose_level);
	if (f_v) {
		cout << "gen_geo::main2 after generate_all" << endl;
	}



	V = inc->Encoding->v;
	iso_type *it;

	it = inc->iso_type_at_line[V - 1];

	if (GB->Descr->f_output_to_inc_file) {

		if (f_v) {
			cout << "gen_geo::main2 f_output_to_inc_file" << endl;
		}

		string fname;

		fname.assign(inc_file_name);
		fname.append(".inc");

		if (f_v) {
			cout << "gen_geo::main2 before it->write_inc_file" << endl;
		}

		it->write_inc_file(fname, verbose_level);

		if (f_v) {
			cout << "gen_geo::main2 after it->write_inc_file" << endl;
		}

		orbiter_kernel_system::file_io Fio;

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	if (GB->Descr->f_output_to_sage_file) {

		cout << "gen_geo::main2 f_output_to_sage_file" << endl;
		string fname;

		fname.assign(inc_file_name);
		fname.append(".sage");
		it->write_sage_file(fname, verbose_level);

		orbiter_kernel_system::file_io Fio;

		cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
	}

	it = inc->iso_type_at_line[V - 1];

	if (it->Canonical_forms->B.size()) {

		if (GB->Descr->f_output_to_blocks_latex_file) {
			string fname;

			fname.assign(inc_file_name);
			fname.append(".blocks_long");
			it->write_blocks_file_long(fname, verbose_level);

			orbiter_kernel_system::file_io Fio;

			cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;

		}

		geometry::object_with_canonical_form *OiP;

		OiP = (geometry::object_with_canonical_form *) it->Canonical_forms->Objects[0];

		if (inc->is_block_tactical(V, OiP->set)) {

			if (GB->Descr->f_output_to_blocks_file) {
				string fname;

				fname.assign(inc_file_name);
				fname.append(".blocks");
				it->write_blocks_file(fname, verbose_level);

				orbiter_kernel_system::file_io Fio;

				cout << "Written file " << fname << " of size " << Fio.file_size(fname) << endl;
			}
		}
	}


	if (inc->iso_type_no_vhbars) {
		it = inc->iso_type_no_vhbars;
		{
			string fname;
			fname.assign(inc_file_name);
			fname.append("_resolved.inc");
			it->write_inc_file(fname, verbose_level);
		}
	}


	print(cout, V, V);

	int i, a;

	for (i = 0; i < GB->Descr->print_at_line.size(); i++) {
		a = GB->Descr->print_at_line[i];
		inc->iso_type_at_line[a - 1]->print_geos(verbose_level);
	}


	if (f_v) {
		cout << "gen_geo::main2 done" << endl;
	}

}

void gen_geo::generate_all(int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 3);

	if (f_v) {
		cout << "gen_geo::generate_all, verbose_level = " << verbose_level << endl;
	}

	//int ret = FALSE;
	int f_already_there;
	//int s_nb_i_vbar, s_nb_i_hbar;
	iso_type *it0, *it1;

	if (f_v) {
		if (GB->Descr->f_lambda) {
			cout << "lambda = " << GB->Descr->lambda << endl;
		}
	}


	setup_output_files(verbose_level);

	if (f_v) {
		cout << "gen_geo::generate_all before it0 = ..." << endl;
	}

	it0 = inc->iso_type_at_line[inc->Encoding->v - 1];

	if (it0 == NULL) {
		cout << "please install a test at line " << inc->Encoding->v << endl;
		exit(1);
	}

	if (f_v) {
		cout << "gen_geo::generate_all before it1 = ..." << endl;
	}

	it1 = inc->iso_type_no_vhbars;


	if (it1 && forget_ivhbar_in_last_isot) {
		cout << "gen_geo::generate_all inc.iso_type_no_vhbars && forget_ivhbar_in_last_isot" << endl;
		goto l_exit;
	}

	inc->gl_nb_GEN = 0;
	if (!Geometric_backtrack_search->First(0 /*verbose_level - 5*/)) {
		//ret = TRUE;

		cout << "gen_geo::generate_all Geometric_backtrack_search->First "
				"returns FALSE, no geometry exists. This is perhaps a bit unusual." << endl;
		goto l_exit;
	}


	while (TRUE) {


		inc->gl_nb_GEN++;
		if (f_v) {
			cout << "gen_geo::generate_all nb_GEN=" << inc->gl_nb_GEN << endl;
			print(cout, inc->Encoding->v, inc->Encoding->v);
			//cout << "pairs:" << endl;
			//inc->print_pairs(inc->Encoding->v);

#if 0
			if ((inc->gl_nb_GEN % gen_print_intervall) == 0) {
				//inc->print(cout, inc->Encoding->v);
			}
#endif
		}


		//cout << "*** do_geo *** geometry no. " << gg->inc.gl_nb_GEN << endl;
#if 0
		if (incidence_back_test(&gg->inc,
			TRUE /* f_verbose */,
			FALSE /* f_very_verbose */)) {
			}
#endif



#if 0
		if (forget_ivhbar_in_last_isot) {
			s_nb_i_vbar = inc->nb_i_vbar;
			s_nb_i_hbar = inc->nb_i_hbar;
			inc->nb_i_vbar = 1;
			inc->nb_i_hbar = 1;
		}
#endif
		if (f_v) {
			cout << "gen_geo::generate_all before it0->add_geometry" << endl;
		}

		it0->add_geometry(inc->Encoding,
				FALSE /* f_partition_fixing_last */,
				f_already_there,
				verbose_level);

		if (f_v) {
			cout << "gen_geo::generate_all it0->add_geometry, f_already_there=" << f_already_there << endl;
		}


		if (GB->Descr->f_search_tree) {
			record_tree(inc->Encoding->v, f_already_there);
		}


		if (FALSE) {
			cout << "gen_geo::generate_all after isot_add for it0" << endl;
		}

		if (f_vv && it0->Canonical_forms->B.size() % 1 == 0) {
			cout << it0->Canonical_forms->B.size() << endl;
			print(cout, inc->Encoding->v, inc->Encoding->v);
		}

#if 0
		if (forget_ivhbar_in_last_isot) {
			inc->nb_i_vbar = s_nb_i_vbar;
			inc->nb_i_hbar = s_nb_i_hbar;
		}
		if (!f_already_there && inc->iso_type_no_vhbars) {
			s_nb_i_vbar = inc->nb_i_vbar;
			s_nb_i_hbar = inc->nb_i_hbar;
			inc->nb_i_vbar = 1;
			inc->nb_i_hbar = 1;

			if (FALSE) {
				cout << "gen_geo::generate_all before isot_add for it1" << endl;
			}
			it1->add_geometry(inc->Encoding,
					inc->Encoding->v, inc,
					f_already_there,
					verbose_level - 2);

			//record_tree(inc->Encoding->v, f_already_there);

			if (FALSE) {
				cout << "gen_geo::generate_all after isot_add for it1" << endl;
			}
			if (!f_already_there) {
				//save_theX(inc->iso_type_no_vhbars->fp);
			}
			inc->nb_i_vbar = s_nb_i_vbar;
			inc->nb_i_hbar = s_nb_i_hbar;
		}
		if (!f_already_there && !inc->iso_type_no_vhbars) {
		}
#endif

		if (!Geometric_backtrack_search->Next(0 /*verbose_level - 5*/)) {
			if (f_v) {
				cout << "gen_geo::generate_all Geometric_backtrack_search->Next "
						"returns FALSE, finished" << endl;
			}
			break;
		}
	}


	close_output_files(verbose_level);


l_exit:
	if (f_v) {
		cout << "gen_geo::generate_all done" << endl;
	}
}

void gen_geo::setup_output_files(int verbose_level)
{
	int f_v = (verbose_level >= 1);


	if (GB->Descr->f_search_tree) {
		fname_search_tree.assign(inc_file_name);
		fname_search_tree.append("_tree.txt");

		if (f_v) {
			cout << "gen_geo::setup_output_files, opening file " << fname_search_tree << endl;
		}

		ost_search_tree = new ofstream;
		ost_search_tree->open (fname_search_tree, std::ofstream::out);
	}
	else {
		ost_search_tree = NULL;
	}

	if (GB->Descr->f_search_tree_flags) {
		fname_search_tree_flags.assign(inc_file_name);
		fname_search_tree_flags.append("_tree_flags.txt");

		if (f_v) {
			cout << "gen_geo::setup_output_files, opening file " << fname_search_tree << endl;
		}

		ost_search_tree_flags = new ofstream;
		ost_search_tree_flags->open (fname_search_tree_flags, std::ofstream::out);
	}
	else {
		ost_search_tree_flags = NULL;
	}

}

void gen_geo::close_output_files(int verbose_level)
{
	//int f_v = (verbose_level >= 1);

	if (GB->Descr->f_search_tree) {
		*ost_search_tree << "-1" << endl;
		ost_search_tree->close();
	}

	if (GB->Descr->f_search_tree_flags) {
		*ost_search_tree_flags << "-1" << endl;
		ost_search_tree_flags->close();
	}

}

void gen_geo::record_tree(int i1, int f_already_there)
{
	int color;

	if (f_already_there) {
		color = COLOR_RED;
	}
	else {
		color = COLOR_GREEN;
	}


	if (ost_search_tree) {

		int row;
		long int rk;

		*ost_search_tree << i1;
		for (row = 0; row < i1; row++) {
			rk = inc->Encoding->rank_row(row);
			*ost_search_tree << " " << rk;
		}
		*ost_search_tree << " " << color;
		*ost_search_tree << endl;
	}

	if (GB->Descr->f_search_tree_flags) {

		std::vector<int> flags;
		int h;

		inc->Encoding->get_flags(i1, flags);

		*ost_search_tree_flags << flags.size();

		for (h = 0; h < flags.size(); h++) {
			*ost_search_tree_flags << " " << flags[h];

		}
		*ost_search_tree_flags << " " << color;
		*ost_search_tree_flags << endl;

	}

}


void gen_geo::print_I_m(int I, int m)
{
	gen_geo_conf *C = Decomposition_with_fuse->get_conf_IJ(I, 0);
	int i1;

	i1 = C->i0 + m;
	print(cout, i1 + 1, i1 + 1);
}

void gen_geo::print(int v)
{
	print(cout, v, v);
}


void gen_geo::increment_pairs_point(int i1, int col, int k)
{
	int ii, ii1;

	for (ii = 0; ii < k; ii++) {
		ii1 = inc->theY[col][ii];
		inc->pairs[i1][ii1]++;
	}
}

void gen_geo::decrement_pairs_point(int i1, int col, int k)
{
	int ii, ii1;

	for (ii = 0; ii < k; ii++) {
		ii1 = inc->theY[col][ii];
		inc->pairs[i1][ii1]--;
	}
}

void gen_geo::girth_test_add_incidence(int i, int j_idx, int j)
{
	if (Girth_test) {
		Girth_test->add_incidence(i, j_idx, j);
	}
}

void gen_geo::girth_test_delete_incidence(int i, int j_idx, int j)
{
	if (Girth_test) {
		Girth_test->delete_incidence(i, j_idx, j);
	}
}

void gen_geo::girth_Floyd(int i, int verbose_level)
{
	if (Girth_test) {
		Girth_test->Floyd(i, verbose_level);
	}
}

int gen_geo::check_girth_condition(int i, int j_idx, int j, int verbose_level)
{
	if (Girth_test) {
#if 0
		inc->print(cout, i + 1, GB->V);
		Girth_test->print_Si(i);
		Girth_test->print_Di(i);
#endif
		return Girth_test->check_girth_condition(i, j_idx, j, verbose_level);
	}
	else {
		return TRUE;
	}
}


int gen_geo::apply_tests(int I, int m, int J, int n, int j, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int fuse_idx;
	int i1, r, k, j0, j1;

	if (f_v) {
		cout << "gen_geo::apply_tests "
				"I=" << I << " m=" << m
				<< " J=" << J << " n=" << n << " j=" << j << endl;
	}
	gen_geo_conf *C = Decomposition_with_fuse->get_conf_IJ(I, J);
	fuse_idx = C->fuse_idx;
	i1 = C->i0 + m;
		// current row

	r = C->r0 + n;
		// index of current incidence within the row

	j0 = C->j0;

	j1 = j0 + j;

	k = inc->K[j1];

	// We want to place an incidence in (i1,j1).

	if (GB->Descr->f_lambda) {

		// We check that there are no repeated columns
		// in the incidence matrix of the design that we create.


		// If this was the last incidence in column j1,
		// make sure the column is different from the previous column.
		// Do this test based on theY[], which lists the incidences in the column.

		// JS = Jochim Selzer


		if (GB->Descr->f_simple) { /* JS 180100 */

			// Note that K[j1] does not take into account
			// the incidence that we want to place in column j1.

			if (Decomposition_with_fuse->F_last_k_in_col[fuse_idx * GB->b_len + J] &&
					k == Decomposition_with_fuse->K1[fuse_idx * GB->b_len + J] - 1) {
				int ii;

				// check whether the column is
				// different from the previous column:

				for (ii = 0; ii <= k; ii++) {
					if (inc->theY[j1 - 1][ii] != inc->theY[j1][ii]) {
						break; // yes, columns differ !
					}
				}
				if (ii > k) {

					// not OK, columns are equal.
					// The design is not simple.

					inc->Encoding->theX_ir(i1, r) = -1;
					inc->theY[j1][k] = -1;
					if (f_v) {
						cout << "gen_geo::apply_tests "
								"I=" << I << " m=" << m
								<< " J=" << J << " n=" << n << " j=" << j
								<< " rejected because not simple" << endl;
					}
					return FALSE;
				}
			}
		} // JS


		// check whether there is enough capacity in the lambda array.
		// This means, check if all scalar products
		// for previous rows with a one in column j1
		// are strictly less than \lambda:
		// This means that there is room for one more pair
		// created by the present incidence in row i1 and column j1

		int ii, ii1;

		for (ii = 0; ii < k; ii++) {

			ii1 = inc->theY[j1][ii];

			if (inc->pairs[i1][ii1] >= GB->Descr->lambda) {

				// no, fail

				inc->Encoding->theX_ir(i1, r) = -1;
				inc->theY[j1][k] = -1;
				break;
			}
		}


		if (ii < k) {

			// There is not enough capacity in the lambda array.
			// So, we cannot put the incidence at (i1,j1):

			if (f_v) {
				cout << "gen_geo::apply_tests "
						"I=" << I << " m=" << m << " J=" << J
						<< " n=" << n << " j=" << j
						<< " rejected because of pair test" << endl;
			}

			return FALSE;
		}

		// increment the pairs counter:

		increment_pairs_point(i1, j1, k);

		// additional test that applies if the row is complete:

		// In this case, all scalar products with previous rows must be equal to lambda:

		if (J == GB->b_len - 1 && n == C->r - 1) {
			for (ii = 0; ii < i1; ii++) {
				if (inc->pairs[i1][ii] != GB->Descr->lambda) {
					// no, fail
					break;
				}
			}
			if (ii < i1) {

				// reject this incidence:

				decrement_pairs_point(i1, j1, k);

				if (f_v) {
					cout << "gen_geo::apply_tests "
							"I=" << I << " m=" << m << " J=" << J
							<< " n=" << n << " j=" << j << " rejected because at least "
									"one pair is not covered lambda times" << endl;
				}
				return FALSE;
			}
		}
	}
	else {

		if (GB->Descr->f_find_square) { /* JS 120100 */

			// Test the square condition.
			// The square condition tests whether there is
			// a pair of points that is contained in two different blocks
			// (corresponding to a square in the incidence matrix, hence the name).
			// We need only consider the pairs of points that include i1.

			if (inc->find_square(i1, r)) {

				// fail, cannot place incidence here

				inc->Encoding->theX_ir(i1, r) = -1;
				if (f_v) {
					cout << "gen_geo::apply_tests "
							"I=" << I << " m=" << m << " J=" << J
							<< " n=" << n << " j=" << j << " rejected because "
									"of square condition" << endl;
				}
				return FALSE;
			}
		}
	}

	girth_test_add_incidence(i1, r, j1);

	if (!check_girth_condition(i1, r, j1, 0 /*verbose_level*/)) {

		girth_test_delete_incidence(i1, r, j1);

		if (GB->Descr->f_lambda) {

			decrement_pairs_point(i1, j1, k);

		}
		if (f_v) {
			cout << "gen_geo::apply_tests "
					"I=" << I << " m=" << m << " J=" << J
					<< " n=" << n << " j=" << j << " rejected because "
							"of girth condition" << endl;
		}
		return FALSE;
	}



	return TRUE;
}

void gen_geo::print(std::ostream &ost, int v, int v_cut)
{
	inc->Encoding->print_partitioned(ost,
			v, v_cut, this, TRUE /* f_print_isot */);
}


void gen_geo::print_override_theX(std::ostream &ost, int *theX, int v, int v_cut)
{
	inc->Encoding->print_partitioned_override_theX(ost,
			v, v_cut, this, theX, TRUE /* f_print_isot */);
}




}}}



