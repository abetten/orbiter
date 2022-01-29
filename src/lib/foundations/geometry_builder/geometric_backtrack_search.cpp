/*
 * geometric_backtrack_search.cpp
 *
 *  Created on: Dec 27, 2021
 *      Author: betten
 */



#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace geometry_builder {



geometric_backtrack_search::geometric_backtrack_search()
{
	gg = NULL;
	Row_stabilizer_orbits = NULL;
	Row_stabilizer_orbit_idx = NULL;
}

geometric_backtrack_search::~geometric_backtrack_search()
{
	if (Row_stabilizer_orbit_idx) {
		FREE_int(Row_stabilizer_orbit_idx);
	}
}

void geometric_backtrack_search::init(gen_geo *gg, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "geometric_backtrack_search::init" << endl;
	}
	geometric_backtrack_search::gg = gg;

	Row_stabilizer_orbits = (iso_type **) NEW_pvoid(gg->GB->V);
	for (i = 0; i < gg->GB->V; i++) {
		Row_stabilizer_orbits[i] = NULL;
	}
	Row_stabilizer_orbit_idx = NEW_int(gg->GB->V);
	Orbiter->Int_vec->zero(Row_stabilizer_orbit_idx, gg->GB->V);


	if (f_v) {
		cout << "geometric_backtrack_search::done" << endl;
	}
}


int geometric_backtrack_search::First(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_backtrack_search::First" << endl;
	}
	int I;

	I = 0;
	while (TRUE) {
		while (TRUE) {
			if (I >= gg->GB->v_len) {
				return TRUE;
			}
			if (!BlockFirst(I, verbose_level)) {
				break;
			}
			I++;
		}
		// I-th element could not initialize, move on
		while (TRUE) {
			if (I == 0) {
				return FALSE;
			}
			I--;
			if (BlockNext(I, verbose_level)) {
				break;
			}
		}
		// I-th element has been incremented. Initialize elements after it:
		I++;
	}
}

int geometric_backtrack_search::Next(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_backtrack_search::Next" << endl;
	}
	int I;

	I = gg->GB->v_len - 1;
	while (TRUE) {
		while (TRUE) {
			if (BlockNext(I, verbose_level)) {
				break;
			}
			if (I == 0) {
				return FALSE;
			}
			I--;
		}
		// I-th element has been incremented. Initialize elements after it:
		while (TRUE) {
			if (I >= gg->GB->v_len - 1) {
				return TRUE;
			}
			I++;
			if (!BlockFirst(I, verbose_level)) {
				break;
			}
		}
		// I-th element could not initialize, move on
		I--;
	}
}

int geometric_backtrack_search::BlockFirst(int I, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_backtrack_search::BlockFirst I=" << I << endl;
	}

	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, 0);
	int m;

	m = 0;
	while (TRUE) {
		while (TRUE) {
			if (m >= C->v) {
				return TRUE;
			}
			if (!RowFirstSplit(I, m, verbose_level)) {
				break;
			}
			m++;
		}
		// m-th element could not initialize, move on
		while (TRUE) {
			if (m == 0) {
				return FALSE;
			}
			m--;
			if (RowNextSplit(I, m, verbose_level)) {
				break;
			}
		}
		// m-th element has been incremented. Initialize elements after it:
		m++;
	}
}

int geometric_backtrack_search::BlockNext(int I, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_backtrack_search::BlockNext I=" << I << endl;
	}
	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, 0);

	int m;

	m = C->v - 1;
	while (TRUE) {
		while (TRUE) {
			if (RowNextSplit(I, m, verbose_level)) {
				break;
			}
			if (m == 0) {
				return FALSE;
			}
			m--;
		}
		// m-th element has been incremented. Initialize elements after it:
		while (TRUE) {
			if (m >= C->v - 1) {
				return TRUE;
			}
			m++;
			if (!RowFirstSplit(I, m, verbose_level)) {
				break;
			}
		}
		// m-th element could not initialize, move on
		m--;
	}
}

#define GEO_LINE_SPLIT

int geometric_backtrack_search::RowFirstSplit(int I, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_backtrack_search::RowFirstSplit "
				"I=" << I << " m=" << m << endl;
	}

#ifdef GEO_LINE_SPLIT
	iso_type *it;
	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, 0);
	int i1;

	i1 = C->i0 + m;
	it = gg->inc->iso_type_at_line[i1];
	if (it && it->f_split) {
		if ((it->Canonical_forms->B.size() % it->split_modulo) != it->split_remainder) {
			return FALSE;
		}
	}
	if (!RowFirst0(I, m, verbose_level)) {
		return FALSE;
	}
	return TRUE;
#else
	return RowFirst0(I, m, verbose_level);
#endif
}

int geometric_backtrack_search::RowNextSplit(int I, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_backtrack_search::RowNextSplit "
				"I=" << I << " m=" << m << endl;
	}
#ifdef GEO_LINE_SPLIT
	iso_type *it;
	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, 0);
	int i1;

	i1 = C->i0 + m;
	it = gg->inc->iso_type_at_line[i1];
	if (it && it->f_split) {
		if ((it->Canonical_forms->B.size() % it->split_modulo) != it->split_remainder) {
			return FALSE;
		}
	}
	if (!RowNext0(I, m, verbose_level)) {
		return FALSE;
	}
	return TRUE;
#else
	return LineNext0(gg, I, m, verbose_level);
#endif
}

int geometric_backtrack_search::geo_back_test(int I, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_backtrack_search::geo_back_test "
				"I=" << I << endl;
	}
	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, 0);
	int i0, i1, m, f_already_there, control_line;
	iso_type *it;

	i0 = C->i0;
	control_line = i0 + C->v - 1;
	for (m = 0; m < C->v - 1; m++) {
		i1 = i0 + m;
		it = gg->inc->iso_type_at_line[i1];

		if (it && it->f_generate_first && !it->f_beginning_checked) {

			it->add_geometry(gg->inc->Encoding,
					FALSE /* f_partition_fixing_last */,
					f_already_there,
					verbose_level - 2);



			gg->record_tree(i1 + 1, f_already_there);


			if (!f_already_there) {
				it->f_beginning_checked = TRUE;
				continue;
			}
			gg->inc->back_to_line = i1;
			return FALSE;
		}
	}
	return TRUE;
}


int geometric_backtrack_search::RowFirst0(int I, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_backtrack_search::RowFirst0 "
				"I=" << I << " m=" << m << endl;
	}
	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, 0);

	int f_already_there, i1, control_line;
	iso_type *it;

	i1 = C->i0 + m;
	if (!RowFirst(I, m, verbose_level)) {
		return FALSE;
	}
	control_line = C->i0 + C->v - 1;
	it = gg->inc->iso_type_at_line[i1];
	if (i1 != control_line && it && it->f_generate_first) {
		it->f_beginning_checked = FALSE;
		return TRUE;
	}
	if (i1 == control_line) {
		if (!geo_back_test(I, verbose_level)) {
			if (!RowNext(I, m, verbose_level)) {
				return FALSE;
			}
			cout << "geometric_backtrack_search::RowFirst0 "
					"back_to_line && f_new_situation == TRUE" << endl;
			exit(1);
		}
		// survived the back test,
		// and now one test of the first kind:
	}
	if (i1 == gg->inc->Encoding->v - 1) {
		// a new geometry is completed
		// let the main routine add it
		return TRUE;
	}

	// now we know we have a partial geometry on i1 < v lines.

	if (it) {
		// test of the first kind
		while (TRUE) {
			if (f_v) {
				cout << "geometric_backtrack_search::RowFirst0 "
						"I=" << I << " m=" << m
						<< " before isot_add" << endl;
				gg->print(cout, i1 + 1, i1 + 1);
			}

			it->add_geometry(gg->inc->Encoding,
					FALSE /* f_partition_fixing_last */,
					f_already_there,
					verbose_level - 2);

			gg->record_tree(i1 + 1, f_already_there);

			if (f_v) {
				cout << "geometric_backtrack_search::RowFirst0 "
						"I=" << I << " m=" << m
						<< " after isot_add" << endl;
			}
			if (!f_already_there) {
				break;
			}
			if (!RowNext(I, m, verbose_level)) {
				return FALSE;
			}
		}
		// now: a new geometry has been produced,
		// f_already_there is FALSE
	}
	return TRUE;
}

int geometric_backtrack_search::RowNext0(int I, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_backtrack_search::RowNext0 "
				"I=" << I << " m=" << m << endl;
	}
	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, 0);

	int f_already_there, i1, control_line;
	iso_type *it;

	i1 = C->i0 + m;
	if (!RowNext(I, m, verbose_level)) {
		return FALSE;
	}
	control_line = C->i0 + C->v - 1;
	it = gg->inc->iso_type_at_line[i1];
	if (i1 != control_line && it && it->f_generate_first) {
		it->f_beginning_checked = FALSE;
#if 0
		gg->inc.nb_GEO[i1] = ((ISO_TYPE *)
		gg->inc.iso_type[control_line])->nb_GEO;
#endif
		return TRUE;
	}
	if (i1 == control_line) {
		if (!geo_back_test(I, verbose_level)) {
			if (!RowNext(I, m, verbose_level)) {
				return FALSE;
			}
			cout << "geometric_backtrack_search::RowNext0 "
					"back_to_line && f_new_situation == TRUE" << endl;
			exit(1);
		}
		// survived the back test,
		// and now one test of the first kind:
	}
	if (i1 == gg->inc->Encoding->v - 1) {
		// a new geometry is completed
		// let the main routine add it
		return TRUE;
	}
	if (it) {
		while (TRUE) {
			it->add_geometry(gg->inc->Encoding,
					FALSE /* f_partition_fixing_last */,
				f_already_there,
				verbose_level - 2);

			gg->record_tree(i1 + 1, f_already_there);


			if (!f_already_there) {
				break;
			}
			if (!RowNext(I, m, verbose_level)) {
				return FALSE;
			}
		}
		// now: a new geometry has been produced,
		// f_already_there is FALSE
	}
	return TRUE;
}


int geometric_backtrack_search::RowFirst(int I, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, 0);
	int i1;
	int ret;

	i1 = C->i0 + m;

	if (f_v) {
		cout << "geometric_backtrack_search::RowFirst "
				"I=" << I << " m=" << m << " i1=" << i1 << endl;
	}

	gg->girth_Floyd(i1, verbose_level);


	if (gg->GB->Descr->f_orderly) {

		ret = RowFirstOrderly(I, m, verbose_level);

	}
	else {

		ret = RowFirstLexLeast(I, m, verbose_level);
	}


	return ret;

}

int geometric_backtrack_search::RowNext(int I, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_backtrack_search::RowNext "
				"I=" << I << " m=" << m << endl;
	}
	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, 0);
	int i1;
	int ret;

	i1 = C->i0 + m;
	if (gg->inc->back_to_line != -1 && gg->inc->back_to_line < i1) {
		RowClear(I, m);
		return FALSE;
	}
	if (gg->inc->back_to_line != -1 && gg->inc->back_to_line == i1) {
		gg->inc->back_to_line = -1;
	}

	if (gg->GB->Descr->f_orderly) {

		ret = RowNextOrderly(I, m, verbose_level);

	}
	else {
		ret = RowNextLexLeast(I, m, verbose_level);
	}


	return ret;
}

int geometric_backtrack_search::RowFirstLexLeast(int I, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, 0);
	int J, i1;

	i1 = C->i0 + m;

	if (f_v) {
		cout << "geometric_backtrack_search::RowFirstLexLeast "
				"I=" << I << " m=" << m << " i1=" << i1 << endl;
	}


	J = 0;
	while (TRUE) {
		while (TRUE) {
			if (J >= gg->GB->b_len) {
				if (f_v) {
					cout << "geometric_backtrack_search::RowFirstLexLeast" << endl;
					gg->print(cout, i1 + 1, gg->inc->Encoding->v);
				}
				return TRUE;
			}
			if (!ConfFirst(I, m, J, verbose_level)) {
				break;
			}
			J++;
		}
		// J-th element could not initialize, move on
		while (TRUE) {
			if (J == 0) {
				return FALSE;
			}
			J--;
			if (ConfNext(I, m, J, verbose_level)) {
				break;
			}
		}
		// J-th element has been incremented. Initialize elements after it:
		J++;
	}
}

int geometric_backtrack_search::RowNextLexLeast(int I, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, 0);
	int J, i1;

	i1 = C->i0 + m;

	if (f_v) {
		cout << "geometric_backtrack_search::RowNextLexLeast "
				"I=" << I << " m=" << m << " i1=" << i1 << endl;
	}

	J = gg->GB->b_len - 1;
	while (TRUE) {
		while (TRUE) {
			if (ConfNext(I, m, J, verbose_level)) {
				break;
			}
			if (J == 0) {
				return FALSE;
			}
			J--;
		}
		// J-th element has been incremented. Initialize elements after it:
		while (TRUE) {
			if (J >= gg->GB->b_len - 1) {
				if (f_v) {
					cout << "geometric_backtrack_search::RowNextLexLeast" << endl;
					gg->print(cout, i1 + 1, gg->inc->Encoding->v);
				}
				return TRUE;
			}
			J++;
			if (!ConfFirst(I, m, J, verbose_level)) {
				break;
			}
		}
		// J-th element could not initialize, move on
		J--;
	}
}

int geometric_backtrack_search::RowFirstOrderly(int I, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, 0);
	int i1;
	int ret = FALSE;
	int f_already_there;

	i1 = C->i0 + m;

	if (f_v) {
		cout << "geometric_backtrack_search::RowFirstOrderly "
				"I=" << I << " m=" << m << " i1=" << i1 << endl;
	}

	iso_type *It;

	Row_stabilizer_orbits[i1] = NEW_OBJECT(iso_type);

	It = Row_stabilizer_orbits[i1];

	It->init(gg, i1 + 1,
			FALSE /* f_orderly */,
			verbose_level);

	if (!RowFirstLexLeast(I, m, verbose_level - 5)) {
	}
	else {

		while (TRUE) {
			It->add_geometry(gg->inc->Encoding,
					TRUE /* f_partition_fixing_last */,
					f_already_there,
					0 /*verbose_level - 2*/);

			if (!RowNextLexLeast(I, m, verbose_level - 5)) {
				break;
			}
		}
	}

	if (f_v) {
		cout << "geometric_backtrack_search::RowFirstOrderly "
				"I=" << I << " m=" << m << " i1=" << i1
				<< " number of possible rows: " << It->Canonical_forms->B.size() << endl;
	}

	if (It->Canonical_forms->B.size()) {

		place_row(I, m, 0 /* idx */, verbose_level);

		ret = TRUE;
		Row_stabilizer_orbit_idx[i1] = 0;

	}
	else {
		FREE_OBJECT(It);
		Row_stabilizer_orbits[i1] = NULL;
		Row_stabilizer_orbit_idx[i1] = -1;
		ret = FALSE;
	}

	return ret;
}

void geometric_backtrack_search::place_row(int I, int m, int idx, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_backtrack_search::place_row "
				"I=" << I << " m=" << m << " idx=" << idx << endl;
	}

	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, 0);
	int i1;

	i1 = C->i0 + m;

	if (f_v) {
		cout << "geometric_backtrack_search::place_row "
				"I=" << I << " m=" << m << " i1=" << i1 << endl;
	}

	iso_type *It;
	It = Row_stabilizer_orbits[i1];

	geometry::object_with_canonical_form *OwCF;
	int J, r, j, n, j0;

	OwCF = (geometry::object_with_canonical_form *) It->Canonical_forms->Objects[idx];


	r = 0;
	for (J = 0; J < gg->GB->b_len; J++) {
		gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, J);

		j0 = C->j0;

		for (n = 0; n < C->r; n++, r++) {

			j = OwCF->set[It->sum_R_before + r] - i1 * gg->inc->Encoding->b - j0;

			if (f_v) {
				cout << "geometric_backtrack_search::place_row "
						"I=" << I << " m=" << m << " J=" << J
						<< " n=" << n << " j=" << j << " before TryToPlace" << endl;
			}

			if (!TryToPlace(I, m, J, n, j, verbose_level)) {
				cout << "geometric_backtrack_search::place_row !TryToPlace" << endl;
				exit(1);
			}

			if (f_v) {
				cout << "geometric_backtrack_search::place_row "
						"I=" << I << " m=" << m << " J=" << J
						<< " n=" << n << " j=" << j << " after TryToPlace" << endl;
			}

			gg->Test_semicanonical->markers_update(I, m, J, n, j,
					i1, j0, r,
					verbose_level);

		}
	}

	if (f_v) {
		cout << "geometric_backtrack_search::place_row "
				"I=" << I << " m=" << m << " idx=" << idx << " done" << endl;
	}
}

int geometric_backtrack_search::RowNextOrderly(int I, int m, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_backtrack_search::RowNextOrderly "
				"I=" << I << " m=" << m << endl;
	}

	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, 0);
	int i1;
	int ret = FALSE;

	i1 = C->i0 + m;

	if (f_v) {
		cout << "geometric_backtrack_search::RowNextOrderly "
				"I=" << I << " m=" << m << " i1=" << i1 << endl;
	}
	RowClear(I, m);

	iso_type *It;
	It = Row_stabilizer_orbits[i1];


	if (Row_stabilizer_orbit_idx[i1] == It->Canonical_forms->B.size() - 1) {

		FREE_OBJECT(Row_stabilizer_orbits[i1]);
		Row_stabilizer_orbits[i1] = NULL;
		Row_stabilizer_orbit_idx[i1] = -1;
		if (f_v) {
			cout << "geometric_backtrack_search::RowNextOrderly "
					"I=" << I << " m=" << m << " i1=" << i1 << " finished" << endl;
		}
		ret = FALSE;
	}
	else {

		Row_stabilizer_orbit_idx[i1]++;
		place_row(I, m, Row_stabilizer_orbit_idx[i1] /* idx */, verbose_level);
		if (f_v) {
			cout << "geometric_backtrack_search::RowNextOrderly "
					"I=" << I << " m=" << m << " i1=" << i1
					<< " moved on to the next sibling" << endl;
			gg->print(cout, i1 + 1, gg->inc->Encoding->v);
		}

		ret = TRUE;

	}

	return ret;
}

void geometric_backtrack_search::RowClear(int I, int m)
{
	int J;

	for (J = gg->GB->b_len - 1; J >= 0; J--) {
		ConfClear(I, m, J);
	}
}

int geometric_backtrack_search::ConfFirst(int I, int m, int J, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_backtrack_search::ConfFirst "
				"I=" << I << " m=" << m
				<< " J=" << J << endl;
	}
	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, J);
	int n;


	if (J == 0) {

		int i1;

		i1 = C->i0 + m;

		gg->Test_semicanonical->row_init(I, m, J,
					i1,
					verbose_level);

	}
	n = 0;
	while (TRUE) {
		while (TRUE) {
			if (n >= C->r) {
				if (f_v) {
					cout << "geometric_backtrack_search::ConfFirst "
							"I=" << I << " m=" << m
							<< " J=" << J << " returns TRUE" << endl;
				}
				return TRUE;
			}
			if (!XFirst(I, m, J, n, verbose_level)) {
				break;
			}
			n++;
		}
		// n-th element could not initialize, move on
		while (TRUE) {
			if (n == 0) {
				if (f_v) {
					cout << "geometric_backtrack_search::ConfFirst "
							"I=" << I << " m=" << m
							<< " J=" << J << " returns FALSE" << endl;
				}
				return FALSE;
			}
			n--;
			if (XNext(I, m, J, n, verbose_level)) {
				break;
			}
		}
		// n-th element has been incremented. Initialize elements after it:
		n++;
	}
}

int geometric_backtrack_search::ConfNext(int I, int m, int J, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_backtrack_search::ConfNext "
				"I=" << I << " m=" << m << " J=" << J << endl;
	}
	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, J);
	int n, i1;

	i1 = C->i0 + m;


	gg->Test_semicanonical->row_test_continue(I, m, J, i1);


	if (C->r == 0) {
		return FALSE;
	}
	n = C->r - 1;
	while (TRUE) {
		while (TRUE) {
			if (XNext(I, m, J, n, verbose_level)) {
				break;
			}
			if (n == 0) {
				return FALSE;
				if (f_v) {
					cout << "geometric_backtrack_search::ConfNext "
							"I=" << I << " m=" << m
							<< " J=" << J << " returns FALSE" << endl;
				}
			}
			n--;
		}
		// n-th element has been incremented. Initialize elements after it:
		while (TRUE) {
			if (n >= C->r - 1) {
				if (f_v) {
					cout << "geometric_backtrack_search::ConfNext "
							"I=" << I << " m=" << m
							<< " J=" << J << " returns TRUE" << endl;
				}
				return TRUE;
			}
			n++;
			if (!XFirst(I, m, J, n, verbose_level)) {
				break;
			}
		}
		// n-th element could not initialize, move on
		n--;
	}
}

void geometric_backtrack_search::ConfClear(int I, int m, int J)
{
	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, J);
	int n;

	if (C->r == 0) {
		return;
	}
	for (n = C->r - 1; n >= 0; n--) {
		XClear(I, m, J, n);
	}
}

int geometric_backtrack_search::XFirst(int I, int m, int J, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_backtrack_search::XFirst "
				"I=" << I << " m=" << m << " J=" << J
				<< " n=" << n << endl;
	}
	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, J);
	int i1, j0, r, j;

	i1 = C->i0 + m; // current row
	r = C->r0 + n; // current incidence index
	j0 = C->j0;


	j = gg->Test_semicanonical->row_starter(I, m, J, n,
			i1, j0, r,
			verbose_level);


	int ret;

	ret = X_First(I, m, J, n, j, verbose_level);

	if (f_v) {
		cout << "geometric_backtrack_search::XFirst "
				"I=" << I << " m=" << m << " J=" << J
				<< " n=" << n << " returns " << ret << endl;
	}

	return ret;
}

int geometric_backtrack_search::XNext(int I, int m, int J, int n, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "geometric_backtrack_search::XNext "
				"I=" << I << " m=" << m
				<< " J=" << J << " n=" << n << endl;
	}
	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, J);
	int old_x;
	int fuse_idx, i1, j0, r, j, k;

	fuse_idx = C->fuse_idx;
	i1 = C->i0 + m; // current row
	r = C->r0 + n; // current incidence index
	j0 = C->j0;

	old_x = gg->inc->Encoding->theX_ir(i1, r);

	gg->girth_test_delete_incidence(i1, r, old_x);

	gg->inc->K[old_x]--;
	if (gg->GB->Descr->f_lambda) {
		k = gg->inc->K[old_x];
		gg->inc->theY[old_x][k] = -1;

		gg->decrement_pairs_point(i1, old_x, k);

	}


	gg->Test_semicanonical->col_marker_remove(I, m, J, n,
				i1, j0, r, old_x);



#if 0
	if (!gg->GB->Descr->f_orderly) {

		if (J == 0 && n == 0) {
			if (C->f_last_non_zero_in_fuse) {
				return FALSE;
			}
		}

	}
#endif

	for (j = old_x - j0 + 1; j < C->b; j++) {

		if (TryToPlace(I, m, J, n, j, verbose_level)) {

			gg->Test_semicanonical->marker_move_on(I, m, J, n, j,
					i1, j0, r,
					verbose_level);

			if (f_v) {
				cout << "geometric_backtrack_search::XNext "
						"I=" << I << " m=" << m << " J=" << J
						<< " n=" << n << " j=" << j << " returns TRUE" << endl;
			}
			return TRUE;

		}

	}
	if (f_v) {
		cout << "geometric_backtrack_search::XNext "
				"I=" << I << " m=" << m << " J=" << J
				<< " n=" << n << " returns FALSE" << endl;
	}
	return FALSE;
}

void geometric_backtrack_search::XClear(int I, int m, int J, int n)
{
	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, J);
	int old_x;
	int i1, j0, r, k;

	i1 = C->i0 + m; // current row
	r = C->r0 + n; // index of current incidence within the row
	j0 = C->j0;
	old_x = gg->inc->Encoding->theX_ir(i1, r);
	gg->inc->K[old_x]--;


	gg->girth_test_delete_incidence(i1, r, old_x);



	gg->Test_semicanonical->col_marker_remove(I, m, J, n,
				i1, j0, r, old_x);



	gg->inc->Encoding->theX_ir(i1, r) = -1;

	if (gg->GB->Descr->f_lambda) {

		k = gg->inc->K[old_x];
		gg->inc->theY[old_x][k] = -1;

		gg->decrement_pairs_point(i1, old_x, k);

	}
}

int geometric_backtrack_search::X_First(int I, int m, int J, int n, int j,
		int verbose_level)
// Try placing an incidence, starting from column j and moving to the right
// j is local coordinate
// maintains Decomposition_with_fuse->hbar[], vbar[], f_vbar[][],
// inc->Encoding->theX[][], inc->K[]
{
	int f_v = (verbose_level >= 1);

	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, J);
	int fuse_idx, i1, j0, r;

	fuse_idx = C->fuse_idx;
	i1 = C->i0 + m;
		// current row

	r = C->r0 + n;
		// index of current incidence within the row

	j0 = C->j0;

#if 0
	// f_vbar must be off:
	if (f_vbar[i1 * inc->Encoding->dim_n + r]) {
		cout << "I = " << I << " m = " << m << ", J = " << J
				<< ", n = " << n << ", i1 = " << i1
				<< ", r = " << r << ", j0 = " << j0 << endl;
		cout << "X_Fst f_vbar[i1][r]" << endl;
		exit(1);
	}
#endif

	for (; j < C->b; j++) {

		if (TryToPlace(I, m, J, n, j, verbose_level)) {

			gg->Test_semicanonical->markers_update(I, m, J, n, j,
					i1, j0, r,
					verbose_level);

			if (f_v) {
				cout << "geometric_backtrack_search::X_First "
						"I=" << I << " m=" << m << " J=" << J
						<< " n=" << n << " j=" << j << " returns TRUE" << endl;
			}
			return TRUE;
		}
		// continue with the next choice of j

	} // next j

	return FALSE;
}


int geometric_backtrack_search::TryToPlace(int I, int m, int J, int n, int j,
		int verbose_level)
// Try placing an incidence in column j
// j is local coordinate
// maintains Decomposition_with_fuse->hbar[], vbar[], f_vbar[][],
// inc->Encoding->theX[][], inc->K[]
{
	int f_v = (verbose_level >= 1);

	gen_geo_conf *C = gg->Decomposition_with_fuse->get_conf_IJ(I, J);
	int fuse_idx, i1, j0, j1, r, k;

	fuse_idx = C->fuse_idx;
	i1 = C->i0 + m; // current row

	r = C->r0 + n; // index of current incidence within the row

	j0 = C->j0;

	j1 = j0 + j;

	if (gg->inc->K[j1] >= gg->Decomposition_with_fuse->K1[fuse_idx * gg->GB->b_len + J]) {

		// column j1 is full, move on

		if (f_v) {
			cout << "geometric_backtrack_search::TryToPlace "
					"I=" << I << " m=" << m << " J=" << J
					<< " n=" << n << " j=" << j
					<< " skipped because of column sum" << endl;
		}
		return FALSE;
	}


	if (gg->Test_semicanonical->col_marker_test(j0, j, i1)) {
		return FALSE;
	}

	//gg->inc->Encoding->theX[i1 * gg->inc->Encoding->dim_n + r] = j1;
	gg->inc->Encoding->theX_ir(i1, r) = j1;
	// incidence must be recorded before we call find_square

	k = gg->inc->K[j1];

	gg->inc->theY[j1][k] = i1;

	// and now come the tests:

	if (!gg->apply_tests(I, m, J, n, j, verbose_level - 2)) {
		if (f_v) {
			cout << "geometric_backtrack_search::TryToPlace "
					"I=" << I << " m=" << m << " J=" << J
					<< " n=" << n << " j=" << j <<
					" skipped because of test" << endl;
		}
		return FALSE;
	}

	// the incidence passes the tests:

	// increase column sum:

	gg->inc->K[j1]++;


	return TRUE;

}




}}}

