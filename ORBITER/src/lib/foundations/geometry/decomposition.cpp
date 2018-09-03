// decomposition.C
//
// Anton Betten
//
// December 1, 2012

#include "foundations.h"


decomposition::decomposition()
{
	null();
}

decomposition::~decomposition()
{
	freeself();
}

void decomposition::null()
{
	Inc = 0;
	I = 0;
	Stack = NULL;
	f_has_decomposition = FALSE;
	f_has_row_scheme = FALSE;
	f_has_col_scheme = FALSE;
}

void decomposition::freeself()
{
	if (Inc) {
		FREE_int(Inc);
		}
	if (I) {
		delete I;
		}
	if (Stack) {
		delete Stack;
		}
	if (f_has_decomposition) {
		FREE_int(row_classes);
		FREE_int(row_class_inv);
		FREE_int(col_classes);
		FREE_int(col_class_inv);
		}
	if (f_has_row_scheme) {
		FREE_int(row_scheme);
		}
	if (f_has_col_scheme) {
		FREE_int(col_scheme);
		}
	null();
}

void decomposition::init_inc_and_stack(incidence_structure *Inc, partitionstack *Stack, int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition::init_inc_and_stack" << endl;
		}
	nb_points = Inc->nb_rows;
	nb_blocks = Inc->nb_cols;
	I = Inc;
	decomposition::Stack = Stack;
	if (f_v) {
		cout << "decomposition::init_inc_and_stack done" << endl;
		}
}

void decomposition::init_incidence_matrix(int m, int n, int *M, int verbose_level)
// copies the incidence matrix
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "decomposition::init_incidence_matrix" << endl;
		}
	nb_points = m;
	nb_blocks = n;
	Inc = NEW_int(nb_points * nb_blocks);
	for (i = 0; i < nb_points * nb_blocks; i++) {
		Inc[i] = M[i];
		}
}

void decomposition::setup_default_partition(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition::setup_default_partition" << endl;
		}
	I = NEW_OBJECT(incidence_structure);
	if (f_v) {
		cout << "decomposition::setup_default_partition "
				"before I->init_by_matrix" << endl;
		}
	I->init_by_matrix(nb_points, nb_blocks,
			Inc, 0 /* verbose_level */);
	if (f_v) {
		cout << "decomposition::setup_default_partition "
				"after I->init_by_matrix" << endl;
		}
	Stack = NEW_OBJECT(partitionstack);
	Stack->allocate(nb_points + nb_blocks,
			0 /* verbose_level */);
	Stack->subset_continguous(nb_points, nb_blocks);
	Stack->split_cell(0 /* verbose_level */);
	Stack->sort_cells();
	if (f_v) {
		cout << "decomposition::setup_default_partition done" << endl;
		}
}

void decomposition::compute_TDO(int max_depth, int verbose_level)
// put max_depth = int_MAX if you want full depth
{
	int f_v = (verbose_level >= 1);
	//int depth = int_MAX;

	if (f_v) {
		cout << "decomposition::compute_TDO" << endl;
		}



	if (f_v) {
		cout << "decomposition::compute_TDO "
				"before I->compute_TDO_safe" << endl;
		}
	I->compute_TDO_safe(*Stack, max_depth, verbose_level /*- 2 */);
	if (f_v) {
		cout << "decomposition::compute_TDO "
				"after I->compute_TDO_safe" << endl;
		}

	Stack->allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		0);
	f_has_decomposition = TRUE;

	if (f_v) {
		cout << "decomposition::compute_TDO done" << endl;
		}
		
}

void decomposition::get_row_scheme(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition::get_row_scheme" << endl;
		}
	if (!f_has_decomposition) {
		cout << "decomposition::get_row_scheme "
				"!f_has_decomposition" << endl;
		exit(1);
		}
	f_has_row_scheme = TRUE;
	row_scheme = NEW_int(nb_row_classes * nb_col_classes);
	I->get_row_decomposition_scheme(*Stack, 
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		row_scheme, 0);
	if (f_v) {
		cout << "decomposition::get_row_scheme done" << endl;
		}
}

void decomposition::get_col_scheme(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition::get_col_scheme" << endl;
		}
	if (!f_has_decomposition) {
		cout << "decomposition::get_col_scheme "
				"!f_has_decomposition" << endl;
		exit(1);
		}
	f_has_col_scheme = TRUE;
	col_scheme = NEW_int(nb_row_classes * nb_col_classes);
	I->get_col_decomposition_scheme(*Stack, 
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		col_scheme, 0);
	if (f_v) {
		cout << "decomposition::get_col_scheme done" << endl;
		}
}

void decomposition::print_row_decomposition_tex(
	ostream &ost,
	int f_enter_math, int f_print_subscripts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition::print_row_decomposition_tex" << endl;
		}
	if (!f_has_row_scheme) {
		cout << "decomposition::print_row_decomposition_tex "
				"!f_has_row_scheme" << endl;
		exit(1);
		}
	//I->get_and_print_row_tactical_decomposition_scheme_tex(
	//	file, FALSE /* f_enter_math */, *Stack);
	Stack->print_row_tactical_decomposition_scheme_tex(
		ost, f_enter_math,
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 
		row_scheme, f_print_subscripts);
}

void decomposition::print_column_decomposition_tex(
	ostream &ost,
	int f_enter_math, int f_print_subscripts,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition::print_column_decomposition_tex" << endl;
		}
	//I->get_and_print_column_tactical_decomposition_scheme_tex(
	//	file, FALSE /* f_enter_math */, *Stack);
	Stack->print_column_tactical_decomposition_scheme_tex(
		ost, f_enter_math,
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 
		col_scheme, f_print_subscripts);
}


