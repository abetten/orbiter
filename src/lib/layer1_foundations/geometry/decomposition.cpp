// decomposition.cpp
//
// Anton Betten
//
// December 1, 2012

#include "foundations.h"

using namespace std;



namespace orbiter {
namespace layer1_foundations {
namespace geometry {


decomposition::decomposition()
{
	nb_points = 0;
	nb_blocks = 0;
	N = 0;
	Incma = 0;
	//Enc = NULL;
	//f_has_encoding = false;
	Inc = 0;
	Stack = NULL;
	f_has_decomposition = false;
	Scheme = NULL;
#if 0
	row_classes = NULL;
	row_class_inv = NULL;
	nb_row_classes = 0;
	col_classes = NULL;
	col_class_inv = NULL;
	nb_col_classes = 0;

	SoS_points = NULL;
	SoS_lines = NULL;

	f_has_row_scheme = false;
	row_scheme = NULL;
	f_has_col_scheme = false;
	col_scheme = NULL;
#endif
}



decomposition::~decomposition()
{
	if (Incma) {
		FREE_int(Incma);
	}
#if 0
	if (Inc) {
		FREE_OBJECT(Inc);
	}
#endif
	if (Stack) {
		FREE_OBJECT(Stack);
	}
	if (Scheme) {
		FREE_OBJECT(Scheme);
	}
#if 0
	if (f_has_decomposition) {
		FREE_int(row_classes);
		FREE_int(row_class_inv);
		FREE_int(col_classes);
		FREE_int(col_class_inv);
	}

	if (SoS_points) {
		FREE_OBJECT(SoS_points);
	}
	if (SoS_lines) {
		FREE_OBJECT(SoS_lines);
	}

	if (f_has_row_scheme) {
		FREE_int(row_scheme);
	}
	if (f_has_col_scheme) {
		FREE_int(col_scheme);
	}
#endif
}

void decomposition::init_incidence_structure(
		incidence_structure *Inc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition::init_incidence_structure" << endl;
	}


	decomposition::Inc = Inc;
	nb_points = Inc->nb_rows;
	nb_blocks = Inc->nb_cols;
	N = nb_points + nb_blocks;

#if 0
	incidence_structure *I;
	data_structures::partitionstack *Stack;
	int depth = INT_MAX;

	I = NEW_OBJECT(incidence_structure);
	I->init_projective_space(P, verbose_level);
#endif

	Stack = NEW_OBJECT(data_structures::partitionstack);
	Stack->allocate(Inc->nb_rows + Inc->nb_cols, 0 /* verbose_level */);
	Stack->subset_contiguous(Inc->nb_rows, Inc->nb_cols);
	Stack->split_cell(0 /* verbose_level */);
	Stack->sort_cells();

#if 0
	int *the_points;
	int *the_lines;
	int i;
	the_points = NEW_int(nb_points);
	the_lines = NEW_int(nb_lines);

	for (i = 0; i < nb_points; i++) {
		the_points[i] = points[i];
	}
	for (i = 0; i < nb_lines; i++) {
		the_lines[i] = lines[i];
	}

	Stack->split_cell_front_or_back(
			the_points, nb_points, true /* f_front*/,
			verbose_level);

	Stack->split_line_cell_front_or_back(
			the_lines, nb_lines, true /* f_front*/,
			verbose_level);


	FREE_int(the_points);
	FREE_int(the_lines);
#endif

#if 0
	if (f_v) {
		cout << "decomposition::init_incidence_structure "
				"before I->compute_TDO_safe_and_write_files" << endl;
	}
	I->compute_TDO_safe_and_write_files(
			*Stack, N /*depth*/, fname_base, file_names, verbose_level);
	if (f_v) {
		cout << "decomposition::init_incidence_structure "
				"after I->compute_TDO_safe_and_write_files" << endl;
	}
#endif


	if (f_v) {
		cout << "decomposition::init_incidence_structure done" << endl;
	}
}


void decomposition::init_inc_and_stack(
		incidence_structure *Inc,
		data_structures::partitionstack *Stack,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition::init_inc_and_stack" << endl;
	}
	nb_points = Inc->nb_rows;
	nb_blocks = Inc->nb_cols;
	N = nb_points + nb_blocks;
	decomposition::Inc = Inc;
	decomposition::Stack = Stack;
	if (f_v) {
		cout << "decomposition::init_inc_and_stack done" << endl;
	}
}

void decomposition::init_incidence_matrix(
		int m, int n, int *M, int verbose_level)
// copies the incidence matrix
{
	int f_v = (verbose_level >= 1);
	int i;

	if (f_v) {
		cout << "decomposition::init_incidence_matrix" << endl;
	}
	nb_points = m;
	nb_blocks = n;
	N = nb_points + nb_blocks;
	Incma = NEW_int(nb_points * nb_blocks);
	for (i = 0; i < nb_points * nb_blocks; i++) {
		Incma[i] = M[i];
	}
}

#if 0
void decomposition::init_encoding(
		combinatorics::encoded_combinatorial_object *Enc,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition::init_encoding" << endl;
	}

	decomposition::Enc = Enc;
	f_has_encoding = true;

	I = NEW_OBJECT(geometry::incidence_structure);

	if (f_v) {
		cout << "decomposition::init_encoding "
				"before Inc->init_by_matrix" << endl;
	}
	I->init_by_matrix(
			Enc->nb_rows,
			Enc->nb_cols,
			Enc->get_Incma(), 0 /* verbose_level*/);
	if (f_v) {
		cout << "decomposition::init_encoding "
				"after Inc->init_by_matrix" << endl;
	}
	nb_points = I->nb_points();
	nb_blocks = I->nb_lines();
	N = nb_points + nb_blocks;



	Stack = NEW_OBJECT(data_structures::partitionstack);

	if (f_v) {
		cout << "decomposition::init_encoding "
				"allocating partitionstack" << endl;
	}

	Stack->allocate(N, 0);
	// split off the column class:
	Stack->subset_contiguous(I->nb_points(), I->nb_lines());
	Stack->split_cell(0);

	if (f_v) {
		cout << "decomposition::init_encoding "
				"done" << endl;
	}
}
#endif

void decomposition::compute_TDO_deep(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition::compute_TDO_deep" << endl;
	}

	int TDO_depth;

	TDO_depth = Inc->nb_points() + Inc->nb_lines();


	if (f_v) {
		cout << "decomposition::compute_TDO_deep "
				"before compute_TDO_safe" << endl;
	}
	compute_TDO_safe(TDO_depth, verbose_level - 3);
	//TDO_ht = S.ht;
	if (f_v) {
		cout << "decomposition::compute_TDO_deep "
				"after compute_TDO_safe" << endl;
	}

	if (f_v) {
		cout << "decomposition::compute_TDO_deep done" << endl;
	}
}

void decomposition::compute_the_decomposition(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition::compute_the_decomposition" << endl;
	}


	Scheme = NEW_OBJECT(decomposition_scheme);

	if (f_v) {
		cout << "decomposition::compute_the_decomposition "
				"before Scheme->init_row_and_col_schemes" << endl;
	}
	Scheme->init_row_and_col_schemes(this, verbose_level - 1);
	if (f_v) {
		cout << "decomposition::compute_the_decomposition "
				"after Scheme->init_row_and_col_schemes" << endl;
	}
	f_has_decomposition = true;


#if 0
	ost << "Point orbits:\\\\" << endl;
	SoS_points->print_table_tex(ost);

	ost << "Line orbits:\\\\" << endl;
	SoS_lines->print_table_tex(ost);
#endif


	if (f_v) {
		cout << "decomposition::compute_the_decomposition done" << endl;
	}

}

void decomposition::setup_default_partition(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition::setup_default_partition" << endl;
	}
	Inc = NEW_OBJECT(incidence_structure);
	if (f_v) {
		cout << "decomposition::setup_default_partition "
				"before I->init_by_matrix" << endl;
	}
	Inc->init_by_matrix(nb_points, nb_blocks,
			Incma, 0 /* verbose_level */);
	if (f_v) {
		cout << "decomposition::setup_default_partition "
				"after I->init_by_matrix" << endl;
	}
	Stack = NEW_OBJECT(data_structures::partitionstack);
	Stack->allocate(nb_points + nb_blocks,
			0 /* verbose_level */);
	Stack->subset_contiguous(nb_points, nb_blocks);
	Stack->split_cell(0 /* verbose_level */);
	Stack->sort_cells();
	if (f_v) {
		cout << "decomposition::setup_default_partition done" << endl;
	}
}

void decomposition::compute_TDO(
		int max_depth, int verbose_level)
// put max_depth = INT_MAX if you want full depth
{
	int f_v = (verbose_level >= 1);
	//int depth = INT_MAX;

	if (f_v) {
		cout << "decomposition::compute_TDO" << endl;
	}



	if (f_v) {
		cout << "decomposition::compute_TDO "
				"before compute_TDO_safe" << endl;
	}
	compute_TDO_safe(max_depth, verbose_level /*- 2 */);
	if (f_v) {
		cout << "decomposition::compute_TDO "
				"after compute_TDO_safe" << endl;
	}

	if (f_v) {
		cout << "decomposition::compute_TDO "
				"before compute_the_decomposition" << endl;
	}
	compute_the_decomposition(verbose_level - 1);
	if (f_v) {
		cout << "decomposition::compute_TDO "
				"after compute_the_decomposition" << endl;
	}

#if 0
	Stack->allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes, 
		0);
	f_has_decomposition = true;
#endif

	if (f_v) {
		cout << "decomposition::compute_TDO done" << endl;
	}
		
}

void decomposition::get_row_scheme(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition::get_row_scheme" << endl;
	}
	Scheme->get_row_scheme(verbose_level - 1);
	if (f_v) {
		cout << "decomposition::get_row_scheme done" << endl;
	}
}

void decomposition::get_col_scheme(
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition::get_col_scheme" << endl;
	}
	Scheme->get_col_scheme(verbose_level - 1);
	if (f_v) {
		cout << "decomposition::get_col_scheme done" << endl;
	}
}


void decomposition::compute_TDO_safe(
	int depth, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int f_refine, f_refine_prev;
	int i;

	if (f_v) {
		cout << "decomposition::compute_TDO_safe" << endl;
	}

	f_refine_prev = true;
	for (i = 0; i < depth; i++) {
		if (f_v) {
			cout << "decomposition::compute_TDO_safe i = " << i << endl;
		}


		if (EVEN(i)) {
			if (f_v) {
				cout << "decomposition::compute_TDO_safe "
						"before refine_column_partition_safe" << endl;
			}
			f_refine = refine_column_partition_safe(verbose_level - 2);
			if (f_v) {
				cout << "decomposition::compute_TDO_safe "
						"after refine_column_partition_safe" << endl;
			}
		}
		else {
			if (f_v) {
				cout << "decomposition::compute_TDO_safe "
						"before refine_row_partition_safe" << endl;
			}
			f_refine = refine_row_partition_safe(verbose_level - 2);
			if (f_v) {
				cout << "incidence_structure::compute_TDO_safe "
						"after refine_row_partition_safe" << endl;
			}
		}

		if (f_v) {
			cout << "decomposition::compute_TDO_safe "
					"i=" << i << " after refine" << endl;
			if (EVEN(i)) {
				int f_list_incidences = false;
				get_and_print_col_decomposition_scheme(
						f_list_incidences, false,
						verbose_level);
				Stack->print_classes_points_and_lines(cout);
			}
			else {
				int f_list_incidences = false;
				get_and_print_row_decomposition_scheme(
						f_list_incidences, false,
						verbose_level);
				Stack->print_classes_points_and_lines(cout);
			}
		}

		if (!f_refine_prev && !f_refine) {
			if (f_v) {
				cout << "decomposition::compute_TDO_safe "
						"no refinement, we are done" << endl;
			}
			goto done;
		}
		f_refine_prev = f_refine;
	}

done:
	if (f_v) {
		cout << "decomposition::compute_TDO_safe done" << endl;
	}

}


void decomposition::compute_TDO_safe_and_write_files(
	int depth,
	std::string &fname_base,
	std::vector<std::string> &file_names,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);
	//int f_vv = (verbose_level >= 2);
	//int f_vvv = (verbose_level >= 3);
	int f_refine, f_refine_prev;
	int i;

	if (f_v) {
		cout << "decomposition::compute_TDO_safe_and_write_files" << endl;
	}

	f_refine_prev = true;
	for (i = 0; i < depth; i++) {
		if (f_v) {
			cout << "decomposition::compute_TDO_safe_and_write_files "
					"i = " << i << endl;
		}


		string fname;

		fname = fname_base + std::to_string(i);
		file_names.push_back(fname);


		if (EVEN(i)) {
			if (f_v) {
				cout << "decomposition::compute_TDO_safe_and_write_files "
						"before refine_column_partition_safe" << endl;
			}
			f_refine = refine_column_partition_safe(
					verbose_level - 2);
			if (f_v) {
				cout << "decomposition::compute_TDO_safe_and_write_files "
						"after refine_column_partition_safe" << endl;
			}

			{
				ofstream ost(fname);

				get_and_print_column_tactical_decomposition_scheme_tex(
					ost, false /* f_enter_math */,
					true /* f_print_subscripts */);
			}
		}
		else {
			if (f_v) {
				cout << "decomposition::compute_TDO_safe_and_write_files "
						"before refine_row_partition_safe" << endl;
			}
			f_refine = refine_row_partition_safe(
					verbose_level - 2);
			if (f_v) {
				cout << "decomposition::compute_TDO_safe_and_write_files "
						"after refine_row_partition_safe" << endl;
			}
			{
				ofstream ost(fname);

				get_and_print_row_tactical_decomposition_scheme_tex(
						ost, false /* f_enter_math */,
						true /* f_print_subscripts */);
			}
		}

		if (f_v) {
			cout << "decomposition::compute_TDO_safe_and_write_files "
					"i=" << i << " after refine" << endl;
			if (EVEN(i)) {
				int f_list_incidences = false;
				get_and_print_col_decomposition_scheme(
						f_list_incidences, false, verbose_level);
				Stack->print_classes_points_and_lines(cout);
			}
			else {
				int f_list_incidences = false;
				get_and_print_row_decomposition_scheme(
						f_list_incidences, false, verbose_level);
				Stack->print_classes_points_and_lines(cout);
			}
		}

		if (!f_refine_prev && !f_refine) {
			if (f_v) {
				cout << "decomposition::compute_TDO_safe_and_write_files "
						"no refinement, we are done" << endl;
			}
			goto done;
		}
		f_refine_prev = f_refine;
	}

done:
	if (f_v) {
		cout << "decomposition::compute_TDO_safe_and_write_files done" << endl;
	}

}


int decomposition::refine_column_partition_safe(
		int verbose_level)
{

	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 5);
	int i, j, c, h, I, ht, first, next, N;
	int *row_classes;
	int *row_class_idx;
	int nb_row_classes;
	int *col_classes;
	int *col_class_idx;
	int nb_col_classes;

	int *data;
	int *neighbors;

	if (f_v) {
		cout << "decomposition::refine_column_partition_safe" << endl;
	}
	if (f_v) {
		cout << "decomposition::refine_column_partition_safe "
				"Stack->ht" << Stack->ht << endl;
	}
	row_classes = NEW_int(Stack->ht);
	col_classes = NEW_int(Stack->ht);
	row_class_idx = NEW_int(Stack->ht);
	col_class_idx = NEW_int(Stack->ht);

	if (f_v) {
		cout << "decomposition::refine_column_partition_safe "
				"before get_partition" << endl;
	}
	get_partition(
			row_classes, row_class_idx, nb_row_classes,
			col_classes, col_class_idx, nb_col_classes);
	if (f_v) {
		cout << "decomposition::refine_column_partition_safe "
				"after get_partition" << endl;
	}

	N = Inc->nb_points() + Inc->nb_lines();
	if (f_v) {
		cout << "decomposition::refine_column_partition_safe "
				"nb_row_classes= " << nb_row_classes << endl;
	}
	data = NEW_int(N * nb_row_classes);
	Int_vec_zero(data, N * nb_row_classes);

	neighbors = NEW_int(Inc->max_k);

	for (j = 0; j < Inc->nb_lines(); j++) {
		Inc->get_points_on_line(neighbors, j, 0 /*verbose_level - 2*/);
		for (h = 0; h < Inc->nb_points_on_line[j]; h++) {
			i = neighbors[h];
			c = Stack->cellNumber[Stack->invPointList[i]];
			I = row_class_idx[c];
			if (I == -1) {
				cout << "decomposition::refine_column_partition_safe "
						"I == -1" << endl;
				exit(1);
			}
			data[(Inc->nb_points() + j) * nb_row_classes + I]++;
		}
	}
	if (f_vv) {
		cout << "decomposition::refine_column_partition_safe "
				"data:" << endl;
		Int_vec_print_integer_matrix_width(cout,
			data + Inc->nb_points() * nb_row_classes,
			Inc->nb_lines(), nb_row_classes, nb_row_classes, 3);
	}

	ht = Stack->ht;

	for (c = 0; c < ht; c++) {
		if (Stack->is_row_class(c)) {
			continue;
		}

		if (Stack->cellSize[c] == 1) {
			continue;
		}
		first = Stack->startCell[c];
		next = first + Stack->cellSize[c];

		Stack->radix_sort(first /* left */,
				   next - 1 /* right */,
				   data, nb_row_classes, 0 /*radix*/, false);
	}

	FREE_int(data);
	FREE_int(neighbors);

	FREE_int(row_classes);
	FREE_int(col_classes);
	FREE_int(row_class_idx);
	FREE_int(col_class_idx);
	if (f_v) {
		cout << "decomposition::refine_column_partition_safe done" << endl;
	}
	if (Stack->ht == ht) {
		return false;
	}
	else {
		return true;
	}
}

int decomposition::refine_row_partition_safe(
		int verbose_level)
{

	int f_v = (verbose_level >= 1);
	int f_vv = (verbose_level >= 5);
	int i, j, c, h, J, ht, first, next;
	int *row_classes;
	int *row_class_idx;
	int nb_row_classes;
	int *col_classes;
	int *col_class_idx;
	int nb_col_classes;

	int *data;
	int *neighbors;

	if (f_v) {
		cout << "decomposition::refine_row_partition_safe" << endl;
	}
	row_classes = NEW_int(Stack->ht);
	col_classes = NEW_int(Stack->ht);
	row_class_idx = NEW_int(Stack->ht);
	col_class_idx = NEW_int(Stack->ht);

	if (f_v) {
		cout << "decomposition::refine_row_partition_safe "
				"before get_partition" << endl;
	}
	get_partition(
			row_classes, row_class_idx, nb_row_classes,
			col_classes, col_class_idx, nb_col_classes);
	if (f_v) {
		cout << "decomposition::refine_row_partition_safe "
				"after get_partition" << endl;
	}
	if (f_v) {
		cout << "decomposition::refine_row_partition_safe "
				"nb_col_classes=" << nb_col_classes << endl;
	}

	data = NEW_int(Inc->nb_points() * nb_col_classes);
	Int_vec_zero(data, Inc->nb_points() * nb_col_classes);


	neighbors = NEW_int(Inc->max_r);

	for (i = 0; i < Inc->nb_points(); i++) {
		Inc->get_lines_on_point(neighbors, i, 0 /*verbose_level - 2*/);
		for (h = 0; h < Inc->nb_lines_on_point[i]; h++) {
			j = neighbors[h] + Inc->nb_points();
			c = Stack->cellNumber[Stack->invPointList[j]];
			J = col_class_idx[c];
			if (J == -1) {
				cout << "decomposition::refine_row_partition_safe "
						"J == -1" << endl;
				exit(1);
			}
			data[i * nb_col_classes + J]++;
		}
	}
	if (f_vv) {
		cout << "decomposition::refine_row_partition_safe "
				"data:" << endl;
		Int_vec_print_integer_matrix_width(cout, data, Inc->nb_points(),
				nb_col_classes, nb_col_classes, 3);
	}

	ht = Stack->ht;

	for (c = 0; c < ht; c++) {
		if (Stack->is_col_class(c)) {
			continue;
		}

		if (Stack->cellSize[c] == 1) {
			continue;
		}
		first = Stack->startCell[c];
		next = first + Stack->cellSize[c];

		Stack->radix_sort(first /* left */,
				   next - 1 /* right */,
				   data, nb_col_classes, 0 /*radix*/, 0);
	}

	FREE_int(data);
	FREE_int(neighbors);

	FREE_int(row_classes);
	FREE_int(col_classes);
	FREE_int(row_class_idx);
	FREE_int(col_class_idx);
	if (f_v) {
		cout << "decomposition::refine_row_partition_safe done" << endl;
	}
	if (Stack->ht == ht) {
		return false;
	}
	else {
		return true;
	}
}

void decomposition::get_partition(
	int *row_classes, int *row_class_idx, int &nb_row_classes,
	int *col_classes, int *col_class_idx, int &nb_col_classes)
{
	int i, c;

	Stack->get_row_and_col_classes(
		row_classes, nb_row_classes,
		col_classes, nb_col_classes, 0 /*verbose_level*/);
	for (i = 0; i < Stack->ht; i++) {
		row_class_idx[i] = col_class_idx[i] = -1;
	}
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		row_class_idx[c] = i;
	}
	for (i = 0; i < nb_col_classes; i++) {
		c = col_classes[i];
		col_class_idx[c] = i;
	}
}

void decomposition::get_and_print_row_decomposition_scheme(
	int f_list_incidences,
	int f_local_coordinates, int verbose_level)
{
	int *row_classes, *row_class_inv, nb_row_classes;
	int *col_classes, *col_class_inv, nb_col_classes;
	int *row_scheme;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition::get_and_print_row_decomposition_scheme "
				"computing col scheme" << endl;
	}
	Stack->allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
		0);

	row_scheme = NEW_int(nb_row_classes * nb_col_classes);

	get_row_decomposition_scheme(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
		row_scheme, 0);

	//cout << *this << endl;

	if (f_v) {
		cout << "row_scheme:" << endl;
		Stack->print_decomposition_scheme(cout,
			row_classes, nb_row_classes,
			col_classes, nb_col_classes,
			row_scheme);
	}

	if (f_list_incidences) {
		cout << "incidences by row-scheme:" << endl;
		print_row_tactical_decomposition_scheme_incidences_tex(
			cout, false /* f_enter_math_mode */,
			row_classes, row_class_inv, nb_row_classes,
			col_classes, col_class_inv, nb_col_classes,
			f_local_coordinates, 0 /*verbose_level*/);
	}

	FREE_int(row_classes);
	FREE_int(row_class_inv);
	FREE_int(col_classes);
	FREE_int(col_class_inv);
	FREE_int(row_scheme);
}

void decomposition::get_and_print_col_decomposition_scheme(
	int f_list_incidences,
	int f_local_coordinates, int verbose_level)
{
	int *row_classes, *row_class_inv, nb_row_classes;
	int *col_classes, *col_class_inv, nb_col_classes;
	int *col_scheme;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition::get_and_print_col_decomposition_scheme "
				"computing col scheme" << endl;
	}
	Stack->allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
		verbose_level);

	col_scheme = NEW_int(nb_row_classes * nb_col_classes);

	get_col_decomposition_scheme(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
		col_scheme, 0 /*verbose_level*/);

	//cout << *this << endl;

	if (f_v) {
		cout << "col_scheme:" << endl;
		Stack->print_decomposition_scheme(cout,
			row_classes, nb_row_classes,
			col_classes, nb_col_classes,
			col_scheme);
	}

	if (f_list_incidences) {
		cout << "incidences by col-scheme:" << endl;
		print_col_tactical_decomposition_scheme_incidences_tex(
			cout, false /* f_enter_math_mode */,
			row_classes, row_class_inv, nb_row_classes,
			col_classes, col_class_inv, nb_col_classes,
			f_local_coordinates, 0 /*verbose_level*/);
	}

	FREE_int(row_classes);
	FREE_int(row_class_inv);
	FREE_int(col_classes);
	FREE_int(col_class_inv);
	FREE_int(col_scheme);
}

void decomposition::get_row_decomposition_scheme(
	int *row_classes, int *row_class_inv, int nb_row_classes,
	int *col_classes, int *col_class_inv, int nb_col_classes,
	int *row_scheme, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int I, J, i, j, c1, f1, l1, x, y, u, c, nb;
	int *neighbors;
	int *data0;
	int *data1;

	if (f_v) {
		cout << "decomposition::get_row_decomposition_scheme" << endl;
	}
	neighbors = NEW_int(Inc->max_r);
	data0 = NEW_int(nb_col_classes);
	data1 = NEW_int(nb_col_classes);
	Int_vec_zero(row_scheme, nb_row_classes * nb_col_classes);

	for (I = 0; I < nb_row_classes; I++) {
		c1 = row_classes[I];
		f1 = Stack->startCell[c1];
		l1 = Stack->cellSize[c1];
		Int_vec_zero(data0, nb_col_classes);

		for (i = 0; i < l1; i++) {

			x = Stack->pointList[f1 + i];
			Int_vec_zero(data1, nb_col_classes);
			nb = Inc->get_lines_on_point(
					neighbors, x, verbose_level - 2);

			for (u = 0; u < nb; u++) {
				y = neighbors[u];
				j = Inc->nb_points() + y;
				c = Stack->cellNumber[Stack->invPointList[j]];
				J = col_class_inv[c];
				data1[J]++;
			}
			if (i == 0) {
				Int_vec_copy(data1, data0, nb_col_classes);
			}
			else {
				for (J = 0; J < nb_col_classes; J++) {
					if (data0[J] != data1[J]) {
						cout << "decomposition::get_row_decomposition_scheme "
								"not row-tactical I=" << I << " i=" << i
								<< " J=" << J << endl;
					}
				}
			}
		} // next i

		Int_vec_copy(data0,
				row_scheme + I * nb_col_classes,
				nb_col_classes);
	}
	FREE_int(neighbors);
	FREE_int(data0);
	FREE_int(data1);
	if (f_v) {
		cout << "decomposition::get_row_decomposition_scheme done" << endl;
	}
}

void decomposition::get_row_decomposition_scheme_if_possible(
	int *row_classes, int *row_class_inv, int nb_row_classes,
	int *col_classes, int *col_class_inv, int nb_col_classes,
	int *row_scheme, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int I, J, i, j, c1, f1, l1, x, y, u, c, nb;
	int *neighbors;
	int *data0;
	int *data1;

	if (f_v) {
		cout << "decomposition::get_row_decomposition_scheme_if_possible" << endl;
	}
	neighbors = NEW_int(Inc->max_r);
	data0 = NEW_int(nb_col_classes);
	data1 = NEW_int(nb_col_classes);
	Int_vec_zero(row_scheme, nb_row_classes * nb_col_classes);

	for (I = 0; I < nb_row_classes; I++) {
		c1 = row_classes[I];
		f1 = Stack->startCell[c1];
		l1 = Stack->cellSize[c1];
		Int_vec_zero(data0, nb_col_classes);

		for (i = 0; i < l1; i++) {
			x = Stack->pointList[f1 + i];
			Int_vec_zero(data1, nb_col_classes);

			nb = Inc->get_lines_on_point(
					neighbors, x, verbose_level - 2);

			for (u = 0; u < nb; u++) {
				y = neighbors[u];
				j = Inc->nb_points() + y;
				c = Stack->cellNumber[Stack->invPointList[j]];
				J = col_class_inv[c];
				data1[J]++;
			}
			if (i == 0) {
				Int_vec_copy(data1, data0, nb_col_classes);
			}
			else {
				for (J = 0; J < nb_col_classes; J++) {
					if (data0[J] != data1[J]) {
						data0[J] = -1;
						//cout << "not row-tactical I=" << I
						//<< " i=" << i << " J=" << J << endl;
					}
				}
			}
		} // next i
		for (J = 0; J < nb_col_classes; J++) {
			row_scheme[I * nb_col_classes + J] = data0[J];
		}
	}
	FREE_int(neighbors);
	FREE_int(data0);
	FREE_int(data1);
}

void decomposition::get_col_decomposition_scheme(
	int *row_classes, int *row_class_inv, int nb_row_classes,
	int *col_classes, int *col_class_inv, int nb_col_classes,
	int *col_scheme, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int I, J, j, c1, f1, l1, x, y, u, c, nb;
	int *neighbors;
	int *data0;
	int *data1;

	if (f_v) {
		cout << "decomposition::get_col_decomposition_scheme" << endl;
	}
	neighbors = NEW_int(Inc->max_k);
	data0 = NEW_int(nb_row_classes);
	data1 = NEW_int(nb_row_classes);
	Int_vec_zero(col_scheme, nb_row_classes * nb_col_classes);

	for (J = 0; J < nb_col_classes; J++) {
		c1 = col_classes[J];
		f1 = Stack->startCell[c1];
		l1 = Stack->cellSize[c1];
		Int_vec_zero(data0, nb_row_classes);

		for (j = 0; j < l1; j++) {
			y = Stack->pointList[f1 + j] - Inc->nb_points();
			Int_vec_zero(data1, nb_row_classes);

			nb = Inc->get_points_on_line(
					neighbors, y, verbose_level - 2);

			for (u = 0; u < nb; u++) {
				x = neighbors[u];
				c = Stack->cellNumber[Stack->invPointList[x]];
				I = row_class_inv[c];
				data1[I]++;
			}
			if (j == 0) {
				Int_vec_copy(data1, data0, nb_row_classes);
#if 0
				for (I = 0; I < nb_row_classes; I++) {
					data0[I] = data1[I];
				}
#endif
			}
			else {
				for (I = 0; I < nb_row_classes; I++) {
					if (data0[I] != data1[I]) {
						cout << "not col-tactical J=" << J
								<< " j=" << j << " I=" << I << endl;
					}
				}
			}
		} // next j
		for (I = 0; I < nb_row_classes; I++) {
			col_scheme[I * nb_col_classes + J] = data0[I];
		}
	}
	FREE_int(neighbors);
	FREE_int(data0);
	FREE_int(data1);
}

void decomposition::row_scheme_to_col_scheme(
	int *row_classes, int *row_class_inv, int nb_row_classes,
	int *col_classes, int *col_class_inv, int nb_col_classes,
	int *row_scheme, int *col_scheme, int verbose_level)
{
	int I, J, c1, l1, c2, l2, a, b, c;

	for (I = 0; I < nb_row_classes; I++) {
		c1 = row_classes[I];
		l1 = Stack->cellSize[c1];
		for (J = 0; J < nb_col_classes; J++) {
			c2 = col_classes[J];
			l2 = Stack->cellSize[c2];
			a = row_scheme[I * nb_col_classes + J];
			b = a * l1;
			if (b % l2) {
				cout << "decomposition::row_scheme_to_col_scheme: "
						"not tactical" << endl;
				exit(1);
			}
			c = b / l2;
			col_scheme[I * nb_col_classes + J] = c;
		}
	}
}

void decomposition::print_row_tactical_decomposition_scheme_incidences_tex(
	std::ostream &ost, int f_enter_math_mode,
	int *row_classes, int *row_class_inv, int nb_row_classes,
	int *col_classes, int *col_class_inv, int nb_col_classes,
	int f_local_coordinates, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *incidences;
	int c1, c2, f1, f2, l1; //, l2;
	int i, j, rij;
	int u, v, x, a, b, c, J;
	int *row_scheme;
	int *S;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "decomposition::print_row_tactical_decomposition_scheme_incidences_tex" << endl;
	}

	row_scheme = NEW_int(nb_row_classes * nb_col_classes);

	get_row_decomposition_scheme(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
		row_scheme, verbose_level - 2);


	for (i = 0; i < nb_row_classes; i++) {
		c1 = row_classes[i];
		f1 = Stack->startCell[c1];
		l1 = Stack->cellSize[c1];

		for (j = 0; j < nb_col_classes; j++) {

			rij = row_scheme[i * nb_col_classes + j];

			if (rij == 0) {
				continue;
			}

			S = NEW_int(rij);
			get_incidences_by_row_scheme(
				row_classes, row_class_inv, nb_row_classes,
				col_classes, col_class_inv, nb_col_classes,
				i, j,
				rij, incidences, verbose_level - 2);

			c2 = col_classes[j];
			f2 = Stack->startCell[c2];
			//l2 = PStack.cellSize[c2];

			ost << "\\subsubsection*{Row class " << i
					<< " (cell " << c1 << ") vs. col class "
					<< j << " (cell " << c2 << ")";
			if (f_local_coordinates) {
				ost << " (in local coordinates)";
			}
			ost << ", $r_{" << i << ", " << j << "}=" << rij << "$}" << endl;
			//ost << "f1=" << f1 << " l1=" << l1 << endl;
			//ost << "f2=" << f2 << " l2=" << l2 << endl;
			for (u = 0; u < l1; u++) {
				x = Stack->pointList[f1 + u];
				ost << setw(4) << u << " : $P_{" << setw(4) << x
						<< "}$ is incident with ";
				for (v = 0; v < rij; v++) {
					a = incidences[u * rij + v];
					if (f_local_coordinates) {
						b = Inc->nb_points() + a;
						c = Stack->cellNumber[Stack->invPointList[b]];
						J = col_class_inv[c];
						if (J != j) {
							cout << "decomposition::print_row_tactical_decomposition_scheme_incidences_tex J != j" << endl;
							cout << "j=" << j << endl;
							cout << "J=" << J << endl;
						}
						a = Stack->invPointList[b] - f2;
					}
					S[v] = a;
					//ost << a << " ";
				}
				Sorting.int_vec_heapsort(S, rij);
				ost << "$\\{";
				for (v = 0; v < rij; v++) {
					ost << "\\ell_{" << setw(4) << S[v] << "}";
					if (v < rij - 1) {
						ost << ", ";
					}
				}
				ost << "\\}$";

				ost << "\\\\" << endl;
			}

			FREE_int(incidences);
			FREE_int(S);
		} // next j
	} // next i

	FREE_int(row_scheme);
}

void decomposition::print_col_tactical_decomposition_scheme_incidences_tex(
	std::ostream &ost, int f_enter_math_mode,
	int *row_classes, int *row_class_inv, int nb_row_classes,
	int *col_classes, int *col_class_inv, int nb_col_classes,
	int f_local_coordinates, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int *incidences;
	int c1, c2, f1, f2, /*l1,*/ l2;
	int i, j, kij;
	int u, v, y, a, b, c, I;
	int *col_scheme;
	int *S;
	data_structures::sorting Sorting;

	if (f_v) {
		cout << "decomposition::print_col_tactical_decomposition_scheme_incidences_tex" << endl;
	}

	col_scheme = NEW_int(nb_row_classes * nb_col_classes);

	get_col_decomposition_scheme(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
		col_scheme, verbose_level - 2);


	for (i = 0; i < nb_row_classes; i++) {
		c1 = row_classes[i];
		f1 = Stack->startCell[c1];
		//l1 = PStack.cellSize[c1];

		for (j = 0; j < nb_col_classes; j++) {

			kij = col_scheme[i * nb_col_classes + j];

			if (kij == 0) {
				continue;
			}

			S = NEW_int(kij);
			get_incidences_by_col_scheme(
				row_classes, row_class_inv, nb_row_classes,
				col_classes, col_class_inv, nb_col_classes,
				i, j,
				kij, incidences, verbose_level - 2);

			c2 = col_classes[j];
			f2 = Stack->startCell[c2];
			l2 = Stack->cellSize[c2];

			ost << "\\subsubsection*{Row class " << i
					<< " (cell " << c1 << ") vs. col class "
					<< j << " (cell " << c2 << ")";
			if (f_local_coordinates) {
				ost << " (in local coordinates)";
			}
			ost << ", $k_{" << i << ", " << j << "}="
					<< kij << "$}" << endl;
			//ost << "f1=" << f1 << " l1=" << l1 << endl;
			//ost << "f2=" << f2 << " l2=" << l2 << endl;
			for (u = 0; u < l2; u++) {
				y = Stack->pointList[f2 + u] - Inc->nb_points();
				ost << setw(4) << u << " : $\\ell_{" << setw(4)
						<< y << "}$ is incident with ";
				for (v = 0; v < kij; v++) {
					a = incidences[u * kij + v];
					if (f_local_coordinates) {
						b = a;
						c = Stack->cellNumber[Stack->invPointList[b]];
						I = row_class_inv[c];
						if (I != i) {
							cout << "decomposition::print_col_tactical_decomposition_scheme_incidences_tex I != i" << endl;
							cout << "i=" << i << endl;
							cout << "I=" << I << endl;
						}
						a = Stack->invPointList[b] - f1;
					}
					S[v] = a;
					//ost << a << " ";
				}
				Sorting.int_vec_heapsort(S, kij);
				ost << "$\\{";
				for (v = 0; v < kij; v++) {
					ost << "P_{" << setw(4) << S[v] << "}";
					if (v < kij - 1) {
						ost << ", ";
					}
				}
				ost << "\\}$";

				ost << "\\\\" << endl;
			}

			FREE_int(incidences);
			FREE_int(S);
		} // next j
	} // next i

	FREE_int(col_scheme);
}

void decomposition::get_incidences_by_row_scheme(
	int *row_classes, int *row_class_inv, int nb_row_classes,
	int *col_classes, int *col_class_inv, int nb_col_classes,
	int row_class_idx, int col_class_idx,
	int rij, int *&incidences, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int c1, c2, f1, /*f2,*/ l1, /*l2,*/ x, nb, u, y, c, i, j;
	int *sz;
	int *neighbors;

	if (f_v) {
		cout << "decomposition::get_incidences_by_row_scheme" << endl;
	}
	c1 = row_classes[row_class_idx];
	f1 = Stack->startCell[c1];
	l1 = Stack->cellSize[c1];
	c2 = col_classes[col_class_idx];
	//f2 = PStack.startCell[c2];
	//l2 = PStack.cellSize[c2];
	incidences = NEW_int(l1 * rij);
	neighbors = NEW_int(Inc->max_r);
	sz = NEW_int(l1);
	for (i = 0; i < l1; i++) {
		sz[i] = 0;
	}
	for (i = 0; i < l1; i++) {
		x = Stack->pointList[f1 + i];
		nb = Inc->get_lines_on_point(
				neighbors, x, verbose_level - 2);
		//O.lines_on_point_by_line_rank(x, neighbors, verbose_level - 2);
		for (u = 0; u < nb; u++) {
			y = neighbors[u];
			j = Inc->nb_points() + y;
			c = Stack->cellNumber[Stack->invPointList[j]];
			if (c == c2) {
				incidences[i * rij + sz[i]++] = y;
			}
		}
	} // next i

	for (i = 0; i < l1; i++) {
		if (sz[i] != rij) {
			cout << "sz[i] != rij" << endl;
			exit(1);
		}
	}

	FREE_int(sz);
	FREE_int(neighbors);
}

void decomposition::get_incidences_by_col_scheme(
	int *row_classes, int *row_class_inv, int nb_row_classes,
	int *col_classes, int *col_class_inv, int nb_col_classes,
	int row_class_idx, int col_class_idx,
	int kij, int *&incidences, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int c1, c2, /*f1,*/ f2, /*l1,*/ l2, x, nb, u, y, c, i, j;
	int *sz;
	int *neighbors;

	if (f_v) {
		cout << "decomposition::get_incidences_by_col_scheme" << endl;
	}
	c1 = row_classes[row_class_idx];
	//f1 = PStack.startCell[c1];
	//l1 = PStack.cellSize[c1];
	c2 = col_classes[col_class_idx];
	f2 = Stack->startCell[c2];
	l2 = Stack->cellSize[c2];
	incidences = NEW_int(l2 * kij);
	neighbors = NEW_int(Inc->max_k);
	sz = NEW_int(l2);
	for (j = 0; j < l2; j++) {
		sz[j] = 0;
	}
	for (j = 0; j < l2; j++) {
		y = Stack->pointList[f2 + j] - Inc->nb_points();
		nb = Inc->get_points_on_line(
				neighbors, y, verbose_level - 2);
		for (u = 0; u < nb; u++) {
			x = neighbors[u];
			i = x;
			c = Stack->cellNumber[Stack->invPointList[i]];
			if (c == c1) {
				incidences[j * kij + sz[j]++] = x;
			}
		}
	} // next j

	for (j = 0; j < l2; j++) {
		if (sz[j] != kij) {
			cout << "sz[j] != kij" << endl;
			exit(1);
		}
	}

	FREE_int(sz);
	FREE_int(neighbors);
}

void decomposition::get_and_print_decomposition_schemes()
{
	int *row_classes, *row_class_inv, nb_row_classes;
	int *col_classes, *col_class_inv, nb_col_classes;
	int *row_scheme, *col_scheme;
	int verbose_level = 0;
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "decomposition::get_and_print_decomposition_schemes "
				"computing both schemes" << endl;
	}
	Stack->allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
		verbose_level);

	row_scheme = NEW_int(nb_row_classes * nb_col_classes);
	col_scheme = NEW_int(nb_row_classes * nb_col_classes);

	get_row_decomposition_scheme(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
		row_scheme, verbose_level);

	row_scheme_to_col_scheme(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
		row_scheme, col_scheme, verbose_level);

	//cout << *this << endl;

	cout << "row_scheme:" << endl;
	Stack->print_decomposition_scheme(cout,
		row_classes, nb_row_classes,
		col_classes, nb_col_classes,
		row_scheme);

	cout << "col_scheme:" << endl;
	Stack->print_decomposition_scheme(cout,
		row_classes, nb_row_classes,
		col_classes, nb_col_classes,
		col_scheme);

	FREE_int(row_classes);
	FREE_int(row_class_inv);
	FREE_int(col_classes);
	FREE_int(col_class_inv);
	FREE_int(row_scheme);
	FREE_int(col_scheme);
}

void decomposition::get_and_print_row_tactical_decomposition_scheme_tex(
	std::ostream &ost,
	int f_enter_math, int f_print_subscripts)
{
	int *row_classes, *row_class_inv, nb_row_classes;
	int *col_classes, *col_class_inv, nb_col_classes;
	int *row_scheme; //, *col_scheme;
	int verbose_level = 0;
	int f_v = false;

	if (f_v) {
		cout << "decomposition::get_and_print_row_tactical_decomposition_scheme_tex" << endl;
	}

	if (f_v) {
		cout << "decomposition::get_and_print_row_tactical_decomposition_scheme_tex "
				"computing row scheme" << endl;
	}
	Stack->allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
		verbose_level);

	row_scheme = NEW_int(nb_row_classes * nb_col_classes);
	//col_scheme = NEW_int(nb_row_classes * nb_col_classes);

	if (f_v) {
		cout << "decomposition::get_and_print_row_tactical_decomposition_scheme_tex "
				"before get_row_decomposition_scheme" << endl;
	}
	get_row_decomposition_scheme(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
		row_scheme, verbose_level);


	if (f_v) {
		cout << "decomposition::get_and_print_row_tactical_decomposition_scheme_tex "
				"before print_row_tactical_decomposition_scheme_tex" << endl;
	}
	print_row_tactical_decomposition_scheme_tex(ost, f_enter_math,
		row_classes, nb_row_classes,
		col_classes, nb_col_classes,
		row_scheme, f_print_subscripts);

	FREE_int(row_classes);
	FREE_int(row_class_inv);
	FREE_int(col_classes);
	FREE_int(col_class_inv);
	FREE_int(row_scheme);
	//FREE_int(col_scheme);
}

void decomposition::get_and_print_column_tactical_decomposition_scheme_tex(
	std::ostream &ost,
	int f_enter_math, int f_print_subscripts)
{
	int *row_classes, *row_class_inv, nb_row_classes;
	int *col_classes, *col_class_inv, nb_col_classes;
	int *col_scheme;
	int verbose_level = 0;
	int f_v = false;

	if (f_v) {
		cout << "decomposition::get_and_print_column_tactical_decomposition_scheme_tex" << endl;
	}

	if (f_v) {
		cout << "decomposition::get_and_print_column_tactical_decomposition_scheme_tex "
				"computing column scheme" << endl;
	}
	Stack->allocate_and_get_decomposition(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
		verbose_level);

	col_scheme = NEW_int(nb_row_classes * nb_col_classes);

	get_col_decomposition_scheme(
		row_classes, row_class_inv, nb_row_classes,
		col_classes, col_class_inv, nb_col_classes,
		col_scheme, verbose_level);


	if (f_v) {
		cout << "decomposition::get_and_print_column_tactical_decomposition_scheme_tex "
				"before print_column_tactical_decomposition_scheme_tex" << endl;
	}
	print_column_tactical_decomposition_scheme_tex(
		ost, f_enter_math,
		row_classes, nb_row_classes,
		col_classes, nb_col_classes,
		col_scheme, f_print_subscripts);

	FREE_int(row_classes);
	FREE_int(row_class_inv);
	FREE_int(col_classes);
	FREE_int(col_class_inv);
	FREE_int(col_scheme);
}


void decomposition::print_partitioned(
	std::ostream &ost,
	int f_labeled)
{
	//int *A;
	int *row_classes;
	int nb_row_classes;
	int *col_classes;
	int nb_col_classes;
	int I, i, cell, l;
	int width;
	int mn;
	number_theory::number_theory_domain NT;

	mn = Inc->nb_points() * Inc->nb_lines();

	width = NT.int_log10(mn) + 1;

	//A = get_incidence_matrix();

	row_classes = NEW_int(Inc->nb_points() + Inc->nb_lines());
	col_classes = NEW_int(Inc->nb_points() + Inc->nb_lines());
	Stack->get_row_and_col_classes(row_classes, nb_row_classes,
		col_classes, nb_col_classes, 0 /* verbose_level */);
	//ost << "nb_row_classes = " << nb_row_classes << endl;
	//ost << "nb_col_classes = " << nb_col_classes << endl;

	if (f_labeled) {
		print_column_labels(ost,
			col_classes, nb_col_classes, width);
	}
	for (I = 0; I <= nb_row_classes; I++) {
		print_hline(ost, col_classes,
				nb_col_classes, width, f_labeled);
		cell = row_classes[I];
		if (I < nb_row_classes) {
			l = Stack->cellSize[cell];
			for (i = 0; i < l; i++) {
				print_line(ost, cell, i,
						col_classes, nb_col_classes, width, f_labeled);
				ost << endl;
			}
		}
	}
	FREE_int(row_classes);
	FREE_int(col_classes);
	//FREE_int(A);
}

void decomposition::print_column_labels(
	std::ostream &ost,
	int *col_classes, int nb_col_classes, int width)
{
	int f2, e2;
	int J, j, h, l, cell;
	int first_column_element = Stack->startCell[1];

	for (h = 0; h < width; h++) {
		ost << " ";
	}
	for (J = 0; J <= nb_col_classes; J++) {
		ost << "|";
		if (J == nb_col_classes) {
			break;
		}
		cell = col_classes[J];
		f2 = Stack->startCell[cell];
		l = Stack->cellSize[cell];
		for (j = 0; j < l; j++) {
			e2 = Stack->pointList[f2 + j] - first_column_element;
			ost << setw((int) width) << e2 + first_column_element;
		}
	}
	ost << endl;
}

void decomposition::print_hline(
	std::ostream &ost,
	int *col_classes, int nb_col_classes, int width, int f_labeled)
{
	int J, j, h, l, cell;

	if (f_labeled) {
		for (h = 0; h < width; h++) {
			ost << "-";
		}
	}
	else {
		//ost << "-";
	}
	for (J = 0; J <= nb_col_classes; J++) {
		ost << "+";
		if (J == nb_col_classes) {
			break;
		}
		cell = col_classes[J];
		l = Stack->cellSize[cell];
		for (j = 0; j < l; j++) {
			if (f_labeled) {
				for (h = 0; h < width; h++) {
					ost << "-";
				}
			}
			else {
				ost << "-";
			}
		}
	}
	ost << endl;
}


void decomposition::print_line(
	std::ostream &ost,
	int row_cell, int i, int *col_classes, int nb_col_classes,
	int width, int f_labeled)
{
	int f1, f2, e1, e2;
	int J, j, h, l, cell;
	int first_column_element = Stack->startCell[1];

	f1 = Stack->startCell[row_cell];
	e1 = Stack->pointList[f1 + i];
	if (f_labeled) {
		ost << setw((int) width) << e1;
	}
	for (J = 0; J <= nb_col_classes; J++) {
		ost << "|";
		if (J == nb_col_classes)
			break;
		cell = col_classes[J];
		f2 = Stack->startCell[cell];
		l = Stack->cellSize[cell];
		for (j = 0; j < l; j++) {
			e2 = Stack->pointList[f2 + j] - first_column_element;
			if (Inc->get_ij(e1, e2)) {
				if (f_labeled) {
					ost << setw((int) width) << e1 * Inc->nb_lines() + e2;
				}
				else {
					ost << "X";
					//ost << "1";
				}
			}
			else {
				if (f_labeled) {
					for (h = 0; h < width - 1; h++) {
						ost << " ";
					}
					ost << ".";
				}
				else {
					ost << ".";
					//ost << "0";
				}
			}
		}
	}
	//ost << endl;
}

void decomposition::print_row_tactical_decomposition_scheme_tex(
	std::ostream &ost, int f_enter_math_mode,
	int *row_classes, int nb_row_classes,
	int *col_classes, int nb_col_classes,
	int *row_scheme, int f_print_subscripts)
{
	int c, i, j;

	ost << "%{\\renewcommand{\\arraycolsep}{1pt}" << endl;
	if (f_enter_math_mode) {
		ost << "\\begin{align*}" << endl;
	}
	ost << "\\begin{array}{r|*{" << nb_col_classes << "}{r}}" << endl;
	ost << "\\rightarrow ";
	for (j = 0; j < nb_col_classes; j++) {
		ost << " & ";
		c = col_classes[j];
		ost << setw(6) << Stack->cellSize[c];
		if (f_print_subscripts) {
			ost << "_{" << setw(3) << c << "}";
		}
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		ost << setw(6) << Stack->cellSize[c];
			if (f_print_subscripts) {
				ost << "_{" << setw(3) << c << "}";
			}
		//f = P.startCell[c];
		for (j = 0; j < nb_col_classes; j++) {
			ost << " & " << setw(12) << row_scheme[i * nb_col_classes + j];
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	if (f_enter_math_mode) {
		ost << "\\end{align*}" << endl;
	}
	ost << "%}" << endl;
}

void decomposition::print_column_tactical_decomposition_scheme_tex(
	std::ostream &ost, int f_enter_math_mode,
	int *row_classes, int nb_row_classes,
	int *col_classes, int nb_col_classes,
	int *col_scheme, int f_print_subscripts)
{
	int c, i, j;

	ost << "%{\\renewcommand{\\arraycolsep}{1pt}" << endl;
	if (f_enter_math_mode) {
		ost << "\\begin{align*}" << endl;
	}
	ost << "\\begin{array}{r|*{" << nb_col_classes << "}{r}}" << endl;
	ost << "\\downarrow ";
	for (j = 0; j < nb_col_classes; j++) {
		ost << " & ";
		c = col_classes[j];
		ost << setw(6) << Stack->cellSize[c];
		if (f_print_subscripts) {
			ost << "_{" << setw(3) << c << "}";
		}
	}
	ost << "\\\\" << endl;
	ost << "\\hline" << endl;
	for (i = 0; i < nb_row_classes; i++) {
		c = row_classes[i];
		ost << setw(6) << Stack->cellSize[c];
		if (f_print_subscripts) {
			ost << "_{" << setw(3) << c << "}";
		}
		//f = P.startCell[c];
		for (j = 0; j < nb_col_classes; j++) {
			ost << " & " << setw(12) << col_scheme[i * nb_col_classes + j];
		}
		ost << "\\\\" << endl;
	}
	ost << "\\end{array}" << endl;
	if (f_enter_math_mode) {
		ost << "\\end{align*}" << endl;
	}
	ost << "%}" << endl;
}




}}}


