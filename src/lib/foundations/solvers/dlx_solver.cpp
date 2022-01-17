/*
 * dlx_solver.cpp
 *
 *  Created on: Jan 12, 2022
 *      Author: betten
 */

// based on earlier work from before 2013 by:
// Xi Chen
// Student, Computer Science and Engineering
// University of New South Wales
// Kensington
// hypernewbie@gmail.com
// http://cgi.cse.unsw.edu.au/~xche635/dlx_sodoku/



#include "foundations.h"


using namespace std;

namespace orbiter {
namespace foundations {




#define DLX_FANCY_LEVEL 30




dlx_solver::dlx_solver()
{
	Descr = NULL;

	Input_data = NULL;
	nb_rows = 0;
	nb_cols = 0;


	//std::string solutions_fname;
	fp_sol = NULL;

	//std::string tree_fname;
	write_tree_cnt = 0;

	fp_tree = NULL;


	nRow = 0;
	nCol = 0;

	f_has_RHS = FALSE; // [nCol]
	target_RHS = NULL; // [nCol]
	current_RHS = NULL; // [nCol]
	current_row = NULL; // [nCol]
	current_row_save = NULL; // [sum_rhs]


	// we allow three types of conditions:
	// equations t_EQ
	// inequalities t_LE
	// zero or a fixed value t_ZOR

	f_type = FALSE;
	type = NULL; // [nCol]
	changed_type_columns = NULL; // [nCol]
	nb_changed_type_columns = NULL; // [sum_rhs]
	nb_changed_type_columns_total= 0;

	Result = NULL; // [nRow]
	Nb_choices = NULL; // [nRow]
	Cur_choice = NULL; // [nRow]
	Cur_col = NULL; // [nRow]
	Nb_col_nodes = NULL; // [nCol]
	nb_sol = 0;
	nb_backtrack_nodes = 0;
	Matrix = NULL; // [nRow * nCol]
	Root = NULL;
	f_has_callback_solution_found = FALSE;
	callback_solution_found = NULL;
	callback_solution_found_data = NULL;

}


dlx_solver::~dlx_solver()
{
	if (Input_data) {
		FREE_int(Input_data);
	}
}


void dlx_solver::init(
		dlx_problem_description *Descr,
		int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "dlx_solver::init" << endl;
	}

	dlx_solver::Descr = Descr;


	if (Descr->f_data_label) {
		Orbiter->get_matrix_from_label(Descr->data_label,
				Input_data, nb_rows, nb_cols);
	}
	else if (Descr->f_data_matrix) {
		nb_rows = Descr->data_matrix_m;
		nb_cols = Descr->data_matrix_n;
		Input_data = NEW_int(nb_rows * nb_cols);
		Orbiter->Int_vec->copy(Descr->data_matrix, Input_data, nb_rows * nb_cols);
	}
	else {
		cout << "dlx_solver::init please use option -data_label or use data_matrix" << endl;
		exit(1);
	}

	if (f_v) {
		cout << "dlx_solver::init done" << endl;
	}
}



int dlx_solver::dataLeft(int i)
{
	return i - 1 < 0 ? nCol - 1 : i - 1;
}

int dlx_solver::dataRight(int i)
{
	return (i + 1) % nCol;
}

int dlx_solver::dataUp(int i)
{
	return i - 1 < 0 ? nRow - 1 : i - 1;
}

int dlx_solver::dataDown(int i)
{
	return (i + 1) % nRow;
}


void dlx_solver::install_callback_solution_found(
	void (*callback_solution_found)(
			int *solution, int len, int nb_sol, void *data),
	void *callback_solution_found_data)
{
	f_has_callback_solution_found = TRUE;
	dlx_solver::callback_solution_found =
			callback_solution_found;
	dlx_solver::callback_solution_found_data =
			callback_solution_found_data;
}

void dlx_solver::de_install_callback_solution_found()
{
	f_has_callback_solution_found = FALSE;
}

#if 0
void dlx_solver::Test()
{
	int Data[] = {
		0, 0, 1, 0, 1, 1, 0,
		1, 0, 0, 1, 0, 0, 1,
		0, 1, 1, 0, 0, 1, 0,
		1, 0, 0, 1, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 1,
		0, 0, 0, 1, 1, 0, 1,
		};
	// solutions: rows 0, 3, 4 make 1,1,1,1,1,1,1

	int nb_rows = 6;
	int nb_cols = 7;
	int nb_sol, nb_backtrack;

	AppendRowAndSolve(Data, nb_rows, nb_cols, 0);
}

void dlx_solver::TransposeAppendAndSolve(int *Data, int nb_rows, int nb_cols,
	int verbose_level)
{
	int *Data2;
	int i, j;

	Data2 = NEW_int(nb_cols * nb_rows);
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Data2[j * nb_rows + i] = Data[i * nb_cols + j];
		}
	}
	AppendRowAndSolve(Data2, nb_cols, nb_rows,
		verbose_level);

	FREE_int(Data2);
}

void dlx_solver::TransposeAndSolveRHS(int *Data, int nb_rows, int nb_cols,
	int *RHS, int f_has_type, diophant_equation_type *type,
	int verbose_level)
{
	int *Data2;
	int i, j;

	Data2 = NEW_int(nb_cols * nb_rows);
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Data2[j * nb_rows + i] = Data[i * nb_cols + j];
		}
	}
	AppendRowAndSolveRHS(Data2, nb_cols, nb_rows,
		RHS, f_has_type, type,
		verbose_level);

	FREE_int(Data2);
}

void dlx_solver::AppendRowAndSolve(int *Data, int nb_rows, int nb_cols,
	int verbose_level)
{
	int *Data2;
	int i, j;

	Data2 = NEW_int((nb_rows + 1) * nb_cols);
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Data2[i * nb_cols + j] = Data[i * nb_cols + j];
		}
	}
	i = nb_rows;
	for (j = 0; j < nb_cols; j++) {
		Data2[i * nb_cols + j] = 1;
	}
	Solve(Data2, nb_rows + 1, nb_cols,
		verbose_level);

	FREE_int(Data2);
}

void dlx_solver::AppendRowAndSolveRHS(int *Data, int nb_rows, int nb_cols,
	int *RHS, int f_has_type, diophant_equation_type *type,
	int verbose_level)
{
	int *Data2;
	int i, j;

	Data2 = NEW_int((nb_rows + 1) * nb_cols);
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Data2[i * nb_cols + j] = Data[i * nb_cols + j];
		}
	}

	// fill in the RHS in the header:
	i = nb_rows;
	for (j = 0; j < nb_cols; j++) {
		Data2[i * nb_cols + j] = RHS[j];
	}


	Solve_with_RHS(Data2, nb_rows + 1, nb_cols,
		RHS, f_has_type, type,
		verbose_level);

	FREE_int(Data2);
}
#endif

void dlx_solver::Solve(int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "dlx_solver::Solve nb_rows = " << nb_rows << " nb_cols = "
				<< nb_cols << endl;
	}

	int *Input_data_transposed;

	int nr, nc;
	int i, j;

	nr = nb_cols;
	nr++; // add a row of ones
	nc = nb_rows;

	Input_data_transposed = NEW_int(nr * nc);
	for (i = 0; i < nb_cols; i++) {
		for (j = 0; j < nc; j++) {
			Input_data_transposed[i * nc + j] = Input_data[j * nb_cols + i];
		}
	}
	i = nb_cols;
	for (j = 0; j < nc; j++) {
		Input_data_transposed[i * nc + j] = 1;
	}


	if (f_v) {
		cout << "dlx_solver::Solve before CreateMatrix" << endl;
	}
	CreateMatrix(Input_data_transposed, nr, nc, verbose_level - 1);
	if (f_v) {
		cout << "dlx_solver::Solve after CreateMatrix" << endl;
	}

	open_solution_file(verbose_level);

	open_tree_file(verbose_level);

	nb_backtrack_nodes = 0;


	Search(0);


	if (f_v) {
		cout << "dlx_solver::Solve finds " << nb_sol << " solutions "
				"with nb_backtrack_nodes=" << nb_backtrack_nodes << endl;
	}

	close_solution_file();
	close_tree_file();




	DeleteMatrix();

	FREE_int(Input_data_transposed);

	if (f_v) {
		cout << "dlx_solver::Solve done" << endl;
	}
}

void dlx_solver::Solve_with_RHS(int *RHS, int f_has_type,
		diophant_equation_type *type,
	int verbose_level)
{
	int f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "dlx_solver::Solve_with_RHS nb_rows = " << nb_rows
				<< " nb_cols = " << nb_cols << endl;
	}


	CreateMatrix(Input_data, nb_rows, nb_cols, verbose_level - 1);
	Create_RHS(nb_cols, RHS, f_has_type, type, verbose_level);

	open_solution_file(verbose_level);

	if (Descr->f_write_tree) {
		open_tree_file(verbose_level);
	}

	nb_backtrack_nodes = 0;



	SearchRHS(0, verbose_level);




	if (f_v) {
		cout << "dlx_solver::Solve_with_RHS finds "
				<< nb_sol << " solutions "
				"with nb_backtrack_nodes="
				<< nb_backtrack_nodes << endl;
	}

	close_solution_file();
	close_tree_file();




	DeleteMatrix();
	Delete_RHS();


	if (f_v) {
		cout << "dlx_solver::Solve_with_RHS done" << endl;
	}
}

void dlx_solver::open_solution_file(int verbose_level)
{
	if (Descr->f_write_solutions) {

		solutions_fname.assign(Descr->label_txt);
		solutions_fname.append("_solutions.txt");

		fp_sol = new ofstream;
		fp_sol->open(solutions_fname);
	}
}

void dlx_solver::close_solution_file()
{
	if (Descr->f_write_solutions) {




		*fp_sol << -1 << " " << nb_sol << " "
				<< nb_backtrack_nodes << endl;
		fp_sol->close();
		delete fp_sol;
	}
}

void dlx_solver::open_tree_file(int verbose_level)
{
	if (Descr->f_write_tree) {

		tree_fname.assign(Descr->label_txt);
		tree_fname.append("_tree.txt");


		write_tree_cnt = 0;
		fp_tree = new ofstream;
		fp_tree->open(tree_fname);
		*fp_tree << "# " << nCol << endl;
	}
}

void dlx_solver::close_tree_file()
{
	if (Descr->f_write_tree) {
		*fp_tree << -1 << " " << write_tree_cnt << endl;
		delete fp_tree;
	}
}


void dlx_solver::Create_RHS(int nb_cols, int *RHS, int f_has_type,
		diophant_equation_type *type, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int i, sum_rhs;

	if (f_v) {
		cout << "dlx_solver::Create_RHS" << endl;
	}

	f_has_RHS = TRUE;
	if (nb_cols != nCol) {
		cout << "dlx_solver::Create_RHS nb_cols != nCol" << endl;
		exit(1);
	}


	target_RHS = NEW_int(nCol);
	current_RHS = NEW_int(nCol);
	current_row = NEW_int(nCol);

	for (i = 0; i < nCol; i++) {
		target_RHS[i] = RHS[i];
		current_RHS[i] = 0;
		current_row[i] = -1;
	}

	sum_rhs = 0;
	for (i = 0; i < nCol; i++) {
		sum_rhs += target_RHS[i];
	}

	if (f_v) {
		cout << "sum_rhs=" << sum_rhs << endl;
	}

	current_row_save = NEW_int(sum_rhs);

	dlx_solver::type = NEW_OBJECTS(diophant_equation_type, nCol);
	if (f_has_type) {
		for (i = 0; i < nCol; i++) {
			dlx_solver::type[i] = type[i];
		}
	}
	else {
		for (i = 0; i < nCol; i++) {
			dlx_solver::type[i] = t_EQ;
		}
	}
	changed_type_columns = NEW_int(nCol);
	nb_changed_type_columns = NEW_int(sum_rhs);
	nb_changed_type_columns_total = 0;

	if (f_v) {
		cout << "dlx_solver::Create_RHS done" << endl;
	}
}

void dlx_solver::Delete_RHS()
{
	if (target_RHS) {
		FREE_int(target_RHS);
		target_RHS = NULL;
	}
	if (current_RHS) {
		FREE_int(current_RHS);
		current_RHS = NULL;
	}
	if (current_row) {
		FREE_int(current_row);
		current_row = NULL;
	}
	if (current_row_save) {
		FREE_int(current_row_save);
		current_row_save = NULL;
	}
	if (f_type) {
		FREE_OBJECTS(type);
		type = NULL;
	}
}

void dlx_solver::CreateMatrix(int *Data,
		int nb_rows, int nb_cols, int verbose_level)
{
	int f_v = (verbose_level >= 1);
	int a, b, i, j;

	if (f_v) {
		cout << "dlx_solver::CreateMatrix" << endl;
	}
	nRow = nb_rows;
	nCol = nb_cols;
	nb_sol = 0;

	if (f_v) {
		cout << "dlx_solver::CreateMatrix" << endl;
		cout << "The " << nb_rows << " x " << nb_cols
				<< " matrix is:" << endl;
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				cout << Data[i * nb_cols + j];
			}
			cout << endl;
		}
		//int_matrix_print(Data, nb_rows, nb_cols);
		cout << endl;
	}

	Matrix = new dlx_node[nRow * nCol];
	Root = new dlx_node;
	//RowHeader = new pdlx_node[nRow];


	Result = NEW_int(nRow);
	Nb_choices = NEW_int(nRow);
	Cur_choice = NEW_int(nRow);
	Cur_col = NEW_int(nRow);
	Nb_col_nodes = NEW_int(nCol);

	for (j = 0; j < nCol; j++) {
		Nb_col_nodes[j] = 0;
	}


	// Build toroidal linked list according to the data matrix:
	for (a = 0; a < nRow; a++) {
		for (b = 0; b < nCol; b++) {
			Matrix[a * nCol + b].row = a;
			Matrix[a * nCol + b].col = b;
		}
	}

	// Connect the coefficients which are nonzero to
	// their neighbors up and down and left and right:

	if (f_v) {
		cout << "dlx_solver::CreateMatrix building the toroidal matrix" << endl;
	}
	for (a = 0; a < nRow; a++) {
		for (b = 0; b < nCol; b++) {
			if (Data[a * nCol + b] != 0) {

				// Left pointer
				i = a;
				j = b;
				do {
					j = dataLeft(j);
				} while (Data[i * nCol + j] == 0);

				Matrix[a * nCol + b].Left = &Matrix[i * nCol + j];

				// Right pointer
				i = a;
				j = b;
				do {
					j = dataRight(j);
				} while (Data[i * nCol + j] == 0);

				Matrix[a * nCol + b].Right = &Matrix[i * nCol + j];

				// Up pointer
				i = a;
				j = b;
				do {
					i = dataUp(i);
				} while (Data[i * nCol + j] == 0);

				Matrix[a * nCol + b].Up = &Matrix[i * nCol + j];

				// Down pointer
				i = a;
				j = b;
				do {
					i = dataDown(i);
				} while (Data[i * nCol + j] == 0);

				Matrix[a * nCol + b].Down = &Matrix[i * nCol + j];

#if 0
				cout << "at " << a << "/" << b << ":";
				cout << " Left="; print_position(Matrix[a * nCol + b].Left);
				cout << " Right="; print_position(Matrix[a * nCol + b].Right);
				cout << " Up="; print_position(Matrix[a * nCol + b].Up);
				cout << " Down="; print_position(Matrix[a * nCol + b].Down);
				cout << endl;
#endif
				// Header pointer at the very bottom:
				Matrix[a * nCol + b].Header =
						&Matrix[(nRow - 1) * nCol + b];
				//Row Header
				//RowHeader[a] = &Matrix[a * nCol + b];
			}
		}
	}
	if (f_v) {
		cout << "dlx_solver::CreateMatrix building the toroidal matrix finished" << endl;
	}

	// Count the number of nodes in each column
	// (i.e. the number of ones in the column of the matrix)
	for (j = 0; j < nCol; j++) {
		Nb_col_nodes[j] = 0;

		if (f_v) {
			cout << "dlx_solver::CreateMatrix counting nodes in column " << j << endl;
		}
		dlx_node *ColNode, *RowNode;

		// this is the RHS
		ColNode = &Matrix[(nRow - 1) * nCol + j];

		for (RowNode = ColNode->Down; RowNode != ColNode; RowNode = RowNode->Down) {
			Nb_col_nodes[j]++;
		}
	}

#if 0
	for (a = 0; a < nCol; a++) {
		Matrix[(nRow - 1) * nCol + a].row = nRow - 1;
		Matrix[(nRow - 1) * nCol + a].col = a;
	}
#endif
	//Insert root at the end of all RHS nodes:
	Root->Left = &Matrix[(nRow - 1) * nCol + (nCol - 1)];
	Root->Right = &Matrix[(nRow - 1) * nCol + 0];
	Matrix[(nRow - 1) * nCol + nCol - 1].Right = Root;
	Matrix[(nRow - 1) * nCol + 0].Left = Root;
	Root->row = -1;
	Root->col = -1;
	if (f_v) {
		cout << "dlx_solver::CreateMatrix done" << endl;
	}
}

void dlx_solver::DeleteMatrix()
{
	delete [] Matrix;
	delete Root;
	FREE_int(Result);
	FREE_int(Nb_choices);
	FREE_int(Cur_choice);
	FREE_int(Nb_col_nodes);
}

dlx_node *dlx_solver::get_column_header(int c)
{
	dlx_node *Node;

	for (Node = Root->Right; Node != Root; Node = Node->Right) {
		if (Node->col == c) {
			return Node;
		}
	}
	cout << "dlx_solver::get_column_header cannot find column " << c << endl;
	exit(1);
}

dlx_node *dlx_solver::ChooseColumnFancy()
{
	int j, nb_node, nb_node_min = 0;
	dlx_node *Node, *Node_min = NULL;

	for (Node = Root->Right; Node != Root; Node = Node->Right) {
		j = Node->col;
		nb_node = Nb_col_nodes[j];
		if (Node_min == NULL) {
			Node_min = Node;
			nb_node_min = nb_node;
		}
		else {
			if (nb_node < nb_node_min) {
				Node_min = Node;
				nb_node_min = nb_node;
			}
		}
	}
	return Node_min;
}

dlx_node *dlx_solver::ChooseColumn()
{
	if (Root->Right == Root) {
		cout << "dlx_solver::ChooseColumn Root->Right == Root" << endl;
		exit(1);
	}
	return Root->Right;
}

dlx_node *dlx_solver::ChooseColumnFancyRHS()
{
	int j, nb_node, nb_node_min = 0;
	dlx_node *Node, *Node_min = NULL;

	for (Node = Root->Right; Node != Root; Node = Node->Right) {
		j = Node->col;
		if (type[j] == t_LE || type[j] == t_ZOR) {
			continue;
		}
		nb_node = Nb_col_nodes[j];
		if (Node_min == NULL) {
			Node_min = Node;
			nb_node_min = nb_node;
		}
		else {
			if (nb_node < nb_node_min) {
				Node_min = Node;
				nb_node_min = nb_node;
			}
		}
	}
	return Node_min;
}

dlx_node *dlx_solver::ChooseColumnRHS()
{
	dlx_node *Node;
	int j;

	if (Root->Right == Root) {
		cout << "dlx_solver::ChooseColumn Root->Right == Root" << endl;
		exit(1);
	}

	for (Node = Root->Right; Node != Root; Node = Node->Right) {
		j = Node->col;
		if (type[j] == t_LE || type[j] == t_ZOR) {
			continue;
		}
		return Node;
	}
	cout << "dlx_solver::ChooseColumnRHS could not find a node" << endl;
	exit(1);
}

void dlx_solver::write_tree(int k)
{
	if (Descr->f_write_tree) {
		int i;
		*fp_tree << k;
		for (i = 0; i < k; i++) {
			*fp_tree << " " << Result[i];
		}
		*fp_tree << endl;
		write_tree_cnt++;
	}
}

void dlx_solver::print_if_necessary(int k)
{
	if ((nb_backtrack_nodes & ((1 << 20) - 1)) == 0) {
		int a, d;

		a = nb_backtrack_nodes >> 20;
		cout << "Search: " << a << " * 2^20 nodes, "
				<< " nb_sol=" << nb_sol << " position ";

		d = MINIMUM(Descr->tracking_depth, k);

		for (int i = 0; i < d; i++) {
			cout << Cur_choice[i] << " / " << Nb_choices[i];
			if (i < d - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}
}

void dlx_solver::process_solution(int k)
{
#if 0
	if (f_v) {
        	cout << "solution " << nb_sol << " : ";
		//PrintSolution();
		int_vec_print(cout, Result, k);
		cout << endl;
	}
#endif
	if (Descr->f_write_solutions) {
		int i;
		*fp_sol << k;
		for (i = 0; i < k; i++) {
			*fp_sol << " " << Result[i];
		}
		*fp_sol << endl;
	}
	if (f_has_callback_solution_found) {
		(*callback_solution_found)(Result,
				k, nb_sol, callback_solution_found_data);
	}
	nb_sol++;
}

void dlx_solver::count_nb_choices(int k, dlx_node *Column)
{
	dlx_node *RowNode;
	int r, d;

	Nb_choices[k] = 0;

	d = MINIMUM(Descr->tracking_depth, k);

	if (k < d) {
		for (RowNode = Column->Down;
				RowNode != Column;
				RowNode = RowNode->Down) {
			Nb_choices[k]++;
		}

		if (FALSE) {
			cout << "Choice set: ";
			for (RowNode = Column->Down;
					RowNode != Column;
					RowNode = RowNode->Down) {
				r = RowNode->row;
				cout << " " << r;
			}
			cout << endl;
		}
	}
}

int dlx_solver::IsDone()
{
	dlx_node *N;
	int c;

#if 0
	cout << "c : current_RHS[c] : target_RHS[c]" << endl;
	for (c = 0; c < nCol; c++) {
		cout << c << " : " << current_RHS[c]
		<< " : " << target_RHS[c] << endl;
		}
#endif
	N = Root->Left;
	while (TRUE) {
		if (N == Root) {
			//cout << "is done" << endl;
			return TRUE;
		}
		c = N->col;
		if (IsColumnNotDone(c)) {
			//cout << "is not done because of column " << c << endl;
			return FALSE;
		}
		N = N->Left;
	}
}

int dlx_solver::IsColumnDone(int c)
{
	if (current_RHS[c] == target_RHS[c]) {
		return TRUE;
	}
	return FALSE;
}

int dlx_solver::IsColumnNotDone(int c)
{
	if (type[c] == t_EQ && current_RHS[c] < target_RHS[c]) {
		return TRUE;
	}
	return FALSE;
}

void dlx_solver::Search(int k)
{
	//cout << "Search k=" << k << endl;
	//print_root();

	write_tree(k);

	nb_backtrack_nodes++;
	print_if_necessary(k);

	if (Root->Left == Root && Root->Right == Root) {
		// All header columns gone means we have a valid solution!

		process_solution(k);
		return;
	}


	dlx_node *Column;


	if (k < DLX_FANCY_LEVEL) {
		Column = ChooseColumnFancy();
	}
	else {
		Column = ChooseColumn();
	}

	Cover(Column);

	dlx_node *RowNode;
	dlx_node *RightNode;
	dlx_node *LeftNode;

	count_nb_choices(k, Column);

	Cur_choice[k] = 0;

	// we loop over all nodes in that column:

	for (RowNode = Column->Down;
			RowNode != Column;
			RowNode = RowNode->Down, Cur_choice[k]++) {

		// Try this row node on!
		Result[k] = RowNode->row;

		// Since we have made our choice of row, we can now remove the
		// columns where the chosen row has a one.
		// These equations are also satisfied now.

		for (RightNode = RowNode->Right;
				RightNode != RowNode;
				RightNode = RightNode->Right) {
			Cover(RightNode->Header);
		}

		// And we recurse:
		Search(k + 1);


		// corrected an error in the original version:
		for (LeftNode = RowNode->Left;
				LeftNode != RowNode;
				LeftNode = LeftNode->Left) {
			UnCover(LeftNode->Header);
		}
	}

	UnCover(Column);
}

void dlx_solver::SearchRHS(int k, int verbose_level)
{
	int f_v = (verbose_level >= 1);

#if 0
	if ((nb_backtrack_nodes % 100000) == 0) {
		verbose_level += 1;
	}
#endif


	if (f_v) {
		int i;

		cout << "dlx_solver::SearchRHS k=" << k
				<< " nb_backtrack_nodes="
				<< nb_backtrack_nodes << " : ";
		for (i = 0; i < k; i++) {
			cout << Cur_choice[i] << "/" << Nb_choices[i];
			if (i < k - 1) {
				cout << ", ";
			}
		}
		cout << endl;
	}
	//print_root();

	write_tree(k);

	nb_backtrack_nodes++;
	print_if_necessary(k);

	if (IsDone()) {
		// All header columns gone means we have a valid solution!
		if (f_v) {
			cout << "dlx_solver::SearchRHS k=" << k << " solution ";
			Orbiter->Int_vec->print(cout, Result, k);
			cout << " found" << endl;
		}

		process_solution(k);
		return;
	}


	dlx_node *Column;
	int r, c, f_done, c0, c2;


	Column = NULL;

	// First we check if we the column from
	// the previous level is satisfied or not.
	// If it is not, we need to work on this column again.
	if (k) {
		c0 = Cur_col[k - 1];
		if (IsColumnNotDone(c0)) {
			Column = get_column_header(c0);
		}
	}

	if (Column == NULL) {
		if (k < DLX_FANCY_LEVEL) {
			Column = ChooseColumnFancyRHS();
		}
		else {
			Column = ChooseColumnRHS();
		}
	}

	c = Column->col;
	Cur_col[k] = c;
	if (f_v) {
		cout << "dlx_solver::SearchRHS k=" << k
				<< " choosing column " << c << endl;
	}

	current_row_save[k] = current_row[c];
	if (current_RHS[c] == 0) {
		current_row[c] = -1;
	}
	current_RHS[c]++;

	if (f_v) {
		cout << "dlx_solver::SearchRHS k=" << k << " choosing column " << c
				<< " RHS = " << current_RHS[c] << " / "
				<< target_RHS[c] << endl;
	}

	if (current_RHS[c] > target_RHS[c]) {
		cout << "dlx_solver::SearchRHS current_RHS[c] > "
				"target_RHS[c] error" << endl;
		exit(1);
	}


	f_done = IsColumnDone(c);
		// have we reached the RHS in this column?

	if (f_done) {
		if (f_v) {
			cout << "dlx_solver::SearchRHS k=" << k << " column " << c
					<< " is done, so we cover it" << endl;
		}
		// we have reached the RHS in this column,
		// so we cannot place any more in this column.
		// Hence, rows which have a
		// nonzero entry in this column can be removed.
		Cover(Column);
  	}

	dlx_node *RowNode;
	dlx_node *RightNode;
	dlx_node *LeftNode;

	count_nb_choices(k, Column);

	if (f_v) {
		cout << "dlx_solver::SearchRHS k=" << k << " column " << c
				<< " number of choices is " << Nb_choices[k] << endl;
	}


	Cur_choice[k] = 0;

	// we loop over all nodes in that column:


	for (RowNode = Column->Down; RowNode != Column;
			RowNode = RowNode->Down, Cur_choice[k]++) {

		// Try this row node on!
		r = RowNode->row;

		nb_changed_type_columns[k] = 0;


		Result[k] = r;

		if (f_v) {
			cout << "dlx_solver::SearchRHS k=" << k << " column " << c
					<< " choice " << Cur_choice[k] << " / "
					<< Nb_choices[k] << " which is ";
			Orbiter->Int_vec->print(cout, Result, k + 1);
			cout << endl;
		}

#if 1

		// The next test is needed to prevent
		// solutions from being found repeatedly,
		// namely once for each rearrangement
		// of the rows associated to a fixed column
		// This can only happen if the RHS
		// in that column is greater than one.


		if (r <= current_row[c]) {
			// we should not choose this row,
			// because we have dealt with this row before.
			if (f_v) {
				cout << "dlx_solver::SearchRHS skip" << endl;
			}
			continue;
		}
#endif


		current_row[c] = r;
			// store the current row so that
			// if we get to choose another node in this column,
			// we require that that node is
			// in a row higher than the current one.
			// In particular, we cannot choose the same node twice.

		// Since we have made our choice of row,
		// we can now remove the
		// columns where the RHS is now satisfied
		// because the chosen row has a one in it
		// (other than the column c that we are working on).
		// For each of these columns we need to call Cover
		// to remove further rows which have a one in that column

		for (RightNode = RowNode->Right;
				RightNode != RowNode;
				RightNode = RightNode->Right) {
			c2 = RightNode->col;
			if (c2 != c) {
				current_RHS[c2]++;
				if (current_RHS[c2] > target_RHS[c2]) {
					cout << "dlx_solver::SearchRHS current_RHS[c2] > "
							"target_RHS[c2] error" << endl;
					exit(1);
				}
				if (current_RHS[c2] == target_RHS[c2]) {
					Cover(RightNode->Header);
				}

#if 1
				// Here we change the type of a condition:
				// ZOR's are changed into EQ's.
				// We record which ones we changed
				// so we can later change back.

				if (current_RHS[c2] == 1 && type[c2] == t_ZOR) {
					type[c2] = t_EQ;
					changed_type_columns[
							nb_changed_type_columns_total++] = c2;
					if (nb_changed_type_columns_total >= nCol) {
						cout << "dlx_solver::SearchRHS nb_changed_type_columns_total >= nCol" << endl;
						exit(1);
					}
					nb_changed_type_columns[k]++;
				}
#endif
			}
		}



		if (f_v) {
			cout << "dlx_solver::SearchRHS k=" << k << " column " << c
					<< " choice " << Cur_choice[k] << " / "
					<< Nb_choices[k] << " which is ";
			Orbiter->Int_vec->print(cout, Result, k + 1);
			cout << " recursing" << endl;
		}


		// And we recurse:
		SearchRHS(k + 1, verbose_level);

		if (f_v) {
			cout << "dlx_solver::SearchRHS k=" << k << " column " << c
					<< " choice " << Cur_choice[k] << " / "
					<< Nb_choices[k] << " which is ";
			Orbiter->Int_vec->print(cout, Result, k + 1);
			cout << " after recursion" << endl;
		}


		// corrected an error in the original version:
		for (LeftNode = RowNode->Left;
				LeftNode != RowNode;
				LeftNode = LeftNode->Left) {
			c2 = LeftNode->col;
			if (c2 != c) {
				if (current_RHS[c2] == target_RHS[c2]) {
					UnCover(LeftNode->Header);
				}
				current_RHS[c2]--;
			}
		}

#if 1
		// Here we undo the change of type from above
		// EQ's are changed back into ZOR's:

		int i;

		for (i = 0; i < nb_changed_type_columns[k]; i++) {
			c2 = changed_type_columns[--nb_changed_type_columns_total];
			if (current_RHS[c2] != 0) {
				cout << "dlx_solver::SearchRHS current_RHS[c2] != 0 error, "
						"current_RHS[c2]=" << current_RHS[c2]
						<< " c2=" << c2
						<< " c=" << c << " k=" << k << endl;
				exit(1);
			}
			type[c2] = t_ZOR;
		}
#endif
	}



	if (f_done) {
		// undo the Cover operation
		UnCover(Column);
	}

	current_row[c] = current_row_save[k];
	current_RHS[c]--;

}

void dlx_solver::Cover(dlx_node *ColNode)
{

	//cout << "Cover" << endl;
	dlx_node *RowNode, *RightNode;
	int j;

	// remove the column by crosslisting
	// the left and right neighbors.
	// recall that this is in the header.

	ColNode->Right->Left = ColNode->Left;
	ColNode->Left->Right = ColNode->Right;

	// remove rows which have a 1 in column ColNode
	// updates column counts

	// Go down the column,
	// visit each row with a 1 in that particular column
	// and remove that row by directly connecting
	// the up-neighbor to the down-neighbor.
	// This is done for each entry 1 in that row
	// which is not in the particular column that we started with.

	for (RowNode = ColNode->Down;
			RowNode != ColNode;
			RowNode = RowNode->Down) {
		//cout << "RowNode " << RowNode->row
		//<< "/" << RowNode->col << endl;

		for (RightNode = RowNode->Right;
				RightNode != RowNode;
				RightNode = RightNode->Right) {
			j = RightNode->col;
			Nb_col_nodes[j]--;
			RightNode->Up->Down = RightNode->Down;
			RightNode->Down->Up = RightNode->Up;
		}
	}


	//cout << "Cover done" << endl;
}

void dlx_solver::UnCover(dlx_node *ColNode)
{

	//cout << "dlx_solver::UnCover" << endl;
	dlx_node *RowNode, *LeftNode;
	int j;

	// puts rows back in which have previously been removed in Cover:

	for (RowNode = ColNode->Up;
			RowNode!= ColNode;
			RowNode = RowNode->Up) {
		for (LeftNode = RowNode->Left;
				LeftNode != RowNode;
				LeftNode = LeftNode->Left) {
			j = LeftNode->col;
			Nb_col_nodes[j]++;
			LeftNode->Up->Down = LeftNode;
			LeftNode->Down->Up = LeftNode;
		}
	}

	// put the column back in:

	ColNode->Right->Left = ColNode;
	ColNode->Left->Right = ColNode;


	//cout << "dlx_solver::UnCover done" << endl;
}



}}

