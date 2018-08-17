// dlx.C
//
// Xi Chen
// Student, Computer Science and Engineering
// University of New South Wales
// Kensington
// hypernewbie@gmail.com
// http://cgi.cse.unsw.edu.au/~xche635/dlx_sodoku/
//
//
// modified by Anton Betten
//
//
//
// started:  April 7, 2013




#include "foundations.h"


#define DLX_FANCY_LEVEL 30
#define DLX_TRACKING_DEPTH 10

typedef struct dlx_node dlx_node;
typedef struct dlx_node *pdlx_node;

struct dlx_node {
    
    dlx_node * Header;
    
    dlx_node * Left;
    dlx_node * Right;
    dlx_node * Up;
    dlx_node * Down;
    
    INT  row; // row index
    INT  col; // col index
};

INT DLX_f_write_tree = FALSE;
INT DLX_write_tree_cnt = 0;
ofstream *DLX_fp_tree = NULL;


INT nRow;
INT nCol;

INT f_has_RHS = FALSE; // [nCol]
INT *target_RHS = NULL; // [nCol]
INT *current_RHS = NULL; // [nCol]
INT *current_row = NULL; // [nCol]
INT *current_row_save = NULL; // [sum_rhs]


// we allow three types of conditions:
// equations t_EQ
// inequalities t_LE
// zero or a fixed value t_ZOR

INT f_type = FALSE;
diophant_equation_type *type = NULL; // [nCol]
INT *changed_type_columns = NULL; // [nCol]
INT *nb_changed_type_columns = NULL; // [sum_rhs]
INT nb_changed_type_columns_total;

INT *Result; // [nRow]
INT *Nb_choices; // [nRow]
INT *Cur_choice; // [nRow]
INT *DLX_Cur_col; // [nRow]
INT *Nb_col_nodes; // [nCol]
INT dlx_nb_sol = 0;
INT dlx_nb_backtrack_nodes;
dlx_node *Matrix = NULL; // [nRow * nCol]
dlx_node *Root = NULL; 
//dlx_node **RowHeader = NULL; // [nRow]
INT dlx_f_write_to_file = FALSE;
ofstream *dlx_fp_sol = NULL;
INT f_has_callback_solution_found = FALSE;
void (*callback_solution_found)(INT *solution, INT len, INT nb_sol, void *data);
void *callback_solution_found_data;

inline INT dataLeft(INT i) { return i-1<0?nCol-1:i-1; }
inline INT dataRight(INT i) { return (i+1)%nCol; }
inline INT dataUp(INT i) { return i-1<0?nRow-1:i-1; }
inline INT dataDown(INT i) { return (i+1)%nRow; }

void open_solution_file(INT f_write_file, const BYTE *solution_fname, INT verbose_level);
void close_solution_file(INT f_write_file);
void open_tree_file(INT f_write_tree_file, const BYTE *tree_fname, INT verbose_level);
void close_tree_file(INT f_write_tree_file);
void print_position(dlx_node *p);
void Create_RHS(INT nb_cols, INT *RHS, INT f_has_type, diophant_equation_type *type, INT verbose_level);
void Delete_RHS();
void CreateMatrix(INT *Data, INT nb_rows, INT nb_cols, INT verbose_level);
void DeleteMatrix();
dlx_node *get_column_header(INT c);
dlx_node *ChooseColumnFancy(void);
dlx_node *ChooseColumn(void);
dlx_node *ChooseColumnFancyRHS(void);
dlx_node *ChooseColumnRHS(void);
void print_root();
void write_tree(INT k);
void print_if_necessary(INT k);
void process_solution(INT k);
void count_nb_choices(INT k, dlx_node *Column);
INT IsDone();
INT IsColumnDone(INT c);
INT IsColumnNotDone(INT c);
void DlxSearch(INT k);
void DlxSearchRHS(INT k);
void Cover(dlx_node *ColNode);
void UnCover(dlx_node *ColNode);


void install_callback_solution_found(
	void (*callback_solution_found)(INT *solution, INT len, INT nb_sol, void *data),
	void *callback_solution_found_data)
{
	f_has_callback_solution_found = TRUE;
	::callback_solution_found = callback_solution_found;
	::callback_solution_found_data = callback_solution_found_data;
}

void de_install_callback_solution_found()
{
	f_has_callback_solution_found = FALSE;
}

void DlxTest()
{
	INT Data[] = {
0, 0, 1, 0, 1, 1, 0,
1, 0, 0, 1, 0, 0, 1,
0, 1, 1, 0, 0, 1, 0,
1, 0, 0, 1, 0, 0, 0,
0, 1, 0, 0, 0, 0, 1,
0, 0, 0, 1, 1, 0, 1,
};
// solutions: rows 0, 3, 4 make 1,1,1,1,1,1,1

	INT nb_rows = 6;
	INT nb_cols = 7;
	INT nb_sol, nb_backtrack;

	DlxAppendRowAndSolve(Data, nb_rows, nb_cols, 
		nb_sol, nb_backtrack, 
		FALSE, "", 
		FALSE, "", 
		0);
}

void DlxTransposeAppendAndSolve(INT *Data, INT nb_rows, INT nb_cols, 
	INT &nb_sol, INT &nb_backtrack, 
	INT f_write_file, const BYTE *solution_fname, 
	INT f_write_tree_file, const BYTE *tree_fname, 
	INT verbose_level)
{
	INT *Data2;
	INT i, j;

	Data2 = NEW_INT(nb_cols * nb_rows);
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Data2[j * nb_rows + i] = Data[i * nb_cols + j];
			}
		}
	DlxAppendRowAndSolve(Data2, nb_cols, nb_rows, 
		nb_sol, nb_backtrack, 
		f_write_file, solution_fname, 
		f_write_tree_file, tree_fname, 
		verbose_level);

	FREE_INT(Data2);
}

void DlxTransposeAndSolveRHS(INT *Data, INT nb_rows, INT nb_cols, 
	INT *RHS, INT f_has_type, diophant_equation_type *type, 
	INT &nb_sol, INT &nb_backtrack, 
	INT f_write_file, const BYTE *solution_fname, 
	INT f_write_tree_file, const BYTE *tree_fname, 
	INT verbose_level)
{
	INT *Data2;
	INT i, j;

	Data2 = NEW_INT(nb_cols * nb_rows);
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Data2[j * nb_rows + i] = Data[i * nb_cols + j];
			}
		}
	DlxAppendRowAndSolveRHS(Data2, nb_cols, nb_rows, 
		RHS, f_has_type, type, 
		nb_sol, nb_backtrack, 
		f_write_file, solution_fname, 
		f_write_tree_file, tree_fname, 
		verbose_level);

	FREE_INT(Data2);
}

void DlxAppendRowAndSolve(INT *Data, INT nb_rows, INT nb_cols, 
	INT &nb_sol, INT &nb_backtrack, 
	INT f_write_file, const BYTE *solution_fname, 
	INT f_write_tree_file, const BYTE *tree_fname, 
	INT verbose_level)
{
	INT *Data2;
	INT i, j;

	Data2 = NEW_INT((nb_rows + 1) * nb_cols);
	for (i = 0; i < nb_rows; i++) {
		for (j = 0; j < nb_cols; j++) {
			Data2[i * nb_cols + j] = Data[i * nb_cols + j];
			}
		}
	i = nb_rows;
	for (j = 0; j < nb_cols; j++) {
		Data2[i * nb_cols + j] = 1;
		}
	DlxSolve(Data2, nb_rows + 1, nb_cols, 
		nb_sol, nb_backtrack, 
		f_write_file, solution_fname, 
		f_write_tree_file, tree_fname,
		verbose_level);

	FREE_INT(Data2);
}

void DlxAppendRowAndSolveRHS(INT *Data, INT nb_rows, INT nb_cols, 
	INT *RHS, INT f_has_type, diophant_equation_type *type, 
	INT &nb_sol, INT &nb_backtrack, 
	INT f_write_file, const BYTE *solution_fname, 
	INT f_write_tree_file, const BYTE *tree_fname, 
	INT verbose_level)
{
	INT *Data2;
	INT i, j;

	Data2 = NEW_INT((nb_rows + 1) * nb_cols);
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

	
	DlxSolve_with_RHS(Data2, nb_rows + 1, nb_cols, 
		RHS, f_has_type, type, 
		nb_sol, nb_backtrack, 
		f_write_file, solution_fname, 
		f_write_tree_file, tree_fname,
		verbose_level);

	FREE_INT(Data2);
}

void DlxSolve(INT *Data, INT nb_rows, INT nb_cols, 
	INT &nb_sol, INT &nb_backtrack, 
	INT f_write_file, const BYTE *solution_fname, 
	INT f_write_tree_file, const BYTE *tree_fname, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "DlxSolve nb_rows = " << nb_rows << " nb_cols = " << nb_cols << endl;
		}


	CreateMatrix(Data, nb_rows, nb_cols, verbose_level - 1);

	open_solution_file(f_write_file, solution_fname, verbose_level);
	open_tree_file(f_write_tree_file, tree_fname, verbose_level);

	dlx_nb_backtrack_nodes = 0;



	DlxSearch(0);


	nb_sol = ::dlx_nb_sol;
	nb_backtrack = dlx_nb_backtrack_nodes;


	if (f_v) {
		cout << "DlxSolve finds " << dlx_nb_sol << " solutions with nb_backtrack_nodes=" << dlx_nb_backtrack_nodes << endl;
		}

	close_solution_file(f_write_file);
	close_tree_file(f_write_tree_file);
	



	DeleteMatrix();


	if (f_v) {
		cout << "DlxSolve done" << endl;
		}
}

void DlxSolve_with_RHS(INT *Data, INT nb_rows, INT nb_cols, 
	INT *RHS, INT f_has_type, diophant_equation_type *type, 
	INT &nb_sol, INT &nb_backtrack, 
	INT f_write_file, const BYTE *solution_fname, 
	INT f_write_tree_file, const BYTE *tree_fname, 
	INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

	if (f_v) {
		cout << "DlxSolve_with_RHS nb_rows = " << nb_rows << " nb_cols = " << nb_cols << endl;
		}


	CreateMatrix(Data, nb_rows, nb_cols, verbose_level - 1);
	Create_RHS(nb_cols, RHS, f_has_type, type, verbose_level);

	open_solution_file(f_write_file, solution_fname, verbose_level);
	open_tree_file(f_write_tree_file, tree_fname, verbose_level);

	dlx_nb_backtrack_nodes = 0;



	DlxSearchRHS(0, verbose_level);


	nb_sol = ::dlx_nb_sol;
	nb_backtrack = dlx_nb_backtrack_nodes;


	if (f_v) {
		cout << "DlxSolve_with_RHS finds " << dlx_nb_sol << " solutions with nb_backtrack_nodes=" << dlx_nb_backtrack_nodes << endl;
		}

	close_solution_file(f_write_file);
	close_tree_file(f_write_tree_file);
	



	DeleteMatrix();
	Delete_RHS();


	if (f_v) {
		cout << "DlxSolve_with_RHS done" << endl;
		}
}

void open_solution_file(INT f_write_file, const BYTE *solution_fname, INT verbose_level)
{
	if (f_write_file) {
		dlx_fp_sol = new ofstream;
		dlx_f_write_to_file = TRUE;
		dlx_fp_sol->open(solution_fname);		
		}
	else {
		dlx_f_write_to_file = FALSE;
		}
}

void close_solution_file(INT f_write_file)
{
	if (f_write_file) {
		*dlx_fp_sol << -1 << " " << dlx_nb_sol << " " << dlx_nb_backtrack_nodes << endl;
		dlx_fp_sol->close();
		delete dlx_fp_sol;
		dlx_f_write_to_file = FALSE;
		}
}

void open_tree_file(INT f_write_tree_file, const BYTE *tree_fname, INT verbose_level)
{
	if (f_write_tree_file) {
		DLX_f_write_tree = TRUE;
		DLX_write_tree_cnt = 0;
		DLX_fp_tree = new ofstream;
		DLX_fp_tree->open(tree_fname);
		*DLX_fp_tree << "# " << nCol << endl;
		}
	else {
		DLX_f_write_tree = FALSE;
		}
}

void close_tree_file(INT f_write_tree_file)
{
	if (f_write_tree_file) {
		*DLX_fp_tree << -1 << " " << DLX_write_tree_cnt << endl;
		delete DLX_fp_tree;
		}
}

void print_position(dlx_node *p)
{
	cout << p->row << "/" << p->col;
}

void Create_RHS(INT nb_cols, INT *RHS, INT f_has_type, diophant_equation_type *type, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT i, sum_rhs;

	if (f_v) {
		cout << "dlx.C: Create_RHS" << endl;
		}

	f_has_RHS = TRUE;
	if (nb_cols != nCol) {
		cout << "dlx.C: Create_RHS nb_cols != nCol" << endl;
		exit(1);
		}


	target_RHS = NEW_INT(nCol);
	current_RHS = NEW_INT(nCol);
	current_row = NEW_INT(nCol);
	
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

	current_row_save = NEW_INT(sum_rhs);

	if (f_has_type) {
		::type = new diophant_equation_type[nCol];
		for (i = 0; i < nCol; i++) {
			::type[i] = type[i];
			}
		}
	else {
		::type = new diophant_equation_type[nCol];
		for (i = 0; i < nCol; i++) {
			::type[i] = t_EQ;
			}
		}
	changed_type_columns = NEW_INT(nCol);
	nb_changed_type_columns = NEW_INT(sum_rhs);
	nb_changed_type_columns_total = 0;

	if (f_v) {
		cout << "dlx.C: Create_RHS done" << endl;
		}
}

void Delete_RHS()
{
	if (target_RHS) {
		FREE_INT(target_RHS);
		target_RHS = NULL;
		}
	if (current_RHS) {
		FREE_INT(current_RHS);
		current_RHS = NULL;
		}
	if (current_row) {
		FREE_INT(current_row);
		current_row = NULL;
		}
	if (current_row_save) {
		FREE_INT(current_row_save);
		current_row_save = NULL;
		}
	if (f_type) {
		delete [] type;
		type = NULL;
		}
}

void CreateMatrix(INT *Data, INT nb_rows, INT nb_cols, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);
	INT a, b, i, j;

	nRow = nb_rows;
	nCol = nb_cols;
	dlx_nb_sol = 0;

	if (f_v) {
		cout << "dlx.C: CreateMatrix" << endl;
		cout << "The " << nb_rows << " x " << nb_cols << " matrix is:" << endl;
		for (i = 0; i < nb_rows; i++) {
			for (j = 0; j < nb_cols; j++) {
				cout << Data[i * nb_cols + j];
				}
			cout << endl;
			}
		//INT_matrix_print(Data, nb_rows, nb_cols);
		cout << endl;
		}

	Matrix = new dlx_node[nRow * nCol];
	Root = new dlx_node;
	//RowHeader = new pdlx_node[nRow];

	
	Result = NEW_INT(nRow);
	Nb_choices = NEW_INT(nRow);
	Cur_choice = NEW_INT(nRow);
	DLX_Cur_col = NEW_INT(nRow);
	Nb_col_nodes = NEW_INT(nCol);

	for (j = 0; j < nCol; j++) {
		Nb_col_nodes[j] = 0;
		}

	
	// Build toroidal linklist matrix according to data bitmap
	for (a = 0; a < nRow; a++) {
		for (b = 0; b < nCol; b++) {
			Matrix[a * nCol + b].row = a;
			Matrix[a * nCol + b].col = b;
			}
		}
	
	// Connect the coefficients which are nonzero to their up and down and left and right neighbors:

	for (a = 0; a < nRow; a++) {
		for (b = 0; b < nCol; b++) {
			if (Data[a * nCol + b] != 0) {
				// Left pointer
				i = a; j = b; do { j = dataLeft(j); } while (Data[i * nCol + j] == 0);
				Matrix[a * nCol + b].Left = &Matrix[i * nCol + j]; 
				// Right pointer
				i = a; j = b; do { j = dataRight(j); } while (Data[i * nCol + j] == 0);
				Matrix[a * nCol + b].Right = &Matrix[i * nCol + j];
				// Up pointer
				i = a; j = b; do { i = dataUp(i); } while (Data[i * nCol + j] == 0);
				Matrix[a * nCol + b].Up = &Matrix[i * nCol + j];
				// Down pointer
				i = a; j = b; do { i = dataDown(i); } while (Data[i * nCol + j] == 0);
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
				Matrix[a * nCol + b].Header = &Matrix[(nRow - 1) * nCol + b];
				//Row Header
				//RowHeader[a] = &Matrix[a * nCol + b];
				}
			}
		}

	// Count the number of nodes in each column (i.e. the number of ones in the column of the matrix)
	for (j = 0; j < nCol; j++) {
		Nb_col_nodes[j] = 0;

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
}

void DeleteMatrix()
{
	delete Matrix;
	delete Root;
	FREE_INT(Result);
	FREE_INT(Nb_choices);
	FREE_INT(Cur_choice);
	FREE_INT(Nb_col_nodes);
}

dlx_node *get_column_header(INT c)
{
	dlx_node *Node;
	
	for (Node = Root->Right; Node != Root; Node = Node->Right) {
		if (Node->col == c) {
			return Node;
			}
		}
	cout << "get_column_header cannot find column " << c << endl;
	exit(1);
}

dlx_node *ChooseColumnFancy(void)
{
	INT j, nb_node, nb_node_min = 0;
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

dlx_node *ChooseColumn(void)
{
	if (Root->Right == Root) {
		cout << "ChooseColumn Root->Right == Root" << endl;
		exit(1);
		}
	return Root->Right;
}

dlx_node *ChooseColumnFancyRHS(void)
{
	INT j, nb_node, nb_node_min = 0;
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

dlx_node *ChooseColumnRHS(void)
{
	dlx_node *Node;
	INT j;
	
	if (Root->Right == Root) {
		cout << "ChooseColumn Root->Right == Root" << endl;
		exit(1);
		}
	
	for (Node = Root->Right; Node != Root; Node = Node->Right) {
		j = Node->col;
		if (type[j] == t_LE || type[j] == t_ZOR) {
			continue;
			}
		return Node;
		}
	cout << "ChooseColumnRHS cound not find a node" << endl;
	exit(1);
}

void print_root()
{
	dlx_node *Node, *N;

	for (Node = Root->Right; Node != Root; Node = Node->Right) {
		cout << "printing column ";
		print_position(Node);
		cout << endl;
		for (N = Node->Down; N != Node; N = N->Down) {
			cout << "Node ";
			print_position(N);
			cout << endl;
			}
		}
}

void write_tree(INT k)
{
	if (DLX_f_write_tree) {
		INT i;
		*DLX_fp_tree << k;
		for (i = 0; i < k; i++) {
			*DLX_fp_tree << " " << Result[i];
			}
		*DLX_fp_tree << endl;
		DLX_write_tree_cnt++;
		}
}

void print_if_necessary(INT k)
{
	if ((dlx_nb_backtrack_nodes & ((1 << 20) - 1)) == 0) {
		INT a;

		a = dlx_nb_backtrack_nodes >> 20;
		cout << "DlxSearch: " << a << " * 2^20 nodes, " << " nb_sol=" << dlx_nb_sol << " position ";
		for (int i = 0; i < MINIMUM(DLX_TRACKING_DEPTH, k); i++) {
			cout << Cur_choice[i] << " / " << Nb_choices[i];
			if (i < DLX_TRACKING_DEPTH - 1) {
				cout << ", ";
				}
			}
		cout << endl;
		}
}

void process_solution(INT k)
{
#if 0
	if (f_v) {
        	cout << "DlxSearch solution " << dlx_nb_sol << " : ";
		//PrintSolution();
		INT_vec_print(cout, Result, k);
		cout << endl;
		}
#endif
	if (dlx_f_write_to_file) {
		INT i;
		*dlx_fp_sol << k;
		for (i = 0; i < k; i++) {
			*dlx_fp_sol << " " << Result[i];
			}
		*dlx_fp_sol << endl;
		}
	if (f_has_callback_solution_found) {
		(*callback_solution_found)(Result, k, dlx_nb_sol, callback_solution_found_data);
		}
	dlx_nb_sol++;
}

void count_nb_choices(INT k, dlx_node *Column)
{
	dlx_node *RowNode;
	INT r;

	Nb_choices[k] = 0;
	
	if (k < DLX_TRACKING_DEPTH) {
		for (RowNode = Column->Down; RowNode != Column; RowNode = RowNode->Down) {
			Nb_choices[k]++;
			}

		if (FALSE) {
			cout << "Choice set: ";
			for (RowNode = Column->Down; RowNode != Column; RowNode = RowNode->Down) {
				r = RowNode->row;
				cout << " " << r;
				}
			cout << endl;
			}
		}
}

INT IsDone()
{
	dlx_node *N;
	INT c;
	
#if 0
	cout << "c : current_RHS[c] : target_RHS[c]" << endl;
	for (c = 0; c < nCol; c++) {
		cout << c << " : " << current_RHS[c] << " : " << target_RHS[c] << endl;
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

INT IsColumnDone(INT c)
{
	if (current_RHS[c] == target_RHS[c]) {
		return TRUE;
		}
	return FALSE;
}

INT IsColumnNotDone(INT c)
{
	if (type[c] == t_EQ && current_RHS[c] < target_RHS[c]) {
		return TRUE;
		}
	return FALSE;
}

void DlxSearch(INT k)
{
	//cout << "DlxSearch k=" << k << endl;
	//print_root();
	
	write_tree(k);

	dlx_nb_backtrack_nodes++;
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

	for (RowNode = Column->Down; RowNode != Column; RowNode = RowNode->Down, Cur_choice[k]++) {
		
		// Try this row node on!
		Result[k] = RowNode->row;

		// Since we have made our choice of row, we can now remove the 
		// columns where the chosen row has a one. 
		// These equations are also satisfied now.
 
		for (RightNode = RowNode->Right; RightNode != RowNode; RightNode = RightNode->Right) {
			Cover(RightNode->Header);
			}

		// And we recurse:
		DlxSearch(k + 1);


		// corrected an error in the original version:
		for (LeftNode = RowNode->Left; LeftNode != RowNode; LeftNode = LeftNode->Left) {
			UnCover(LeftNode->Header);
			}
		}
    
	UnCover(Column);
}

void DlxSearchRHS(INT k, INT verbose_level)
{
	INT f_v = (verbose_level >= 1);

#if 0
	if ((dlx_nb_backtrack_nodes % 100000) == 0) {
		verbose_level += 1;
		}
#endif


	if (f_v) {
		INT i;
		
		cout << "DlxSearchRHS k=" << k << " dlx_nb_backtrack_nodes=" << dlx_nb_backtrack_nodes << " : ";
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

	dlx_nb_backtrack_nodes++;
	print_if_necessary(k);
	
	if (IsDone()) {
		// All header columns gone means we have a valid solution!
		if (f_v) {
			cout << "DlxSearchRHS k=" << k << " solution ";
			INT_vec_print(cout, Result, k);
			cout << " found" << endl;
			}

		process_solution(k);
		return;
		}


	dlx_node *Column;
	INT r, c, f_done, c0, c2;


	Column = NULL;

	// First we check if we the column from the previous level is satisfied or not.
	// If it is not, we need to work on this column again.
	if (k) {
		c0 = DLX_Cur_col[k - 1];
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
	DLX_Cur_col[k] = c;
	if (f_v) {
		cout << "DlxSearchRHS k=" << k << " choosing column " << c << endl;
		}

	current_row_save[k] = current_row[c];
	if (current_RHS[c] == 0) {
		current_row[c] = -1;
		}
	current_RHS[c]++;

	if (f_v) {
		cout << "DlxSearchRHS k=" << k << " choosing column " << c << " RHS = " << current_RHS[c] << " / " << target_RHS[c] << endl;
		}

	if (current_RHS[c] > target_RHS[c]) {
		cout << "DlxSearchRHS current_RHS[c] > target_RHS[c] error" << endl;
		exit(1);
		}


	f_done = IsColumnDone(c);
		// have we reached the RHS in this column?
	if (f_done) {
		if (f_v) {
			cout << "DlxSearchRHS k=" << k << " column " << c << " is done, so we cover it" << endl;
			}
			// we have reached the RHS in this column, 
			// so we cannot place any more in this column.
			// Hence, rows which have a nonero entry in this column can be removed.
		Cover(Column);
  		}

	dlx_node *RowNode;
	dlx_node *RightNode;
	dlx_node *LeftNode;
	
	count_nb_choices(k, Column);

	if (f_v) {
		cout << "DlxSearchRHS k=" << k << " column " << c << " number of choices is " << Nb_choices[k] << endl;
		}


	Cur_choice[k] = 0;

	// we loop over all nodes in that column:


	for (RowNode = Column->Down; RowNode != Column; RowNode = RowNode->Down, Cur_choice[k]++) {
		
		// Try this row node on!
		r = RowNode->row;

		nb_changed_type_columns[k] = 0;

		
		Result[k] = r;

		if (f_v) {
			cout << "DlxSearchRHS k=" << k << " column " << c << " choice " << Cur_choice[k] << " / " << Nb_choices[k] << " which is ";
			INT_vec_print(cout, Result, k + 1);
			cout << endl;
			}

#if 1

		// The next test is needed to prevent solutions from being found repeatedly,
		// namely once for each rearrangement of the rows associated to a fixed column
		// This can only happen if the RHS in that column is greater than one.


		if (r <= current_row[c]) {
			// we should not choose this row, because we have dealt with this row before.
			if (f_v) {
				cout << "DlxSearchRHS skip" << endl;
				}
			continue;
			}
#endif


		current_row[c] = r;
			// store the current row so that
			// if we get to choose another node in this column, 
			// we require that that node is in a row higher than the current one.
			// In particular, we cannot choose the same node twice.

		// Since we have made our choice of row, we can now remove the 
		// columns where the RHS is now satisfied because the chosen row has a one in it 
		// (other than the column c that we are working on). 
		// For each of these columns we need to call Cover 
		// to remove further rows which have a one in that column
 
		for (RightNode = RowNode->Right; RightNode != RowNode; RightNode = RightNode->Right) {
			c2 = RightNode->col;
			if (c2 != c) {
				current_RHS[c2]++;
				if (current_RHS[c2] > target_RHS[c2]) {
					cout << "DlxSearchRHS current_RHS[c2] > target_RHS[c2] error" << endl;
					exit(1);
					}
				if (current_RHS[c2] == target_RHS[c2]) {
					Cover(RightNode->Header);
					}

#if 1
				// Here we change the type of a condition:
				// ZOR's are changed into EQ's.
				// We record which ones we changed so we can later change back.

				if (current_RHS[c2] == 1 && type[c2] == t_ZOR) {
					type[c2] = t_EQ;
					changed_type_columns[nb_changed_type_columns_total++] = c2;
					if (nb_changed_type_columns_total >= nCol) {
						cout << "DlxSearchRHS nb_changed_type_columns_total >= nCol" << endl;
						exit(1);
						}
					nb_changed_type_columns[k]++;
					}
#endif
				}
			}



		if (f_v) {
			cout << "DlxSearchRHS k=" << k << " column " << c << " choice " << Cur_choice[k] << " / " << Nb_choices[k] << " which is ";
			INT_vec_print(cout, Result, k + 1);
			cout << " recursing" << endl;
			}


		// And we recurse:
		DlxSearchRHS(k + 1, verbose_level);

		if (f_v) {
			cout << "DlxSearchRHS k=" << k << " column " << c << " choice " << Cur_choice[k] << " / " << Nb_choices[k] << " which is ";
			INT_vec_print(cout, Result, k + 1);
			cout << " after recursion" << endl;
			}


		// corrected an error in the original version:
		for (LeftNode = RowNode->Left; LeftNode != RowNode; LeftNode = LeftNode->Left) {
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

		INT i;
		
		for (i = 0; i < nb_changed_type_columns[k]; i++) {
			c2 = changed_type_columns[--nb_changed_type_columns_total];
			if (current_RHS[c2] != 0) {
				cout << "DlxSearchRHS current_RHS[c2] != 0 error, current_RHS[c2]=" << current_RHS[c2] << " c2=" << c2 << " c=" << c << " k=" << k << endl;
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

void Cover(dlx_node *ColNode)
{

	//cout << "Cover" << endl;
	dlx_node *RowNode, *RightNode;
	INT j;

	// remove the column by crosslisting the left and right neighbors.
	// recall that this is in the header.

	ColNode->Right->Left = ColNode->Left;
	ColNode->Left->Right = ColNode->Right;

	// remove rows which have a 1 in column ColNode
	// updates column counts 

	// Go down the column,
	// visit each row with a 1 in that particular column 
	// and remove that row by directly connecting the up-neighbor to the down-neighbor.
	// This is done for each entry 1 in that row which is not in the particular column that we started with.

	for (RowNode = ColNode->Down; RowNode != ColNode; RowNode = RowNode->Down) {
		//cout << "RowNode " << RowNode->row << "/" << RowNode->col << endl;

		for (RightNode = RowNode->Right; RightNode != RowNode; RightNode = RightNode->Right) {
			j = RightNode->col;
			Nb_col_nodes[j]--;
			RightNode->Up->Down = RightNode->Down;
			RightNode->Down->Up = RightNode->Up;
			}
		}

	
	//cout << "Cover done" << endl;
}

void UnCover(dlx_node *ColNode) 
{

	//cout << "UnCover" << endl;
	dlx_node *RowNode, *LeftNode;
	INT j;

	// puts rows back in which have previously been removed in Cover:

	for (RowNode = ColNode->Up; RowNode!= ColNode; RowNode = RowNode->Up) {
		for (LeftNode = RowNode->Left; LeftNode != RowNode; LeftNode = LeftNode->Left) {
			j = LeftNode->col;
			Nb_col_nodes[j]++;
			LeftNode->Up->Down = LeftNode;
			LeftNode->Down->Up = LeftNode;
			}
		}

	// put the column back in:

	ColNode->Right->Left = ColNode;
	ColNode->Left->Right = ColNode;


	//cout << "UnCover done" << endl;
}


