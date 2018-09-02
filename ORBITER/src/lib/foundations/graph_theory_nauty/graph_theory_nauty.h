// graph_theory_nauty.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005

// #############################################################################
// nauty_interface.C:
// #############################################################################

void nauty_interface_graph_bitvec(INT v, uchar *bitvector_adjacency, 
	INT *labeling, INT *partition, 
	INT *Aut, INT &Aut_counter, 
	INT *Base, INT &Base_length, 
	INT *Transversal_length, INT &Ago, INT verbose_level);
void nauty_interface_graph_INT(INT v, INT *Adj, 
	INT *labeling, INT *partition, 
	INT *Aut, INT &Aut_counter, 
	INT *Base, INT &Base_length, 
	INT *Transversal_length, INT &Ago, INT verbose_level);
void nauty_interface_INT(INT v, INT b, INT *X, INT nb_inc, 
	INT *labeling, INT *partition, 
	INT *Aut, INT &Aut_counter, 
	INT *Base, INT &Base_length, 
	INT *Transversal_length, INT &Ago);
void nauty_interface(int v, int b, INT *X, INT nb_inc, 
	int *labeling, int *partition, 
	INT *Aut, int &Aut_counter, 
	int *Base, int &Base_length, 
	int *Transversal_length, int &Ago);
void nauty_interface_matrix(int *M, int v, int b, 
	int *labeling, int *partition, 
	INT *Aut, int &Aut_counter, 
	int *Base, int &Base_length, 
	int *Transversal_length, int &Ago);
void nauty_interface_matrix_INT(INT *M, INT v, INT b, 
	INT *labeling, INT *partition, 
	INT *Aut, INT &Aut_counter, 
	INT *Base, INT &Base_length, 
	INT *Transversal_length, INT &Ago, INT verbose_level);




