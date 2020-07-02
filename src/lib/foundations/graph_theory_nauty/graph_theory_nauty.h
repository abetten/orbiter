// graph_theory_nauty.h
//
// Anton Betten
//
// moved here from galois.h: July 27, 2018
// started as orbiter:  October 23, 2002
// 2nd version started:  December 7, 2003
// galois started:  August 12, 2005




#ifndef ORBITER_SRC_LIB_FOUNDATIONS_GRAPH_THEORY_NAUTY_GRAPH_THEORY_NAUTY_H_
#define ORBITER_SRC_LIB_FOUNDATIONS_GRAPH_THEORY_NAUTY_GRAPH_THEORY_NAUTY_H_



namespace orbiter {
namespace foundations {



// #############################################################################
// nauty_interface.cpp
// #############################################################################

//! low-level interface to the graph canonization software nauty

class nauty_interface {

public:

	void nauty_interface_graph_bitvec(int v, unsigned char *bitvector_adjacency,
		int *labeling, int *partition,
		int *Aut, int &Aut_counter,
		int *Base, int &Base_length,
		int *Transversal_length, int &Ago, int verbose_level);
	void nauty_interface_graph_int(int v, int *Adj,
		int *labeling, int *partition,
		int *Aut, int &Aut_counter,
		int *Base, int &Base_length,
		int *Transversal_length, int &Ago, int verbose_level);
	void nauty_interface_int(int v, int b, int *X, int nb_inc,
		int *labeling, int *partition,
		int *Aut, int &Aut_counter,
		int *Base, int &Base_length,
		int *Transversal_length, int &Ago);
	void nauty_interface_low_level(int v, int b, int *X, int nb_inc,
		int *labeling, int *partition,
		int *Aut, int &Aut_counter,
		int *Base, int &Base_length,
		int *Transversal_length, int &Ago);
	void nauty_interface_matrix(int *M, int v, int b,
		int *labeling, int *partition,
		int *Aut, int &Aut_counter,
		int *Base, int &Base_length,
		int *Transversal_length, int &Ago);
	void nauty_interface_matrix_int(int *M, int v, int b,
		int *labeling, int *partition,
		int *Aut, int &Aut_counter,
		int *Base, int &Base_length,
		int *Transversal_length, int &Ago, int verbose_level);


};

}}


#endif /* ORBITER_SRC_LIB_FOUNDATIONS_GRAPH_THEORY_NAUTY_GRAPH_THEORY_NAUTY_H_ */



