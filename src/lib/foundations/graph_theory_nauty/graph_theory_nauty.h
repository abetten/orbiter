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

	void nauty_interface_graph_bitvec(int v,
			data_structures::bitvector *Bitvec,
		int *partition,
		data_structures::nauty_output *NO,
		int verbose_level);
	void nauty_interface_graph_int(int v, int *Adj,
		int *partition,
		data_structures::nauty_output *NO,
		int verbose_level);
	void nauty_interface_matrix_int(
		combinatorics::encoded_combinatorial_object *Enc,
		data_structures::nauty_output *NO,
		int verbose_level);


};

}}


#endif /* ORBITER_SRC_LIB_FOUNDATIONS_GRAPH_THEORY_NAUTY_GRAPH_THEORY_NAUTY_H_ */



