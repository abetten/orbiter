/*
 * shallow_schreier_ai.h
 *
 *  Created on: Jun 2, 2019
 *      Author: sajeeb
 */


#include "foundations/foundations.h"
#include "group_actions.h"


#ifndef SHALLOW_SCHREIER_AI_H_
#define SHALLOW_SCHREIER_AI_H_

using std::cout;
using std::endl;

using namespace orbiter::foundations;
using namespace orbiter::group_actions;



class shallow_schreier_ai {


public:

	shallow_schreier_ai () {
		s = NULL;
		nb_revert_backs = 0;
		deg_seq = NULL;
		nb_nodes = 0;
	};

	void generate_shallow_tree ( groups::schreier& sch, int vl );
	void get_degree_sequence (groups::schreier& sch, int vl);
	void print_degree_sequence();

	~shallow_schreier_ai();

	groups::schreier* s;
	size_t nb_revert_backs;
	int* deg_seq;
	int nb_nodes;


};

#endif /* SHALLOW_SCHREIER_AI_H_ */
