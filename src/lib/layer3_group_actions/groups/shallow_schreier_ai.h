/*
 * shallow_schreier_ai.h
 *
 *  Created on: Jun 2, 2019
 *      Author: sajeeb
 */


#include "layer1_foundations/foundations.h"
#include "layer2_discreta/discreta.h"
#include "group_actions.h"


using namespace std;


#ifndef SHALLOW_SCHREIER_AI_H_
#define SHALLOW_SCHREIER_AI_H_

using std::cout;
using std::endl;

using namespace orbiter::layer1_foundations;
using namespace orbiter::layer3_group_actions;



#if 0
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
#endif

#endif /* SHALLOW_SCHREIER_AI_H_ */
