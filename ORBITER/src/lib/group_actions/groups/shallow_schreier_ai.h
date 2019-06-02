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

	shallow_schreier_ai(schreier& sch, int vl);

	~shallow_schreier_ai();

private:

//	int verbose_level = 0;
//	schreier* sch = NULL;


};

#endif /* SHALLOW_SCHREIER_AI_H_ */
