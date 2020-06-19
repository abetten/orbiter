/*
 * element.h
 *
 *  Created on: May 31, 2020
 *      Author: sajeeb
 */

#include "../../CPPLOGGER_SYNC.h"

#ifndef EXAMPLES_VECTOR_ELEMENT_H_
#define EXAMPLES_VECTOR_ELEMENT_H_

class element {
public:
	element();
	element(int data);
	~element();

private:
	int data = -1;
};

#endif /* EXAMPLES_VECTOR_ELEMENT_H_ */
