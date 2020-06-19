/*
 * vector.h
 *
 *  Created on: May 31, 2020
 *      Author: sajeeb
 */

#include <iostream>
#include <element.h>
#include "../../CPPLOGGER_SYNC.h"

#ifndef EXAMPLES_VECTOR_VECTOR_H_
#define EXAMPLES_VECTOR_VECTOR_H_

class vector {
public:
	vector();
	~vector();

	void add(element* __e);

private:
	size_t capacity = 10;
	size_t size = 0;
	element** e = new element* [capacity];
};

#endif /* EXAMPLES_VECTOR_VECTOR_H_ */
