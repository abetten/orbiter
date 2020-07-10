/*
 * main.cpp
 *
 *  Created on: May 31, 2020
 *      Author: sajeeb
 */

#include <iostream>
#include <vector.h>
#include "../../CPPLOGGER_SYNC.h"

/**
 * The following should always be called only in the main translation unit.
 * This is to default initialize the logger flags.
 */
logger_init();

int main() {

	vector v;

	for (uint i=0; i<100; ++i) v.add(new element(i));

}
