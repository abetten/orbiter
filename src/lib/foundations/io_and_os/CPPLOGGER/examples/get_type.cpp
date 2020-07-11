/*
 * get_type.cpp
 *
 *  Created on: May 30, 2020
 *      Author: sajeeb
 */

#include "../CPPLOGGER_SYNC.h"

/**
 * The following should always be called only in the main translation unit.
 * This is to default initialize the logger flags.
 */
logger_init();

class sample_class {
public:
	sample_class() {}
	void print() {
		logger_info("print method in class: %s", logger_red(logger_get_type(this)));
	}
};

int main(int argc, char** argv) {

	sample_class c;
	c.print();
	logger_info("c: %s", logger_get_type(c));


	int a = 10;
	logger_info("type of a: %s", logger_get_type(a));
}
