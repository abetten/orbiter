/*
 * print_array.cpp
 *
 *  Created on: Jun 18, 2020
 *      Author: sajeeb
 */

#include <CPPLOGGER_SYNC.h>

logger_init();

int main() {
	int* array = new int [8];
	for (size_t i=0; i<8; ++i) array[i] = i;

	logger_info("Array Contents: %s", logger_array(array, 8));

	delete [] array;
}
