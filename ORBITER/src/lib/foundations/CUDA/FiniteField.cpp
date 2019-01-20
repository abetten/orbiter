/*
 * FiniteField.h
 *
 *  Created on: Oct 25, 2018
 *      Author: sajeeb
 */

#include "FiniteField.h"

namespace FiniteField {

	__device__ void sleep(clock_value_t sleep_cycles)
	{
		clock_value_t start = clock64();
		clock_value_t cycles_elapsed;
		do { cycles_elapsed = clock64() - start; }
		while (cycles_elapsed < sleep_cycles);
	}

}
