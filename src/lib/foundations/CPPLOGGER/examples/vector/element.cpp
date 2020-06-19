/*
 * element.cpp
 *
 *  Created on: May 31, 2020
 *      Author: sajeeb
 */

#include "element.h"

element::element() {
	logger_info("in empty constructor.");
}

element::element(int data) {
	logger_info("created element with data: %d", data);
	this->data = data;
}

element::~element() {}
