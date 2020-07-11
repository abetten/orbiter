/*
 * scoped_logging.cpp
 *
 *  Created on: Jul 3, 2020
 *      Author: sajeeb
 */


#include <CPPLOGGER_SYNC.h>

logger_init();

int main() {
	logger_enable(false);
	logger_enable_global(false);
	logger_info("This is before entering a new scope. This message should not be printed");
	{
	    logger_enable_scope(true);
	    logger_info("This is logging in the new scope.");
	}
	logger_info("This is after leaving the scope. This message should not be printed");
	logger_enable(true);
	logger_enable_global(true);
	logger_info("This message is after enabling global logging.");
}

