/*
 * CPPLOGGER.h
 *
 *  Created on: May 27, 2020
 *      Author: sajeeb
 */

#ifndef CPPLOGGER_SYNC_H_
#define CPPLOGGER_SYNC_H_

#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <stdio.h>
#include <stdarg.h>
#include <thread>
#include <mutex>
#include <cxxabi.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>
#include "lock_free_linked_list.h"

using std::string;
using std::mutex;

#ifndef uint
typedef unsigned int uint;
#endif

#ifndef __force_inline__
#define __force_inline__ __attribute__((always_inline)) inline
#endif

/**
 * Add support for colors for versions older than c++11. This is for backwards compatibility
 */
#if __cplusplus >= 201103L
#define COLOR_UNICODE "\u001b"
#else
#define COLOR_UNICODE "\x1b"
#endif

/**
 * Define the ANSI color codes
 */
#ifndef __ANSI_COLORS__
#define __ANSI_COLORS__
#define ANSI_RESET		COLOR_UNICODE	"[00m"
#define ANSI_BLACK		COLOR_UNICODE	"[30m"
#define ANSI_RED		COLOR_UNICODE	"[31m"
#define ANSI_GREEN		COLOR_UNICODE	"[32m"
#define ANSI_YELLOW		COLOR_UNICODE	"[33m"
#define ANSI_BLUE		COLOR_UNICODE	"[34m"
#define ANSI_PURPLE		COLOR_UNICODE	"[35m"
#define ANSI_CYAN		COLOR_UNICODE	"[36m"
#define ANSI_WHITE		COLOR_UNICODE	"[37m"
#define ANSI_BOLD		COLOR_UNICODE	"[01m"
#define ANSI_UNDERLINE	COLOR_UNICODE	"[04m"
#define ANSI_ITALIC		COLOR_UNICODE	"[03m"
#endif

/**
 * Colors supported by the logger
 */
#ifndef LOGGER_COLORS
#define LOGGER_COLORS
#define logger_black(msg) 		logger::_BLACK_(msg).c_str()
#define logger_red(msg)			logger::_RED_(msg).c_str()
#define logger_green(msg) 		logger::_GREEN_(msg).c_str()
#define logger_yellow(msg) 		logger::_YELLOW_(msg).c_str()
#define logger_blue(msg) 		logger::_BLUE_(msg).c_str()
#define logger_purple(msg) 		logger::_PURPLE_(msg).c_str()
#define logger_cyan(msg) 		logger::_CYAN_(msg).c_str()
#define logger_white(msg) 		logger::_WHITE_(msg).c_str()
#define logger_bold(msg) 		logger::_BOLD_(msg).c_str()
#define logger_underline(msg) 	logger::_UNDERLINE_(msg).c_str()
#define logger_italic(msg) 		logger::_ITALIC_(msg).c_str()
#endif

/**
 * macro to initialize the extern variables in logger
 */
#ifndef logger_init
#define logger_init() \
namespace logger { \
	bool _print_timestamps_ = false; \
	bool _print_thread_id_ = false; \
	bool _print_log_type_ = true; \
	bool _print_file_ = true; \
	bool _print_line_ = true; \
	bool _enable_global_ = true; \
	bool _flush_immediately_ = false; \
	mutex critical_section; \
	va_list __args__; \
	FILE* _output_stream_ = stdout; \
	timestamp_resolution _resolution = timestamp_resolution::millisecond;\
	log_level _loglevel_global_ = log_level::all;\
}
#endif

#ifndef logger_output_stream
#define logger_output_stream(v) logger::_output_stream_ = v
#endif

namespace logger {

	/**
	 *  Mutex lock for critical section. here critical section
	 *  refers to the block of code responsible for printing
	 *  logging information to screen.
	 */
	extern mutex critical_section;

	/**
	 * Setting the following to true will print the timestamps
	 * with the logging information
	 */
	extern bool _print_timestamps_;
	#ifndef logger_print_timestamps
	#define logger_print_timestamps(v) logger::_print_timestamps_ = v
	#endif

	/**
	 * Setting the following to true will print the thread id
	 * with the logging info, i.e. the thread responsible for
	 * printing the current log statement
	 */
	extern bool _print_thread_id_;
	#ifndef logger_print_thread_id
	#define logger_print_thread_id(v) logger::_print_thread_id_ = v
	#endif

	/**
	 * Setting the following to false will not print "[INFO]: ", "[ERROR]: ", "[WARN]: "
	 * before log messages.
	 */
	extern bool _print_log_type_;
	#ifndef logger_print_log_type
	#define logger_print_log_type(v) logger::_print_log_type_ = v
	#endif

	/**
	 * Setting the following to true will the file name from
	 * where the current log statement was invoked
	 */
	extern bool _print_file_;
	#ifndef logger_print_file
	#define logger_print_file(v) logger::_print_file_ = v
	#endif

	/**
	 * Setting the following to true will the line number from
	 * where the current log statement was invoked
	 */
	extern bool _print_line_;
	#ifndef logger_print_line
	#define logger_print_line(v) logger::_print_line_ = v
	#endif

	/**
	 * setting the following to false will disable global logging.
	 */
	extern bool _enable_global_;
	#ifndef logger_enable_global
	#define logger_enable_global(v) logger::_enable_global_ = v
	#endif

	/**
	 * Enable or disable logging in a single translation unit
	 */
	static bool _enable_translation_uint_ = true;
	#ifndef logger_enable
	#define logger_enable(v) logger::_enable_translation_uint_ = v
	#endif

	/**
	 *
	 */
	#ifndef logger_enable_scope
	#define logger_enable_scope(v) logger::__scope__ scope_logging(v)
	#endif

	/**
	 * The output stream of the logger.
	 */
	extern FILE* _output_stream_;

	/**
	 *
	 */
	enum timestamp_resolution{
		nanosecond, microsecond, millisecond, seconds, minutes, hours
	};
	extern timestamp_resolution _resolution;
	#ifndef logger_timestamp_resolution
	#define logger_timestamp_resolution(v) logger::_resolution = v
	#define logger_nanosecond 	logger::timestamp_resolution::nanosecond
	#define logger_microsecond 	logger::timestamp_resolution::microsecond
	#define logger_millisecond 	logger::timestamp_resolution::millisecond
	#define logger_seconds 		logger::timestamp_resolution::seconds
	#define logger_minutes 		logger::timestamp_resolution::minutes
	#define logger_hours 		logger::timestamp_resolution::hours
	#define logger_days 		logger::timestamp_resolution::days
	#define logger_weeks 		logger::timestamp_resolution::weeks
	#define logger_months 		logger::timestamp_resolution::months
	#define logger_years 		logger::timestamp_resolution::years
	#endif

	/**
	 *
	 */
	enum log_level {
		all, info, error, warning, debug
	};
	#ifndef logger_loglevel
	#define logger_loglevel_all 		logger::log_level::all
	#define logger_loglevel_info 		logger::log_level::info
	#define logger_loglevel_error 		logger::log_level::error
	#define logger_loglevel_warning 	logger::log_level::warning
	#define logger_loglevel_debug		logger::log_level::debug
	#endif

	/**
	 *
	 */
	extern log_level _loglevel_global_;
	#ifndef logger_log_level
	#define logger_log_level(v) logger::_loglevel_global_ = v
	#endif

	/**
	 * Used to enable or disable logging within a certain scope
	 * Affects the states of both global logging and translation unit logging
	 */
	class __scope__ {
		bool value;
		bool previous_global_logger_state;
		bool previous_translation_unit_logger_state;

	public:
		explicit __scope__(bool value) {
			bool current_global_logger_state = logger::_enable_global_;
			bool current_translation_unit_logger_state = logger::_enable_translation_uint_;

			previous_global_logger_state = current_global_logger_state;
			previous_translation_unit_logger_state = current_translation_unit_logger_state;

			logger_enable(value);
			logger_enable_global(value);
			this->value = value;
		}

		~__scope__() {
			logger_enable(previous_translation_unit_logger_state);
			logger_enable_global(previous_global_logger_state);
		}
	};

	/**
	 *
	 */
	__force_inline__
	void __print_timestamp__() {
		const auto duration = std::chrono::system_clock::now().time_since_epoch();
		switch (_resolution) {
			case nanosecond: {
				const uint64_t time =
						std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
				fprintf(_output_stream_, "[%" PRIu64 "]", time);
				break;
			}

			case microsecond: {
				const uint64_t time =
						std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
				fprintf(_output_stream_, "[%" PRIu64 "]", time);
				break;
			}

			case millisecond: {
				const uint64_t time =
						std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
				fprintf(_output_stream_, "[%" PRIu64 "]", time);
				break;
			}

			case seconds: {
				const uint64_t time =
						std::chrono::duration_cast<std::chrono::seconds>(duration).count();
				fprintf(_output_stream_, "[%" PRIu64 "]", time);
				break;
			}

			case minutes: {
				const uint64_t time =
						std::chrono::duration_cast<std::chrono::minutes>(duration).count();
				fprintf(_output_stream_, "[%" PRIu64 "]", time);
				break;
			}

			case hours: {
				const uint64_t time =
						std::chrono::duration_cast<std::chrono::hours>(duration).count();
				fprintf(_output_stream_, "[%" PRIu64 "]", time);
				break;
			}
		}
	}

	/**
	 *
	 */
	__force_inline__
	int __print_timestamp__(char* b) {
		int rv = 0;
		const auto duration = std::chrono::system_clock::now().time_since_epoch();
		switch (_resolution) {
			case nanosecond: {
				const uint64_t time =
						std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
				rv += snprintf(b, 20, "[%" PRIu64 "]", time);
				break;
			}

			case microsecond: {
				const uint64_t time =
						std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
				rv += snprintf(b, 20, "[%" PRIu64 "]", time);
				break;
			}

			case millisecond: {
				const uint64_t time =
						std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
				rv += snprintf(b, 20, "[%" PRIu64 "]", time);
				break;
			}

			case seconds: {
				const uint64_t time =
						std::chrono::duration_cast<std::chrono::seconds>(duration).count();
				rv += snprintf(b, 20, "[%" PRIu64 "]", time);
				break;
			}

			case minutes: {
				const uint64_t time =
						std::chrono::duration_cast<std::chrono::minutes>(duration).count();
				rv += snprintf(b, 20, "[%" PRIu64 "]", time);
				break;
			}

			case hours: {
				const uint64_t time =
						std::chrono::duration_cast<std::chrono::hours>(duration).count();
				rv += snprintf(b, 20, "[%" PRIu64 "]", time);
				break;
			}
		}
		return rv;
	}

	/**
	 * global va_list to optimize for performance on single threads
	 */
	extern va_list __args__;

	/**
	 * The following set of macros take in a string object as argument and return a constant char pointer
	 */
	#ifndef __CPPLOGGER_COLOR__
	#define __CPPLOGGER_COLOR__(color, msg) color+msg;
	#endif
	__force_inline__ string _BLACK_(const string& msg)		{return __CPPLOGGER_COLOR__(ANSI_BLACK, msg);		}
	__force_inline__ string _RED_(const string& msg)		{return __CPPLOGGER_COLOR__(ANSI_RED, msg);			}
	__force_inline__ string _GREEN_(const string& msg)		{return __CPPLOGGER_COLOR__(ANSI_GREEN, msg);		}
	__force_inline__ string _YELLOW_(const string& msg)	{return __CPPLOGGER_COLOR__(ANSI_YELLOW, msg);		}
	__force_inline__ string _BLUE_(const string& msg)		{return __CPPLOGGER_COLOR__(ANSI_BLUE, msg);		}
	__force_inline__ string _PURPLE_(const string& msg)	{return __CPPLOGGER_COLOR__(ANSI_PURPLE, msg);		}
	__force_inline__ string _CYAN_(const string& msg)		{return __CPPLOGGER_COLOR__(ANSI_CYAN, msg);		}
	__force_inline__ string _WHITE_(const string& msg)		{return __CPPLOGGER_COLOR__(ANSI_WHITE, msg);		}
	__force_inline__ string _BOLD_(const string& msg)		{return __CPPLOGGER_COLOR__(ANSI_BOLD, msg);		}
	__force_inline__ string _UNDERLINE_(const string& msg)	{return __CPPLOGGER_COLOR__(ANSI_UNDERLINE, msg);	}
	__force_inline__ string _ITALIC_(const string& msg)	{return __CPPLOGGER_COLOR__(ANSI_ITALIC, msg);		}


	/**
	 *
	 */
	template<typename T>
	string _PRINT_ARRAY_(T* array, size_t size) {
		string returnValue;
		returnValue.reserve(size*10);
		returnValue.append("[");
		for (size_t i=0; i<size; ++i) {
			returnValue.append(std::to_string(array[i]));
			if (i+1 != size) returnValue.append(", ");
		}
		returnValue.append("]");
		return returnValue;
	}
	#ifndef logger_array
	#define logger_array(...) logger::_PRINT_ARRAY_(__VA_ARGS__).c_str()
	#endif

	/**
	 * Wrapper macro boilerplate code for printing logging information to screen
	 */
	#ifndef __CPPLOGGER_PRINT__
	#define __CPPLOGGER_PRINT__(color, type) { \
		if (_print_log_type_) { \
			fputs(color "[" type "]" ANSI_RESET, _output_stream_); \
		} \
		if (_print_timestamps_) { \
			__print_timestamp__(); \
		} \
		if (_print_thread_id_) { \
			std::stringstream ss; ss << std::this_thread::get_id(); \
			const unsigned long long int id = std::stoull(ss.str()); \
			fprintf(_output_stream_, "[%04llu]", id); \
		} \
		if (_print_log_type_ || _print_timestamps_ || _print_thread_id_) { \
			fputs(": ", _output_stream_); \
		} \
		if (_print_file_) { \
			fputs(_file_, _output_stream_); \
			fputs(":", _output_stream_);\
		} \
		if (_print_line_) { \
			fprintf(_output_stream_, "%d:", line); \
		} \
		if (_print_file_ || _print_line_) { \
			fputs(": ", _output_stream_); \
		} \
		vfprintf(_output_stream_, fmt, __args__); \
		fputs(ANSI_RESET "\n", _output_stream_); \
		va_end(__args__); \
	}
	#endif

	/**
	 *
	 */
	#ifndef __CPPLOGGER_MT__
	#define __CPPLOGGER_MT__(color, type) {\
		va_list __args__;\
		va_start(__args__, fmt);\
		std::lock_guard<std::mutex> lock(critical_section);\
		__CPPLOGGER_PRINT__(color, type);\
	}
	#endif

	/**
	 * wrapper macro for the functions provided by the logger
	 */
	#ifndef __CPPLOGGER_SYNC_FUNC__
	#define __CPPLOGGER_SYNC_FUNC__(func_name, mt_func_name, log_color, log_level, log_type)\
	inline void func_name(const char* _file_, const int line, const int cvl, const int avl, const char* fmt, ...) {\
		if (_enable_global_ && _enable_translation_uint_ && cvl >= avl) {\
			if (_loglevel_global_ == all || _loglevel_global_ == log_level) {\
				va_start(__args__, fmt);\
				__CPPLOGGER_PRINT__(log_color, log_type);\
			}\
		}\
	}\
	\
	inline void func_name(const char* _file_, const int line, const char* fmt, ...) {\
		if (_enable_global_ && _enable_translation_uint_) {\
			if (_loglevel_global_ == all || _loglevel_global_ == log_level) {\
				va_start(__args__, fmt);\
				__CPPLOGGER_PRINT__(log_color, log_type);\
			}\
		}\
	}\
	\
	inline void mt_func_name(const char* _file_, const int line, const int cvl, const int avl, const char* fmt, ...) {\
		if (_enable_global_ && _enable_translation_uint_ && cvl >= avl) {\
			if (_loglevel_global_ == all || _loglevel_global_ == log_level) {\
				__CPPLOGGER_MT__(log_color, log_type);\
			}\
		}\
	}\
	\
	inline void mt_func_name(const char* _file_, const int line, const char* fmt, ...) {\
		if (_enable_global_ && _enable_translation_uint_) {\
			if (_loglevel_global_ == all || _loglevel_global_ == log_level) {\
				__CPPLOGGER_MT__(log_color, log_type);\
			}\
		}\
	}
	#endif

	/**
	 * Variadic argument function for printing information logs to screen.
	 */
	#ifndef logger_info
	#define logger_info(...) logger::_info_(__FILE__, __LINE__, __VA_ARGS__)
	#define logger_info_mt(...) logger::_info_mt_(__FILE__, __LINE__, __VA_ARGS__)
	#endif

	__CPPLOGGER_SYNC_FUNC__(_info_, _info_mt_, ANSI_GREEN, info, "INFO")

	/**
	 * Variadic argument function for printing error logs to screen.
	 */
	#ifndef logger_error
	#define logger_error(...) logger::_error_(__FILE__, __LINE__, __VA_ARGS__)
	#define logger_error_mt(...) logger::_error_mt_(__FILE__, __LINE__, __VA_ARGS__)
	#endif

	__CPPLOGGER_SYNC_FUNC__(_error_, _error_mt_, ANSI_RED, error, "ERROR")

	/**
	 * Variadic argument function for printing warning logs to screen.
	 */
	#ifndef logger_warning
	#define logger_warning(...) logger::_warning_(__FILE__, __LINE__, __VA_ARGS__)
	#define logger_warning_mt(...) logger::_warning_mt_(__FILE__, __LINE__, __VA_ARGS__)
	#endif

	__CPPLOGGER_SYNC_FUNC__(_warning_, _warning_mt_, ANSI_PURPLE, warning, "WARNING")


	/**
	 * Variadic argument function for printing debug logs to screen.
	 */
	#ifndef logger_debug
	#define logger_debug(...) logger::_debug_(__FILE__, __LINE__, __VA_ARGS__)
	#define logger_debug_mt(...) logger::_debug_mt_(__FILE__, __LINE__, __VA_ARGS__)
	#endif

	__CPPLOGGER_SYNC_FUNC__(_debug_, _debug_mt_, ANSI_CYAN, debug, "DEBUG")


	/**
	 * The following function are used to get the name of a type from its pointer
	 * or its reference
	 */
	#ifndef logger_get_type
	#define logger_get_type(v) logger::get_type(v).c_str()
	#endif

	template<typename T>
	string get_type(T* c) {
		if (!_enable_global_ || !_enable_translation_uint_) return nullptr;
		auto ptr = std::unique_ptr<char, decltype(& std::free)>{
			abi::__cxa_demangle(typeid(*c).name(), nullptr, nullptr, nullptr),
			std::free
		};
		return string(ptr.get());
	}

	#ifndef __clang__
	template<typename T>
	string get_type(T& c) {
		if (!_enable_global_ || !_enable_translation_uint_) return nullptr;
		auto ptr = std::unique_ptr<char, decltype(& std::free)>{
			abi::__cxa_demangle(typeid(c).name(), nullptr, nullptr, nullptr),
			std::free
		};
		return string(ptr.get());
	}
	#endif

}

#endif /* CPPLOGGER_SYNC_H_ */
