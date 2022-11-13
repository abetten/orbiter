/**
* Author:    Sajeeb Roy Chowdhury
* Created:   11/10/22
* Email:     sajeeb.roy.chow@gmail.com
*
**/

#include <stdexcept>
#include <string>

#ifndef EXCEPTION_H
#define EXCEPTION_H

class not_implemented : public std::exception {
    std::string message;
public:
    not_implemented(const std::string& message) : message(message) {}
    const char* what () const noexcept {
        return message.c_str();
    }
};

#endif //EXCEPTION_H
