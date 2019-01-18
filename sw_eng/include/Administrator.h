#ifndef ADMINISTRATOR_H
#define ADMINISTRATOR_H

#include <iostream>
#include <string>
#include <regex>
#include <chrono>

#include "Database.h"
#include "User.h"
#include "Permission.h"

class Database;


class Administrator {
public:
    int givePermission(User *user);
    User *createUser(std::string username, 
        std::string password, std::string email, Database *db, bool manager);

    std::string getUsername();
    std::string getEmail();
    std::string getPassword();

    void diagnose(Database *db);

    static Administrator *createAdministrator(std::string username,
            std::string password, std::string email);

    ~Administrator();
private:
    friend class Database;

    Administrator(std::string u, std::string p, std::string e)
            : username(u), password(p), email(e) { };

    static Administrator *self;

    std::string username;
    std::string password;
    std::string email;

    int validateUserCredentials(std::string username,
            std::string password, std::string email);
};

#endif
