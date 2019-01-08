#ifndef ADMINISTRATOR_H
#define ADMINISTRATOR_H

#include <iostream>
#include <string>
#include <regex>

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

    static Administrator *createAdministrator(std::string username,
            std::string password, std::string email);

    ~Administrator();
private:
    Administrator(std::string username,
            std::string password, std::string email);

    static Administrator *self;

    std::string username;
    std::string password;
    std::string email;

    int validateUserCredentials(std::string username,
            std::string password, std::string email);
};

#endif
