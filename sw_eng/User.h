#ifndef USER_H
#define USER_H

#include <string>

#include "Permission.h"
#include "Administrator.h"

class Administrator;

class User {
public:
    std::string getUsername();
    std::string getPassword();
    std::string getEmail();
    int getId();
    Permission *getPermission();
    int attachPermission(Permission *perm);

    ~User();
private:
    friend class Administrator;
    User(std::string _u, std::string _p, std::string _e, int _id) 
        : username(_u), password(_p), email(_e), id(_id) { };

    std::string username;
    std::string password;
    std::string email;
    int id;
    Permission *permission;
    User *lineManager;
//    std::vector<Notification *> notificationBox;
//    Administrator *admin;
};

#endif
