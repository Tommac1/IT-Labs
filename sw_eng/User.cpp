#include "User.h"

int User::attachPermission(Permission *perm) 
{
    int result = -1;
    if (permission == nullptr) {
        permission = perm;
        result = 0;
    }
    return result;
}

std::string User::getUsername()
{
    return username;
}

std::string User::getPassword()
{
    return password;
}

int User::getId()
{
    return id;
}

std::string User::getEmail()
{
    return email;
}

Permission *User::getPermission()
{
    return permission;
}
