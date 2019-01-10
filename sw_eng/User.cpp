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

void User::addNotification(Artifact *art, std::string &action)
{
    std::string text = art->getName() + ": " + action;
    Notification *notif = new Notification(text);
    notificationBox.push_back(notif);
}

void User::receiveNotification()
{
    std::cout << "User (" << getId() << ") " 
        << getUsername() << " notification: \n\t\""
        << notificationBox.back()->getText() << "\"\n";
    notificationBox.pop_back();
}

void User::receiveNotifications()
{
    while (notificationBox.size() != 0)
        receiveNotification();
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

std::vector<Notification *> *User::getNotifications()
{
    return &notificationBox;
}
