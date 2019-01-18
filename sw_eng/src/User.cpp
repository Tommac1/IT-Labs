#include "User.h"

int User::usersInSystem = 0;

int User::attachPermission(Permission *perm) 
{
    int result = -1;
    if (permission == nullptr) {
        permission = perm;
        result = 0;
    }
    return result;
}

Project *User::createProject(const std::string &name, Database *db)
{
    Project *proj = Project::createProject(name, this);

    if (proj) {
        if (db->addProject(proj)) {
            delete proj;
            proj = nullptr;
        }
    }

    return proj;
}

void User::addNotification(Project *proj, const std::string &action)
{
    std::string text = "";
    if (proj != nullptr)
        text = proj->getName() + ": ";
    text += action;
    Notification *notif = new Notification(text);
    notificationBox.push_back(notif);
}

void User::addNotification(Artifact *art, const std::string &action)
{
    std::string text = "";
    if (art != nullptr)
        text = art->getName() + ": ";
    text += action;
    Notification *notif = new Notification(text);
    notificationBox.push_back(notif);
}

void User::addNotification(std::string &action)
{
    Notification *notif = new Notification(action);
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

int User::getUsersInSystem()
{
    return usersInSystem;
}
