#ifndef USER_H
#define USER_H

#include <string>

#include "Permission.h"
#include "Administrator.h"
#include "Observer.h"
#include "Artifact.h"
#include "Notification.h"

class Administrator;
class Artifact;

class User : public Observer {
public:
    std::string getUsername();
    std::string getPassword();
    std::string getEmail();
    int getId();
    Permission *getPermission();
    std::vector<Notification *> *getNotifications();
    int attachPermission(Permission *perm);

    void addNotification(Artifact *art, std::string &action);
    void receiveNotification();
    void receiveNotifications();

    ~User();
private:
    friend class Administrator;
    User(std::string _u, std::string _p, std::string _e, int _id) 
        : username(_u), password(_p), email(_e), id(_id) { };


    std::string username;
    std::string password;
    std::string email;
    int id;
    Permission *permission = nullptr;
    User *lineManager;
    std::vector<Notification *> notificationBox;
//    Administrator *admin;
};

#endif
