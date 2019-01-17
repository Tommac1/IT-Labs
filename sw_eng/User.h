#ifndef USER_H
#define USER_H

#include <string>

#include "Permission.h"
#include "Administrator.h"
#include "Observer.h"
#include "Artifact.h"
#include "Notification.h"
#include "Project.h"

class Administrator;
class Artifact;

class User : public Observer {
public:
    std::string getUsername() override;
    std::string getPassword();
    std::string getEmail();
    int getId() override;
    Permission *getPermission();
    std::vector<Notification *> *getNotifications();
    static int getUsersInSystem();
    int attachPermission(Permission *perm);

    void addNotification(std::string &action) override;
    void addNotification(Artifact *art, const std::string &action) override;
    void addNotification(Project *proj, const std::string &action) override;
    void receiveNotification();
    void receiveNotifications();

    Project *createProject(const std::string &name, Database *db);

    ~User() { };
private:
    friend class Administrator;
    friend class Database;
    User(std::string _u, std::string _p, std::string _e) 
        : username(_u), password(_p), email(_e) { 
        id = usersInSystem++;    
    };


    std::string username;
    std::string password;
    std::string email;
    int id;
    Permission *permission = nullptr;
    User *lineManager;
    std::vector<Notification *> notificationBox;
//    Administrator *admin;
    static int usersInSystem;
};

#endif
