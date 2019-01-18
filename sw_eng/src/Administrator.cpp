#include "Administrator.h"

Administrator *Administrator::self = nullptr;

Administrator *Administrator::createAdministrator(std::string username,
            std::string password, std::string email)
{
    if (self == nullptr) {
        self = new Administrator(username, password, email);
    }
    return self;
}

User *Administrator::createUser(std::string username, 
        std::string password, std::string email, Database *db, bool manager)
{
    User *new_user = nullptr;

    int validation = validateUserCredentials(username, password, email);
    if (!validation) {
        new_user = new User(username, password, email);
        if (new_user != nullptr) {
            if (manager)
                givePermission(new_user);
            db->addUser(new_user);
        }
    }

    return new_user;
}

int Administrator::validateUserCredentials(std::string username,
        std::string password, std::string email)
{
    int result = 0;
    const std::string email_pattern = "^([a-zA-Z0-9_.\\-])+@([a-zA-Z0-9_.\\-])+\\.([a-zA-Z])+$";
    const std::string username_pattern = "^([a-zA-Z0-9_\\-])+$";
    std::regex username_regex(username_pattern);

    std::smatch sm;
    if (!std::regex_search(username, sm, username_regex)) {
        result = -1;
        std::cout << "username match failed: " << username << "\n";
    }

    std::regex email_regex(email_pattern);

    if (!std::regex_search(email, sm, email_regex)) {
        result = -1;
        std::cout << "email match failed: " << email << "\n";
    }

    return result;
}

void Administrator::diagnose(Database *db)
{
    int users = db->getUsersSize();
    int arts = db->getArtifactsSize();
    int projs = db->getProjectsSize();

    std::cout << "Diagnosis started... ===========================\n";

    std::cout << "Users count: " << users << std::endl;
    std::cout << "Artifacts count: " << arts << std::endl;
    std::cout << "Projects count: " << projs << std::endl;

    auto begin = std::chrono::steady_clock::now();
    db->backupNow();
    auto end = std::chrono::steady_clock::now();

    std::cout << "Backupping took: " 
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() 
              << "ms" << std::endl;
}

int Administrator::givePermission(User *user)
{
    int result = 0;
    Permission *perm = new Permission();
    if (!perm || user->attachPermission(perm)) {
        delete perm;
        result = -1;
    }
    
    return result;
}

std::string Administrator::getPassword()
{
    return password;
}

std::string Administrator::getUsername()
{
    return username;
}

std::string Administrator::getEmail()
{
    return email;
}
