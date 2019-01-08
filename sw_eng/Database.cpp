#include "Database.h"

Database *Database::self = nullptr;

Database *Database::createDatabase(Administrator *admin, std::string serverName)
{
    if (self == nullptr) {
        self = new Database(admin, serverName);
        self->backupThread = self->spawn();
        self->data_mutex = new std::mutex();
    }
    return self;
}

std::thread *Database::spawn() 
{
    return new std::thread ([=] { backupDatabase(); });
}

void Database::backupDatabase()
{
    while (1) {
        int result = writeToDatabase();

        std::cout << "Backup done. Items backupped: " << result << std::endl;

        std::this_thread::sleep_for(std::chrono::seconds(5));
    }
}

int Database::addUser(User *user)
{
    int result = 0;
    std::string username = user->getUsername();

    auto it = std::find_if(users.begin(), users.end(), 
            [&username](User* obj) { return username == obj->getUsername(); } );

    if (it == users.end()) {
        users.push_back(user);
    }
    else {
        std::cout << "User aleary exist in database.\n";
        result = -1;
    }

    return result;
}

int Database::writeToDatabase() 
{
    std::lock_guard<std::mutex> lock(*data_mutex);
    int itemsWritten = 0;

    json data;

    itemsWritten = writeAdminToDatabase(data);
    itemsWritten += writeUsersToDatabase(data);

    std::ofstream os(serverName);
    os << std::setw(4) << data << std::endl;

    return itemsWritten;
}

int Database::writeArtifactsToDatabase(json &data)
{
    int artifactsWritten = 0;
//    int id = 0;

    for (auto &artifact : artifacts) {
//        id = artifact->getId();
//        data["Artifacts"][std::to_string(id)]["Name"] = artifact->getName();
//        data["Artifacts"][std::to_string(id)]["Owner"] = artifact->getOwner();
//        data["Artifacts"][std::to_string(id)]["Creator"] = artifact->getCreator();

        artifactsWritten++;
    }

    return artifactsWritten;    
}

int Database::writeAdminToDatabase(json &data)
{
    int result = 0;
    if (admin != nullptr) {
        data["Admin"]["Username"] = admin->getUsername();
        data["Admin"]["Password"] = admin->getPassword();
        data["Admin"]["Email"] = admin->getEmail();
        result++;
    }
    return result;
}

int Database::writeUsersToDatabase(json &data)
{
    int usersWritten = 0;
    int id = 0;

    for (auto &user : users) {
        id = user->getId();
        data["users"][std::to_string(id)]["Username"] = user->getUsername();
        data["users"][std::to_string(id)]["Password"] = user->getPassword();
        data["users"][std::to_string(id)]["Email"] = user->getEmail();
        data["users"][std::to_string(id)]["Manager"] = 
            (user->getPermission() != nullptr) ? true : false;

        usersWritten++;
    }

    return usersWritten;    
}

int Database::getUsersSize()
{
    return users.size();
}

int Database::getArtifactsSize()
{
    return artifacts.size();
}
