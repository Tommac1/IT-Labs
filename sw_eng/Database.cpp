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
    std::this_thread::sleep_for(std::chrono::seconds(2));

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
        std::lock_guard<std::mutex> lock(*data_mutex);
        users.push_back(user);
    }
    else {
        std::cout << "User aleary exist in database.\n";
        result = -1;
    }

    return result;
}

int Database::addArtifact(Artifact *art) 
{
    int result = 0;
    std::string name = art->getName();

    auto it = std::find_if(artifacts.begin(), artifacts.end(), 
            [&name](Artifact* obj) { return name == obj->getName(); } );

    if (it == artifacts.end()) {
        std::lock_guard<std::mutex> lock(*data_mutex);
        artifacts.push_back(art);
    }
    else {
        std::cout << "Artifact aleary exist in database.\n";
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
    itemsWritten += writeArtifactsToDatabase(data);

    std::ofstream os(serverName);
    os << std::setw(4) << data << std::endl;

    return itemsWritten;
}

int Database::writeArtifactsToDatabase(json &data)
{
    int artifactsWritten = 0;
    User *user; 
    Artifact *parent;
    int id = 0;

    for (auto &artifact : artifacts) {
        id = artifact->getId();
        data["Artifacts"][std::to_string(id)]["Name"] = artifact->getName();
        data["Artifacts"][std::to_string(id)]["Type"] = ArtifactTypeString[artifact->getType()];
        data["Artifacts"][std::to_string(id)]["Status"] = ArtifactStatusString[artifact->getStatus()];
        if (nullptr != (user = artifact->getOwner()))
            data["Artifacts"][std::to_string(id)]["Owner"] = user->getId();
        if (nullptr != (user = artifact->getCreator()))
            data["Artifacts"][std::to_string(id)]["Creator"] = user->getId();
        if (nullptr != (parent = artifact->getParent()))
            data["Artifacts"][std::to_string(id)]["Parent"] = parent->getId();

        artifactsWritten++;
    }

    return artifactsWritten;    
}

int Database::writeAdminToDatabase(json &data)
{
    int result = 0;
    if (admin != nullptr) {
        std::string pass_hashed = hash(admin->getPassword(), hash_key);
        data["Admin"]["Username"] = admin->getUsername();
        data["Admin"]["Password"] = pass_hashed;
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
        writeUserToDatabase(user, data);
        usersWritten++;
    }

    return usersWritten;    
}

void Database::writeUserToDatabase(User *user, json &data)
{
    int id = user->getId();
    std::string pass_hashed = hash(user->getPassword(), hash_key);
    data["Users"][std::to_string(id)]["Username"] = user->getUsername();
    data["Users"][std::to_string(id)]["Password"] = pass_hashed;
    data["Users"][std::to_string(id)]["Email"] = user->getEmail();
    data["Users"][std::to_string(id)]["Manager"] = 
        (user->getPermission() != nullptr) ? true : false;
}

std::string Database::hash(std::string text, std::string key)
{
    key = generateKey(text, key); 
    return cipherText(text, key);
}

int Database::getUsersSize()
{
    std::lock_guard<std::mutex> lock(*data_mutex);
    return users.size();
}

int Database::getArtifactsSize()
{
    std::lock_guard<std::mutex> lock(*data_mutex);
    return artifacts.size();
}
