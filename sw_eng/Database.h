#ifndef DATABASE_H
#define DATABASE_H

#include <iostream>
#include <fstream> // ofstream
#include <iomanip> // setw
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <mutex>
#include <algorithm>

#include <json.hpp> // json lib

#include "Administrator.h"
#include "Label.h"
#include "User.h"
#include "Artifact.h"
#include "Project.h"
#include "Utils.h"

using json = nlohmann::json;

class Administrator;
class Artifact;

class Database {
public:
    static Database *createDatabase(Administrator *admin, std::string serverName);
    int addUser(User *user);
    int addArtifact(Artifact *art);

    int getUsersSize();
    int getArtifactsSize();

private:
    // pointer to singleton guard
    static Database *self;
    std::string serverName;
//    std::vector<Project *> projects;
    std::vector<Label *> labels;
    std::vector<Artifact *> artifacts;
    std::vector<User *> users;
    Administrator *admin;
    std::thread *backupThread;
    std::mutex *data_mutex;

    const std::string hash_key = "INZ_OPR_IS_THE_BEST";
    std::string hash(std::string text, std::string key);

    void backupDatabase();
    int writeToDatabase();
    int writeUsersToDatabase(json &data);
    void writeUserToDatabase(User *user, json &data);
    int writeAdminToDatabase(json &data);
    int writeArtifactsToDatabase(json &data);

    std::thread *spawn();
    Database(Administrator *_admin, std::string _servName) : serverName(_servName), admin(_admin) { };
};

#endif

