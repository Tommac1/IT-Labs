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
class Label;
class Project;

class Database {
public:
    static Database *createDatabase(Administrator *admin, std::string serverName);
    int addUser(User *user);
    int addArtifact(Artifact *art);
    int addProject(Project *proj);

    int getUsersSize();
    int getArtifactsSize();
    int getProjectsSize();

    void backupNow();

    Administrator *getAdmin();
    void setAdmin(Administrator *new_adm);

private:
    // pointer to singleton guard
    static Database *self;

    std::string serverName;
    std::vector<Project *> projects;
    std::vector<Label *> labels;
    std::vector<Artifact *> artifacts;
    std::vector<User *> users;
    Administrator *admin;
    std::thread *backupThread;

    std::mutex *user_data_mutex;
    std::mutex *projs_data_mutex;
    std::mutex *arts_data_mutex;

    const std::string hash_key = "INZ_OPR_IS_THE_BEST";
    std::string hash(std::string text, std::string key);
    std::string dehash(std::string text, std::string key);

    int importDatabase();
    int importAdmin(const json &data);
    int importUsers(const json &data);
    int importUser(const json &data);

    void backupDatabase();
    int writeToDatabase();
    int writeUsersToDatabase(json &data);
    void writeUserToDatabase(User *user, json &data);
    int writeAdminToDatabase(json &data);
    int writeArtifactsToDatabase(json &data);
    void writeArtifactToDatabase(Artifact *art, json &data);
    int writeProjectsToDatabase(json &data);
    void writeProjectToDatabase(Project *proj, json &data);


    std::thread *spawn();
    Database(Administrator *_admin, std::string _servName) 
        : serverName(_servName), admin(_admin) { };
};

#endif

