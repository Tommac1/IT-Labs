#include <iostream>
#include <fstream> // ofstream
#include <iomanip> // setw

#include <json.hpp> // json lib


#include "Database.h"
#include "Administrator.h"
#include "User.h"
#include "Artifact.h"

int main(int argc, char *argv[])
{
    std::cout << "hello\n";
    Administrator *admin = Administrator::createAdministrator("tomaszek", 
            "bajzel", "tomasz.bajzel@buziaczek.pl");
    Database *db = Database::createDatabase(admin, "database.json");

    User *user1 = admin->createUser("tomaszek", "dupsko123", 
            "tomasz.wpierdol@gmail.com", db, false);

    User *user2 = admin->createUser("asd", "dsa", 
            "asd.dsa@com", db, true);

//    Artifact *art1 = Artifact::createArtifact(Task, "moj nowy artefakt!", 
//            user1, db);

    while (1) { }
        

    std::cout << "jacie\n";

    return 0;
}


