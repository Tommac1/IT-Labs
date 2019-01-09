#include <iostream>
#include <fstream> // ofstream
#include <iomanip> // setw

#include <json.hpp> // json lib


#include "Database.h"
#include "Administrator.h"
#include "User.h"
#include "Artifact.h"
#include "Utils.h"

#define UNREF(x) (void)x

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

    Artifact *art1 = Artifact::createArtifact(Story, "moj nowy artefakt!", 
            user1, db);

    Artifact *art2 = Artifact::createArtifact(Story, "moje nowe story", 
            user2, db, art1);

    while (1) { }

    UNREF(user1);
    UNREF(user2);
    UNREF(art1);
    UNREF(art2);

    std::cout << "jacie\n";

    return 0;
}


