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

    Database *db = Database::createDatabase(nullptr, "database.json");
    Administrator *admin = Administrator::createAdministrator("tomaszek", 
            "bajzel", "tomasz.bajzel@buziaczek.pl");
    db->setAdmin(admin);

    while (1) { }

    User *user1 = admin->createUser("tomaszek", "dupsko123", 
            "tomasz.wpierdol@gmail.com", db, false);
    User *user2 = admin->createUser("asd", "dsa", 
            "asd.dsa@com.pl", db, true);
    User *user3 = admin->createUser("andrew", "shotgun", 
            "andrew.andrew@andrew.andrew", db, false);

    Project *proj1 = user2->createProject("Projekt czegos tam", db);

    Artifact *art1 = Artifact::createArtifact(AT_Story, "moje nowe story!!", 
            user1, proj1, db);

    Artifact *art2 = Artifact::createArtifact(AT_Task, "moj nowy tasak!", 
            user2, proj1, db, art1);

    Artifact *art3 = Artifact::createArtifact(AT_Functionality, "funkcjonalnosc zalozona przez non-managera!", 
            user3, proj1, db, art1);

    art2->addSubscriber(user1);
    art2->addSubscriber(user1);
    art1->addSubscriber(user1);
    art2->addSubscriber(user3);

    art1->setOwner(user2);

    art1->setStatus(AS_Analysis);
    art2->setStatus(AS_Implementation);
    art2->setStatus(AS_Analysis);
    art2->setStatus(AS_Implementation);
    art2->setStatus(AS_Validation);
    art2->setStatus(AS_Done);

   // user1->receiveNotifications();
   // user2->receiveNotifications();


    UNREF(user1);
    UNREF(user2);
    UNREF(user3);
    UNREF(proj1);
    UNREF(art1);
    UNREF(art2);
    UNREF(art3);

    std::cout << "jacie\n";

    return 0;
}


