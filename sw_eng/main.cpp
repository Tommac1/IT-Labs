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

    Artifact *art1 = Artifact::createArtifact(AT_Story, "moje nowe story!!", 
            user1, db);

    Artifact *art2 = Artifact::createArtifact(AT_Task, "moj nowy tasak!", 
            user2, db, art1);

    art2->addSubscriber(user1);
    art2->addSubscriber(user1);

    art1->setStatus(AS_Analysis);
    art2->setStatus(AS_Implementation);

   // user1->receiveNotifications();
   // user2->receiveNotifications();

    while (1) { }

    UNREF(user1);
    UNREF(user2);
    UNREF(art1);
    UNREF(art2);

    std::cout << "jacie\n";

    return 0;
}


