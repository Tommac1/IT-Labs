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

    admin->diagnose(db);

    User *user1 = admin->createUser("manager1", "dupsko123", 
            "tomasz.wpierdol@gmail.com", db, true);
    User *user2 = admin->createUser("dev1", "dsa", 
            "asd.dsa@com.pl", db, false);
    User *user3 = admin->createUser("manager2", "asdas", 
            "andrew.andrew@andrew.andrew", db, true);
    User *user4 = admin->createUser("dev2", "3123412njlsdfnsd", 
            "cos@cos.cos", db, false);
    User *user5 = admin->createUser("dev3", "dupko", 
            "andrew@andrew.andrew", db, false);
    User *user6 = admin->createUser("dev3", "dupko", 
            "andrew.andrew", db, false);

    Project *proj1 = user1->createProject("pierwszy", db);
    Project *proj2 = user3->createProject("drugi", db);

    proj1->addEngineer(user2);
    proj1->addEngineer(user4);
    proj1->addEngineer(user5);

    Artifact *func1 = Artifact::createArtifact(AT_Functionality, "funkcja #1", 
            user1, proj1, db);

    Artifact *func2 = Artifact::createArtifact(AT_Functionality, "funckja #2",
            user1, proj1, db);

    Artifact *story1 = Artifact::createArtifact(AT_Story, "historia #1", 
            user1, proj1, db, func1);
    
    Artifact *story2 = Artifact::createArtifact(AT_Story, "historia #2", 
            user3, proj1, db, func2);

    Artifact *story3 = Artifact::createArtifact(AT_Story, "historia #3", 
            user1, proj1, db, func2);

    Artifact *task1 = Artifact::createArtifact(AT_Task, "tasak #1", 
            user1, proj1, db, story1);

    Artifact *task2 = Artifact::createArtifact(AT_Task, "tasak #2", 
            user2, proj1, db, story1);

    Artifact *func3 = Artifact::createArtifact(AT_Functionality, "funckja cos", 
            user3, proj2, db);

    Artifact *story4 = Artifact::createArtifact(AT_Story, "hist #4", 
            user3, proj2, db, func3);

    Artifact *task3 = Artifact::createArtifact(AT_Task, "tasak #3", 
            user3, proj2, db, story4);

    task1->addSubscriber(user3);

    task1->setOwner(user2);

    story1->setStatus(AS_Done);

    // user1->receiveNotifications();
    // user2->receiveNotifications();

    admin->diagnose(db);

    while (1) { }

    UNREF(user1);
    UNREF(user2);
    UNREF(user3);
    UNREF(proj1);
    UNREF(proj2);
    UNREF(func1);
    UNREF(func2);
    UNREF(func3);
    UNREF(story1);
    UNREF(story2);
    UNREF(story3);
    UNREF(story4);
    UNREF(task1);
    UNREF(task2);
    UNREF(task3);

    std::cout << "jacie\n";

    return 0;
}


