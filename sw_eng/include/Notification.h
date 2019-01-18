#ifndef NOTIFICATION_H
#define NOTIFICATION_H

class Notification {
public:
    std::string getText() {
        return text;
    };

    Notification(std::string _t) : text(_t) { };
    ~Notification() { };
private:
    std::string text;
};

#endif
